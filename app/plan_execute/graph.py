"""Plan-and-Execute 图装配：节点注册 + 连线。

节点实现已拆分到 app/plan_execute/nodes/（纯函数 + NodeDeps 注入）：
- planner → executor ⇄ review → generate；clarify 由 review 路由触发
- confirm 是 executor 内部 interrupt（高风险操作审批）
- clarify 是独立节点（缺信息追问），interrupt 后回 executor 重试同 step
- review 是路由中枢，4 出口：done/continue/clarify/replan
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from app.plan_execute.checkpointer import build_checkpointer
from app.plan_execute.nodes import (
    NodeDeps,
    clarify_node,
    executor_node,
    generate_node,
    planner_node,
    review_node,
)
from app.plan_execute.state import AgentState


class PlanExecuteGraph:
    """持有依赖并装配图。

    节点是 nodes/ 下的纯函数，通过 functools.partial 绑定 NodeDeps 后注册，
    替代旧的类绑定方法（拆文件后依赖仍显式、可测试）。
    """

    def __init__(
        self,
        client,
        model: str,
        tool_manager,
        skill_manager=None,
        memory_manager=None,
        temperature: float = 0.7,
    ):
        self.deps = NodeDeps(
            client=client,
            model=model,
            temperature=temperature,
            tool_manager=tool_manager,
            skill_manager=skill_manager,
            memory_manager=memory_manager,
        )

    def build(self):
        """构建并编译 StateGraph。"""
        g = StateGraph(AgentState)
        g.add_node("planner", partial(planner_node, deps=self.deps))
        g.add_node("executor", partial(executor_node, deps=self.deps))
        g.add_node("review", partial(review_node, deps=self.deps))
        g.add_node("clarify", clarify_node)  # 纯 interrupt 节点，无外部依赖
        g.add_node("generate", partial(generate_node, deps=self.deps))

        g.add_edge(START, "planner")
        g.add_edge("planner", "executor")
        g.add_edge("executor", "review")
        g.add_edge("clarify", "executor")  # 回 executor 重试同 step
        g.add_edge("generate", END)

        g.add_conditional_edges("review", self._route, {
            "continue": "executor",
            "clarify": "clarify",
            "replan": "planner",
            "done": "generate",
        })

        return g.compile(checkpointer=build_checkpointer())

    @staticmethod
    def _route(state: AgentState) -> str:
        return state["review_decision"]
