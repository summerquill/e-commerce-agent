"""Plan-and-Execute 图构建：节点定义 + 图连线。

设计（详见 project_memory.md）：
- 5 节点：planner → executor ⇄ review → generate；clarify 由 review 路由触发
- confirm 是 executor 内部 interrupt（高风险操作审批）
- clarify 是独立节点（缺信息追问），interrupt 后回 executor 重试同 step
- review 是路由中枢，4 出口：done/continue/clarify/replan
- review 双路径：确定性信号走规则快路径，failed→replan 走 LLM（TODO）

骨架阶段：控制流真实可用，LLM 调用部分以 TODO stub 占位，
可跑通图编译与冒烟测试，待后续填入真实 LLM 逻辑。
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt

from app.config.settings import settings
from app.plan_execute.checkpointer import build_checkpointer
from app.plan_execute.prompts import (
    EXECUTOR_PROMPT,
    GENERATE_PROMPT,
    PLANNER_PROMPT,
)
from app.plan_execute.state import AgentState, Step, StepResult

# 高风险工具白名单：执行前触发 confirm interrupt
# TODO: 填入真实工具名, e.g. {"refund", "cancel_order", "modify_address"}
HIGH_RISK_TOOLS: set[str] = set()


class PlanExecuteGraph:
    """持有依赖、方法即节点（LangGraph 常见模式）。

    节点方法是绑定方法，直接作为 graph node 注册，
    天然捕获 self.client / self.tool_manager 等依赖。
    """

    def __init__(self, client, model: str, tool_manager, skill_manager=None):
        self.client = client
        self.model = model
        self.tool_manager = tool_manager
        self.skill_manager = skill_manager

    # ──────────────────── 节点 ────────────────────

    def planner(self, state: AgentState) -> dict:
        """规划：把用户请求拆解为有序步骤列表。

        replan 时（plan_iteration>0）读旧 plan + step_results 作上下文重规划。
        """
        is_replan = state.get("plan_iteration", 0) > 0
        # TODO: 调 LLM(structured output) 生成 list[Step]
        #   context = user_query + (replan 时) 旧 plan + step_results + clarified_answer
        #   注入 SkillManager.build_catalog_prompt() + MemoryManager 上下文
        steps = self._stub_plan(state["user_query"], is_replan)
        iteration = state.get("plan_iteration", 0) + 1
        return {
            "plan": steps,
            "plan_iteration": iteration,
            "messages": [{"role": "assistant", "content": f"[planner v{iteration}] 计划已生成"}],
        }

    def executor(self, state: AgentState) -> dict:
        """执行单个 step：解析 {tool,args} → (高风险则 confirm) → 执行工具。

        每次只跑 plan[current_step_idx]，跑完交棒 review。
        缺信息时返回 needs_clarify 不前进 idx。
        """
        idx = state["current_step_idx"]
        step: Step = state["plan"][idx]

        # TODO: 调 LLM 解析 step.description + 历史 step_results + clarified_answer
        #       → {tool, args} 或 {needs_clarify: true, question}
        resolved = self._stub_resolve_step(step, state)

        if resolved.get("needs_clarify"):
            return {
                "step_results": [StepResult(
                    step_id=step["id"], tool=None, input={},
                    output=None, status="needs_clarify", error=None,
                )],
                "clarify_question": resolved["question"],
            }

        tool = resolved.get("tool")
        args = resolved.get("args", {})

        # confirm: 高风险工具执行前 interrupt 等用户确认
        if tool in HIGH_RISK_TOOLS:
            approval = interrupt({
                "type": "confirm",
                "action": step["description"],
                "context": args,
            })
            if not approval.get("approved"):
                return {
                    "step_results": [StepResult(
                        step_id=step["id"], tool=tool, input=args,
                        output=None, status="skipped", error="用户拒绝确认",
                    )],
                    "current_step_idx": idx + 1,
                }

        # TODO: 真实工具执行；骨架阶段跳过实际调用
        if resolved.get("skip"):
            output: Any = "TODO: 真实工具执行结果"
            status = "success"
            error = None
        else:
            output = self.tool_manager.execute_tool(tool, args)
            output_str = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            # TODO: 根据 output 判断 success/failed
            status = "success"
            error = None
            output = output_str

        return {
            "step_results": [StepResult(
                step_id=step["id"], tool=tool, input=args,
                output=output, status=status, error=error,
            )],
            "current_step_idx": idx + 1,
        }

    def review(self, state: AgentState) -> dict:
        """路由中枢：评估执行状态，决定下一步去向。

        快路径（规则，不调 LLM）：
          - needs_clarify → clarify
          - 全部步骤完成 → done
          - 超 plan_iterations 上限 → done（防死循环）
        慢路径（TODO 调 LLM）：
          - failed → 判断 replan or done
        """
        results = state.get("step_results", [])
        last = results[-1] if results else None

        # 快路径：确定性信号
        if last and last["status"] == "needs_clarify":
            return {"review_decision": "clarify"}
        if state["current_step_idx"] >= len(state["plan"]):
            return {"review_decision": "done"}
        if state.get("plan_iteration", 0) >= settings.max_plan_iterations:
            return {"review_decision": "done"}

        # 慢路径：上一步失败 → 是否 replan
        if last and last["status"] == "failed":
            # TODO: 调 LLM 判断该重规划还是直接兜底回复
            return {"review_decision": "done"}

        return {"review_decision": "continue"}

    def clarify(self, state: AgentState) -> dict:
        """缺信息追问：interrupt 暂停等用户回答，写入 clarified_answer。

        resume 后回 executor 重试同一个 step（idx 未前进）。
        """
        question = state.get("clarify_question", "请补充信息")
        answer = interrupt({"type": "clarify", "question": question})
        return {"clarified_answer": answer.get("text", "")}

    def generate(self, state: AgentState) -> dict:
        """综合所有 step_results 生成最终客服回复。"""
        # TODO: 调 LLM 用 user_query + plan + step_results 生成回复
        reply = self._stub_generate(state)
        return {
            "final_response": reply,
            "messages": [{"role": "assistant", "content": reply}],
        }

    # ──────────────────── 路由 ────────────────────

    def _route(self, state: AgentState) -> str:
        return state["review_decision"]

    # ──────────────────── 构图 ────────────────────

    def build(self):
        """构建并编译 StateGraph。"""
        g = StateGraph(AgentState)
        g.add_node("planner", self.planner)
        g.add_node("executor", self.executor)
        g.add_node("review", self.review)
        g.add_node("clarify", self.clarify)
        g.add_node("generate", self.generate)

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

    # ──────────────────── TODO stubs ────────────────────
    # 以下为骨架占位，待接入真实 LLM 调用后替换

    def _stub_plan(self, user_query: str, is_replan: bool) -> list[Step]:
        """骨架占位：返回单步计划。"""
        tag = "重规划" if is_replan else "规划"
        return [Step(id="step_1", description=f"[{tag}TODO] 处理: {user_query}", tool_hint=None)]

    def _stub_resolve_step(self, step: Step, state: AgentState) -> dict:
        """骨架占位：跳过真实工具调用。"""
        return {"tool": None, "args": {}, "skip": True}

    def _stub_generate(self, state: AgentState) -> str:
        """骨架占位：返回占位回复。"""
        results = state.get("step_results", [])
        n = len(results)
        return f"[TODO: 真实生成] 已完成 {n} 个步骤的执行。"
