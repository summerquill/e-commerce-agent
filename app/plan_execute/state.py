"""Plan-and-Execute 图的共享状态定义。

设计要点（详见 project_memory.md "State 设计" 小节）：
- plan 与 step_results 分离：plan 是意图（replan 时整体覆盖），step_results 是执行日志
- Step 是描述型（不含 args），executor 执行时动态解析 {tool, args}
- messages 保留以便后续做上下文压缩（add_messages 支持按 ID 替换/删除）
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages


class Step(TypedDict):
    """计划中的一个步骤（描述型，不含具体 args）。

    args 由 executor 执行时根据 step_results + clarified_answer 动态解析，
    因此 plan 阶段无法填出 args（依赖关系由 LLM 隐式解析）。
    """

    id: str                       # "step_1"
    description: str              # "查询用户最近的 AirPods 订单"
    tool_hint: str | None = None  # 可选工具提示, e.g. "query_orders"


class StepResult(TypedDict):
    """单个步骤的执行结果。"""

    step_id: str
    tool: str | None
    input: dict
    output: Any
    status: str       # success | failed | skipped | needs_clarify
    error: str | None = None


class AgentState(TypedDict):
    """Plan-and-Execute 图的共享状态。"""

    # —— 输入 ——
    user_query: str
    messages: Annotated[list, add_messages]  # 节点间 LLM 推理上下文 + 压缩伏笔

    # —— 规划 ——
    plan: list[Step]        # 当前计划, replan 时整体覆盖（默认 reducer = 覆写）
    plan_iteration: int     # 第几版计划, 防死循环

    # —— 执行 ——
    step_results: Annotated[list[StepResult], operator.add]  # 累积 append
    current_step_idx: int

    # —— 路由 ——
    review_decision: str     # done | continue | clarify | replan
    clarify_question: str    # review 设置 → clarify 读取
    clarified_answer: str     # clarify interrupt resume 后写入

    # —— 输出 ——
    final_response: str
