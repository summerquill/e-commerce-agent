"""clarify 节点：缺信息追问（独立节点，跨 step 级，与具体 tool 无关）。

interrupt 暂停等用户回答 → 写入 clarified_answer → 回 executor 重试同一 step。
纯 HITL 节点，不调 LLM：问题由 executor 解析动作时生成，存在 clarify_question。
"""

from __future__ import annotations

from langgraph.types import interrupt

from app.plan_execute.state import AgentState


def clarify_node(state: AgentState) -> dict:
    question = state.get("clarify_question") or "请补充更多信息"
    answer = interrupt({"type": "clarify", "question": question})
    return {"clarified_answer": answer.get("text", "")}
