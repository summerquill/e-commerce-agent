"""clarify 节点：缺信息追问（独立节点，跨 step 级，与具体 tool 无关）。

interrupt 暂停等用户回答 → 写入 clarified_answer → 回 executor 重试同一 step。
纯 HITL 中继节点，不调 LLM：问题与候选由 executor 解析动作时生成，
存在 clarify_question / clarify_options，此处仅透传进 interrupt payload。
"""

from __future__ import annotations

from langgraph.types import interrupt

from app.plan_execute.state import AgentState


def clarify_node(state: AgentState) -> dict:
    question = state.get("clarify_question") or "请补充更多信息"
    payload: dict = {"type": "clarify", "question": question}
    options = state.get("clarify_options") or []
    if options:
        payload["options"] = list(options)
    answer = interrupt(payload)
    return {"clarified_answer": answer.get("text", "")}
