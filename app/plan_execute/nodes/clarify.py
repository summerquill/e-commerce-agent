"""clarify 节点：缺信息追问（独立节点，跨 step 级，与具体 tool 无关）。

读 executor 写入的 pending_clarification → interrupt 暂停等用户回答 →
把问答写入 messages（对话记录，append-only）→ 清空 pending → 回 executor
重试同一 step（由 executor 从对话记录解析参数，解析结果落 step_results）。
纯 HITL 中继节点，不调 LLM。
"""

from __future__ import annotations

from langgraph.types import interrupt

from app.plan_execute.state import AgentState


def clarify_node(state: AgentState) -> dict:
    pending = state.get("pending_clarification") or {}
    question = pending.get("question") or "请补充更多信息"

    payload: dict = {"type": "clarify", "question": question}
    options = pending.get("options") or []
    if options:
        payload["options"] = list(options)

    answer = interrupt(payload)
    text = answer.get("text", "")

    # 问答进对话记录（后续轮次的回答在 messages 中顺序覆盖语义），
    # pending 清空表示当前交互已关闭
    return {
        "pending_clarification": None,
        "messages": [
            {"role": "assistant", "content": question},
            {"role": "user", "content": text},
        ],
    }
