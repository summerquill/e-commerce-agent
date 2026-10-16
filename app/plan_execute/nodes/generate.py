"""generate 节点：综合 user_query + plan + 全部 step_results 生成最终客服回复。"""

from __future__ import annotations

from app.plan_execute.nodes.context import (
    NodeDeps,
    chat_completion,
    format_step_results,
)
from app.plan_execute.prompts import GENERATE_PROMPT
from app.plan_execute.state import AgentState


def generate_node(state: AgentState, deps: NodeDeps) -> str:
    user_content = _build_user_content(state)

    try:
        resp = chat_completion(
            deps,
            messages=[
                {"role": "system", "content": GENERATE_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        reply = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"⚠️  [generate] LLM 调用失败，使用兜底回复: {e}")
        reply = _fallback_reply(state)

    return {
        "final_response": reply,
        "messages": [{"role": "assistant", "content": reply}],
    }


def _build_user_content(state: AgentState) -> str:
    plan_lines = "\n".join(
        f'{i + 1}. {s.get("description", "")}' for i, s in enumerate(state.get("plan", []))
    ) or "（无）"
    return (
        f"【用户原始请求】{state['user_query']}\n\n"
        f"【执行计划】\n{plan_lines}\n\n"
        f"【各步骤执行结果】\n{format_step_results(state.get('step_results', []))}\n\n"
        "请综合以上信息生成最终回复（纯文本，不要输出 JSON）。"
    )


def _fallback_reply(state: AgentState) -> str:
    """LLM 不可用时的兜底：如实汇总执行情况，不编造结果。"""
    results = state.get("step_results", [])
    failed = [r for r in results if r.get("status") == "failed"]
    if not results:
        return "抱歉，我暂时无法处理您的请求，请稍后再试或转接人工客服。"
    if failed:
        return (
            "抱歉，处理您的请求时部分步骤未成功"
            f"（{failed[-1].get('error') or '未知原因'}），"
            "建议您稍后再试，或我为您转接人工客服。"
        )
    return "已根据现有信息处理您的请求，如还有疑问请随时告诉我。"
