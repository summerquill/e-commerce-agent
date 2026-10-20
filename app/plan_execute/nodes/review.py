"""review 节点：路由中枢，4 出口 done/continue/clarify/replan。

双路径（设计决策）：
- 快路径（规则，不调 LLM）：needs_clarify / needs_replan / 全部完成 / 超 plan_iterations 上限
- 慢路径（LLM）：上一步 failed → 判断该重规划还是兜底回复
"""

from __future__ import annotations

from app.config.settings import settings
from app.plan_execute.nodes.context import (
    NodeDeps,
    chat_completion,
    format_step_results,
    parse_lenient_json,
)
from app.plan_execute.prompts import REVIEW_PROMPT
from app.plan_execute.state import AgentState


def review_node(state: AgentState, deps: NodeDeps) -> dict:
    results = state.get("step_results", [])
    last = results[-1] if results else None

    # 快路径：确定性信号
    if last and last["status"] == "needs_clarify":
        return {"review_decision": "clarify"}
    # 迭代上限兜底优先：防止 failed→replan / needs_replan→replan 死循环
    if state.get("plan_iteration", 0) >= settings.max_plan_iterations:
        return {"review_decision": "done"}
    # executor 判定用户目标已变 → 确定性 replan（判断归属 executor，路由归 review）
    if last and last["status"] == "needs_replan":
        print(f"🔎 [review] 目标已变 → replan（{last.get('error') or ''}）")
        return {"review_decision": "replan"}
    # 慢路径：失败的最后一步仍可能通过 replan 恢复，故须在"全部完成"判断之前
    if last and last["status"] == "failed":
        return {"review_decision": _judge_failure(state, deps)}
    if state["current_step_idx"] >= len(state["plan"]):
        return {"review_decision": "done"}

    return {"review_decision": "continue"}


def _judge_failure(state: AgentState, deps: NodeDeps) -> str:
    """LLM 判断失败是否可通过换步骤/换方式恢复。异常/无法解析时兜底 done。"""
    try:
        resp = chat_completion(
            deps,
            messages=[
                {"role": "system", "content": REVIEW_PROMPT},
                {"role": "user", "content": (
                    f"【用户请求】{state['user_query']}\n\n"
                    f"【当前计划】\n{_format_plan(state.get('plan', []))}\n\n"
                    f"【全部执行结果】\n{format_step_results(state.get('step_results', []))}\n\n"
                    '只输出 JSON：{"decision": "replan" 或 "done", "reason": "..."}'
                )},
            ],
            json_mode=True,
            temperature=0.0,
        )
        payload = parse_lenient_json(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"⚠️  [review] replan 判断失败，兜底 done: {e}")
        return "done"

    decision = str((payload or {}).get("decision", "")).strip().lower()
    if decision in ("replan", "done"):
        print(f"🔎 [review] failed → {decision}（{(payload or {}).get('reason', '')}）")
        return decision
    return "done"


def _format_plan(plan) -> str:
    return "\n".join(
        f'{i + 1}. {s.get("description", "")}' for i, s in enumerate(plan)
    ) or "（空）"
