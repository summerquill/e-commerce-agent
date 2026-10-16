"""planner 节点：把用户请求拆解为有序步骤列表（LLM structured output）。

replan 时（plan_iteration>0）读旧 plan + step_results 作上下文重规划。
注入：skill catalog + 跨轮 memory（MemoryManager 注入 planner 的设计决策）。
"""

from __future__ import annotations

from app.plan_execute.nodes.context import (
    NodeDeps,
    chat_completion,
    memory_sections,
    normalize_steps,
    parse_lenient_json,
    skill_catalog_text,
    tool_catalog_text,
)
from app.plan_execute.prompts import PLANNER_PROMPT
from app.plan_execute.state import AgentState


def planner_node(state: AgentState, deps: NodeDeps) -> dict:
    is_replan = state.get("plan_iteration", 0) > 0

    system_content = PLANNER_PROMPT
    system_content += f"\n\n## 可用工具\n{tool_catalog_text(deps)}\n"
    system_content += skill_catalog_text(deps)

    messages: list[dict] = [{"role": "system", "content": system_content}]
    messages.extend(memory_sections(deps))
    # state["messages"]：首轮为用户原始请求；后续 task3 将注入跨轮 summary/history
    messages.extend(state.get("messages", []))

    if is_replan:
        messages.append({
            "role": "user",
            "content": (
                "上一版计划执行失败，需要重新规划。\n"
                f"【上一版计划】\n{_format_plan(state.get('plan', []))}\n\n"
                f"【已执行结果】\n{_format_results(state.get('step_results', []))}\n\n"
                "请基于失败原因调整计划（换步骤或换方式），重新输出完整 JSON。"
            ),
        })

    try:
        resp = chat_completion(deps, messages, json_mode=True, temperature=0.3)
        raw = parse_lenient_json(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"⚠️  [planner] LLM 调用失败，使用兜底单步计划: {e}")
        raw = None

    steps = normalize_steps(raw, state["user_query"])
    iteration = state.get("plan_iteration", 0) + 1
    print(f"🧭 [planner v{iteration}] 生成 {len(steps)} 个步骤"
          f"{'（重规划）' if is_replan else ''}")

    return {
        "plan": steps,
        "plan_iteration": iteration,
        "current_step_idx": 0,  # 新计划（含 replan）从头执行
        "messages": [{
            "role": "assistant",
            "content": f"[planner v{iteration}] 已生成 {len(steps)} 个步骤的计划",
        }],
    }


def _format_plan(plan) -> str:
    lines = []
    for i, s in enumerate(plan):
        lines.append(f'{i + 1}. {s.get("description", "")}')
    return "\n".join(lines) or "（空）"


def _format_results(results) -> str:
    lines = []
    for r in results:
        lines.append(
            f'- {r.get("step_id")} [{r.get("status")}] '
            f'工具={r.get("tool")} 错误={r.get("error") or "无"}'
        )
    return "\n".join(lines) or "（无）"
