"""planner 节点：把用户请求拆解为有序步骤列表（LLM structured output）。

replan 时（plan_iteration>0）读旧 plan + step_results 作上下文重规划。
注入：skill catalog + 跨轮 memory（MemoryManager 注入 planner 的设计决策）。
"""

from __future__ import annotations

from langchain_core.messages import convert_to_openai_messages

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
    # state["messages"]：经 add_messages reducer 后是 LangChain Message 对象，
    # 直接透传 OpenAI SDK 会缺 role 字段（422），统一转回 {role, content} dict；
    # 后续 task3 将在 _build_input 注入跨轮 summary/history，同样走此转换
    messages.extend(convert_to_openai_messages(state.get("messages", [])))

    if is_replan:
        messages.append({
            "role": "user",
            "content": (
                "上一版计划无法继续执行（步骤失败，或用户在中途改变了目标），"
                "需要重新规划。\n"
                f"【上一版计划】\n{_format_plan(state.get('plan', []))}\n\n"
                f"【已执行结果】\n{_format_results(state.get('step_results', []))}\n\n"
                "请结合上方对话记录中用户的最新意图调整计划（换步骤、换方式，"
                "或围绕新目标重新拆解），重新输出完整 JSON。"
            ),
        })

    try:
        resp = chat_completion(deps, messages, json_mode=True, temperature=0.3)
        raw = parse_lenient_json(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"⚠️  [planner] LLM 调用失败，使用兜底单步计划: {e}")
        raw = None

    iteration = state.get("plan_iteration", 0) + 1

    # 短路：闲聊/无需工具时 plan 为空，图条件边直转 generate（不写 messages，交 generate）
    reply = _direct_reply_of(raw)
    if reply is not None:
        print(f"🧭 [planner v{iteration}] 无需规划 → 直接回复")
        return {
            "plan": [],
            "plan_iteration": iteration,
            "current_step_idx": 0,
            "final_response": reply,  # generate 见已有值则透传，不再调 LLM
        }

    steps = normalize_steps(raw, state["user_query"])
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


def _direct_reply_of(raw) -> "str | None":
    """planner 输出契约的 direct_reply 分支；非该形态返回 None（交 normalize_steps 拆步）。"""
    if not isinstance(raw, dict) or raw.get("action") != "direct_reply":
        return None
    reply = str(raw.get("reply") or "").strip()
    return reply or None


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
