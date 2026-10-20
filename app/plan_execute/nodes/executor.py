"""executor 节点：执行单个 step。

流程：LLM 用 step.description + 对话记录（messages，含追问问答）+ step_results 动态解析动作
  - 信息充足 → 返回 tool_call（OpenAI function calling，schema 即工具定义）
  - 信息不足 → 不调工具，内容为 {"needs_clarify": true, "question": "...", "options"?: [...]},
    同时写入 pending_clarification 交 clarify 节点发起 HITL
  - 无需工具的步骤（闲聊/直接回复）→ {"direct_reply": "..."}
高风险工具执行前触发 confirm interrupt（强绑定具体 tool call）。
每次只跑 plan[current_step_idx]，跑完交棒 review。
"""

from __future__ import annotations

import json

from langchain_core.messages import convert_to_openai_messages
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app.plan_execute.nodes.context import (
    NodeDeps,
    chat_completion,
    classify_tool_output,
    format_step_results,
    parse_lenient_json,
)
from app.plan_execute.prompts import EXECUTOR_PROMPT
from app.plan_execute.state import AgentState, Step, StepResult

# 高风险工具白名单：执行前触发 confirm interrupt
# TODO（任务4）: 填入真实工具名, e.g. {"apply_refund"}
HIGH_RISK_TOOLS: set[str] = set()


def executor_node(state: AgentState, deps: NodeDeps, config: RunnableConfig) -> dict:
    idx = state["current_step_idx"]
    step: Step = state["plan"][idx]

    # 幂等解析：confirm interrupt resume 时节点整体重放，同一执行位点必须复用
    # 首次解析结果，避免二次 LLM 调用可能解析出不同动作。
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    cache_key = (
        thread_id,
        state.get("plan_iteration", 0),
        idx,
        len(state.get("step_results", [])),
        len(state.get("messages", [])),  # clarify 问答每轮追加，对话长度区分轮次
    )
    action = deps.resolve_cache.get(cache_key)
    if action is None:
        action = _resolve_action(deps, step, state)
        deps.resolve_cache[cache_key] = action

    kind = action[0]
    if kind == "tool":
        _, tool, args = action
        return _run_tool(deps, step, idx, tool, args)

    if kind == "clarify":
        return {
            "step_results": [StepResult(
                step_id=step["id"], tool=None, input={},
                output=None, status="needs_clarify", error=None,
            )],
            "pending_clarification": {
                "question": action[1],
                "options": list(action[2] or []),
            },
            # idx 不前进：clarify 回答后回 executor 重试同一步
        }

    if kind == "reply":
        return {
            "step_results": [StepResult(
                step_id=step["id"], tool=None, input={},
                output=action[1], status="success", error=None,
            )],
            "current_step_idx": idx + 1,
        }

    if kind == "replan":
        # 用户最新回答使当前计划失效（判断同 needs_clarify 一样留在 executor，
        # 路由仍归 review）；idx 不前进，replan 后 planner 重置为 0
        print(f"🔎 [executor] 目标已变 → 请求重规划（{action[1]}）")
        return {
            "step_results": [StepResult(
                step_id=step["id"], tool=None, input={},
                output=None, status="needs_replan", error=action[1],
            )],
        }

    # kind == "error"
    return _fail_step(step, idx, action[1])


def _resolve_action(deps: NodeDeps, step: Step, state: AgentState) -> tuple:
    """LLM 解析当前步动作。返回元组：
    ("tool", name, args) | ("clarify", question, options) | ("reply", text) | ("error", msg)

    上下文三层：system 契约 → messages 对话记录（含历史追问问答，append-only）→
    当前步任务卡（step + step_results + 指令）。
    """
    try:
        resp = chat_completion(
            deps,
            messages=[
                {"role": "system", "content": EXECUTOR_PROMPT},
                *convert_to_openai_messages(state.get("messages", [])),
                {"role": "user", "content": _build_user_content(step, state)},
            ],
            tools=deps.tool_manager.tool_definitions,
            tool_choice="auto",
            temperature=0.0,
        )
    except Exception as e:
        return ("error", f"executor LLM 调用失败: {e}")

    message = resp.choices[0].message

    # 路径一：工具调用（v1 一个 step 只取第一个 tool call）
    if message.tool_calls:
        tc = message.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            return ("error", f"工具参数解析失败: {tc.function.arguments}")
        return ("tool", tc.function.name, args)

    # 路径二：无工具调用 → needs_clarify / needs_replan / direct_reply
    payload = parse_lenient_json(message.content or "") or {}
    if payload.get("needs_clarify"):
        question = str(payload.get("question") or message.content or "请补充更多信息").strip()
        raw_options = payload.get("options")
        options = (
            [str(o).strip() for o in raw_options if str(o).strip()]
            if isinstance(raw_options, list) else []
        )
        return ("clarify", question, options)

    if payload.get("needs_replan"):
        reason = str(payload.get("reason") or "用户改变了目标").strip()
        return ("replan", reason)

    reply = str(payload.get("direct_reply") or message.content or "").strip()
    return ("reply", reply)


# ──────────────────── 内部辅助 ────────────────────

def _run_tool(deps: NodeDeps, step: Step, idx: int, tool: str, args: dict) -> dict:
    """confirm 审批 → 执行工具 → 归类结果（在节点调用栈内，interrupt 正常生效）。"""

    # confirm：高风险工具执行前 interrupt 等用户确认
    if tool in HIGH_RISK_TOOLS:
        approval = interrupt({
            "type": "confirm",
            "action": step["description"],
            "context": {"tool": tool, "args": args},
        })
        if not approval.get("approved"):
            return {
                "step_results": [StepResult(
                    step_id=step["id"], tool=tool, input=args,
                    output=None, status="skipped", error="用户拒绝确认",
                )],
                "current_step_idx": idx + 1,
            }

    print(f"🔧 [执行] step {step['id']} → {tool}({json.dumps(args, ensure_ascii=False)})")
    try:
        raw = deps.tool_manager.execute_tool(tool, args)
        status, error = classify_tool_output(raw)
    except Exception as e:
        raw, status, error = None, "failed", f"工具执行异常: {e}"

    return {
        "step_results": [StepResult(
            step_id=step["id"], tool=tool, input=args,
            output=raw, status=status, error=error,
        )],
        "current_step_idx": idx + 1,
    }


def _build_user_content(step: Step, state: AgentState) -> str:
    """当前步任务卡：步骤语义 + 结构化执行日志 + 输出指令。

    用户请求与追问问答都在 messages 对话记录中，此处不重复。
    """
    parts = [
        f"【当前步骤】{step['id']}: {step['description']}"
        f"（工具提示: {step.get('tool_hint') or '无'}）",
        f"【已完成步骤的结果】\n{format_step_results(state.get('step_results', []))}",
    ]
    parts.append(
        "请决定：调用一个工具；或在信息不足时输出 "
        '{"needs_clarify": true, "question": "向用户追问的问题", '
        '"options": ["可选的候选回答（仅在能给出明确候选时提供，否则省略）"]}；'
        '若用户最新回答改变了目标、当前计划已失去意义，输出 '
        '{"needs_replan": true, "reason": "用户的新目标"}；'
        '若本步无需工具即可回复，输出 {"direct_reply": "回复内容"}。'
    )
    return "\n\n".join(parts)


def _fail_step(step: Step, idx: int, error: str) -> dict:
    return {
        "step_results": [StepResult(
            step_id=step["id"], tool=None, input={},
            output=None, status="failed", error=error,
        )],
        "current_step_idx": idx + 1,
    }
