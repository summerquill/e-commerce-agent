import json
from typing import Optional

from openai import OpenAI

from app.prompts.summarizer import (
    SUMMARY_PROMPT,
    build_structured_summary_prompt,
    build_summary_prompt,
)


def split_history(
    messages: list[dict], keep_recent: int,
) -> Optional[tuple[list[dict], list[dict]]]:
    """按保留窗口切分历史，返回 (old, recent)；不满足压缩条件返回 None。

    切分点回退跳过开头的 tool 消息，避免工具结果与其对应的
    tool_calls 消息被拆到两侧。生产压缩与评测 harness 共用此逻辑。
    """
    split = len(messages) - keep_recent
    while split > 0 and messages[split].get("role") == "tool":
        split -= 1
    if split <= 0:
        return None
    return messages[:split], messages[split:]


def build_transcript(messages: list[dict], tool_truncate: int = 200) -> str:
    """把消息列表渲染为 LLM 可读的对话文本（压缩与事实抽取共用的格式）。"""
    lines = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""

        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {})
                name = func.get("name", "?")
                args = func.get("arguments", "{}")
                lines.append(f"客服：[调用工具 {name}({args})]")
            if content:
                lines.append(f"客服：{content}")
        elif role == "tool":
            display = content if len(content) <= tool_truncate else content[:tool_truncate] + "..."
            lines.append(f"[工具结果] {display}")

    return "\n".join(lines)


def _extract_usage(response) -> dict:
    """从 API 响应提取 token 账目（防御 stub 无 usage）。

    reasoning_tokens 单独拆出：推理模型（如 deepseek-v4-flash）的
    completion_tokens 含思维链，不等于最终摘要文本的长度。
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None,
                "reasoning_tokens": None}
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": reasoning,
    }


def summarize(
    client: OpenAI,
    model: str,
    old_messages: list[dict],
    prev_summary: Optional[str],
    hard_cap: Optional[int] = None,
    budget_chars: int = 300,
) -> tuple[str, dict]:
    """把老对话（可选地带上上一次 summary）压缩成新的 summary 文本。

    返回 (summary_text, token_usage)。token_usage 含 prompt_tokens /
    completion_tokens / reasoning_tokens（都可能为 None），供评测做
    字符数、输出 token、累计调用 token 三口径分离的成本账目。

    支持 user / assistant / tool 以及含 tool_calls 的消息。
    budget_chars 为 prompt 级质量目标（无强制力）；hard_cap 为代码级
    硬截断兜底：prompt 约束已被实证不可靠（36 轮中 11 轮超 300 字），
    超限时直接截断，实体由 pinned 锚点兜底。
    """
    parts: list[str] = []
    if prev_summary:
        parts.append(f"【此前摘要】\n{prev_summary}")

    parts.append("【待压缩对话】\n" + build_transcript(old_messages))

    user_content = "\n\n".join(parts)
    system_prompt = (
        SUMMARY_PROMPT if budget_chars == 300 else build_summary_prompt(budget_chars)
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    text = response.choices[0].message.content.strip()
    if hard_cap and len(text) > hard_cap:
        text = text[:hard_cap]
    return text, _extract_usage(response)


def _parse_state_json(raw: str) -> Optional[dict]:
    """lenient JSON 解析：容忍 markdown fences 与前后杂质，提取第一个 {...}。

    与 app/agent/rag/eval/judge.py 的 _parse_json 同策略；解析失败返回 None，
    由调用方决定回退行为（评测适配器回退到上一轮状态）。
    """
    import json
    import re

    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def summarize_structured(
    client: OpenAI,
    model: str,
    old_messages: list[dict],
    prev_state: Optional[str],
    budget_chars: int = 300,
) -> tuple[Optional[dict], str, dict]:
    """结构化摘要：把老对话压缩为固定槽位的状态 JSON。

    prev_state 是上一轮的状态 JSON 字符串（首轮为 None）。走
    response_format=json_object 强制 JSON；token 账目含 prompt_tokens /
    completion_tokens / reasoning_tokens（推理模型 completion 含思维链，
    不等于 JSON 文本长度，故三口径分离记录）。

    返回 (state_dict | None, 原始输出文本, token_usage)。
    state_dict 为 None 表示输出无法解析（由调用方回退）。
    """
    parts: list[str] = []
    if prev_state:
        parts.append(f"【此前状态】\n{prev_state}")

    parts.append("【待压缩对话】\n" + build_transcript(old_messages))

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": build_structured_summary_prompt(budget_chars)},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
    )
    raw = response.choices[0].message.content or ""
    return _parse_state_json(raw), raw, _extract_usage(response)
