import json
from typing import Optional

from openai import OpenAI

from app.prompts.summarizer import SUMMARY_PROMPT


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


def summarize(
    client: OpenAI,
    model: str,
    old_messages: list[dict],
    prev_summary: Optional[str],
) -> str:
    """把老对话（可选地带上上一次 summary）压缩成新的 summary 文本。

    支持 user / assistant / tool 以及含 tool_calls 的 assistant 消息。
    """
    parts: list[str] = []
    if prev_summary:
        parts.append(f"【此前摘要】\n{prev_summary}")

    parts.append("【待压缩对话】\n" + build_transcript(old_messages))

    user_content = "\n\n".join(parts)

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()
