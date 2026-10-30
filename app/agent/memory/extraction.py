"""LLM 事实提取：从对话中抽取短期/长期记忆事实。

模式与 app/agent/summarizer.py 一致：格式化对话 → 调用 LLM → 解析结果。
提取结果统一为 MemoryFact（content / category / fact_type / created_at），
区分「当前状态」与「历史事实」，避免后续把历史记录当成实时状态。
"""

from __future__ import annotations

import json
from datetime import datetime

from openai import OpenAI

from app.agent.memory.fact import FACT_CURRENT, FACT_HISTORICAL, MemoryFact
from app.prompts.memory import LTM_EXTRACTION_PROMPT, STM_EXTRACTION_PROMPT


def _build_transcript(messages: list[dict]) -> str:
    """将消息列表格式化为文本摘要（复用 summarizer 的逻辑）。"""
    lines = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""

        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    lines.append(f"客服：[调用工具 {name}]")
            if content:
                lines.append(f"客服：{content}")
        elif role == "tool":
            display = content if len(content) <= 200 else content[:200] + "..."
            lines.append(f"[工具结果] {display}")

    return "\n".join(lines)


def _parse_facts(raw: str, default_fact_type: str) -> list[MemoryFact]:
    """解析 LLM 输出为结构化 facts。

    优先按 JSON 解析（{"facts": [...]} 或裸数组）；解析失败时回退旧版
    逐行文本格式，保证 prompt 升级期间不会整段丢事实。
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    items = None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            items = data.get("facts")
        elif isinstance(data, list):
            items = data
    except json.JSONDecodeError:
        items = None

    if not isinstance(items, list):
        # 兼容旧格式：每行一条事实
        items = [
            {"content": line.strip().lstrip("- ")}
            for line in text.splitlines()
            if line.strip()
        ]

    now = datetime.now().isoformat(timespec="seconds")
    facts: list[MemoryFact] = []
    for item in items:
        if isinstance(item, str):
            content = item.strip()
            category, fact_type = "other", default_fact_type
        elif isinstance(item, dict):
            content = str(item.get("content") or "").strip()
            category = str(item.get("category") or "other")
            fact_type = str(item.get("fact_type") or default_fact_type)
        else:
            continue
        if not content:
            continue
        if fact_type not in {FACT_CURRENT, FACT_HISTORICAL}:
            fact_type = default_fact_type
        facts.append(MemoryFact(
            content=content,
            category=category,
            fact_type=fact_type,
            created_at=now,
        ))
    return facts


def extract_short_term_facts(
    client: OpenAI,
    model: str,
    recent_messages: list[dict],
    existing_facts: list[MemoryFact],
) -> list[MemoryFact]:
    """从最近对话中提取/更新短期记忆事实。"""
    transcript = _build_transcript(recent_messages)
    if not transcript.strip():
        return existing_facts

    existing_text = (
        "\n".join(f"- {f.content}" for f in existing_facts)
        if existing_facts else "（暂无）"
    )
    prompt = STM_EXTRACTION_PROMPT.format(existing_facts=existing_text)

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
    )
    raw = response.choices[0].message.content.strip()

    if "无新信息" in raw:
        return existing_facts

    new_facts = _parse_facts(raw, default_fact_type=FACT_HISTORICAL)
    if not new_facts:
        return existing_facts

    merged = list(existing_facts)
    existing_contents = {f.content.lower() for f in merged}
    for fact in new_facts:
        if fact.content.lower() not in existing_contents:
            merged.append(fact)
            existing_contents.add(fact.content.lower())
    return merged


def extract_long_term_facts(
    client: OpenAI,
    model: str,
    messages: list[dict],
    summary: str | None,
    existing_facts: list[MemoryFact],
) -> tuple[list[MemoryFact], str]:
    """从完整会话中提取长期记忆事实 + 交互摘要。"""
    parts = []
    if summary:
        parts.append(f"【对话摘要】\n{summary}")

    transcript = _build_transcript(messages)
    if transcript:
        parts.append(f"【对话内容】\n{transcript}")

    if not parts:
        return [], ""

    existing_text = (
        "\n".join(
            f"- [{f.category}][{f.fact_type}] {f.content}"
            for f in existing_facts
        )
        if existing_facts else "（暂无）"
    )
    prompt = LTM_EXTRACTION_PROMPT.format(existing_ltm=existing_text)

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
    )
    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], "（提取失败）"

    if not isinstance(data, dict):
        return [], "（提取失败）"

    new_facts = _parse_facts(
        json.dumps(data.get("facts", []), ensure_ascii=False),
        default_fact_type=FACT_HISTORICAL,
    )

    interaction_summary = data.get("interaction_summary", "")
    return new_facts, interaction_summary
