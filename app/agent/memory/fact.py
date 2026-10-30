"""记忆事实的统一结构化模型。

STM 与 LTM 共用同一 schema，核心字段：
- content  : 事实正文
- category : identity / preference / behavior / issue / other
- fact_type: current（当前仍有效的状态）| historical（过去发生、可能已过时）
- created_at: 记录时间（ISO）

设计动机：旧版 STM 是 list[str]，历史事件与当前状态混在一起，
注入 prompt 后模型容易把「过去咨询过会员权益」当成「用户现在仍是该状态」。
结构化后，渲染与生成阶段可以分开展示、分开约束。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

FACT_CURRENT = "current"
FACT_HISTORICAL = "historical"

_VALID_FACT_TYPES = {FACT_CURRENT, FACT_HISTORICAL}


@dataclass
class MemoryFact:
    content: str
    category: str = "other"
    fact_type: str = FACT_HISTORICAL
    created_at: str = ""
    source_session: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> "MemoryFact":
        """兼容旧数据：旧版 STM 保存的是纯字符串，默认视为历史事实。"""
        if isinstance(data, str):
            return cls(content=data, fact_type=FACT_HISTORICAL)
        fact_type = str(data.get("fact_type") or FACT_HISTORICAL)
        if fact_type not in _VALID_FACT_TYPES:
            fact_type = FACT_HISTORICAL
        return cls(
            content=str(data.get("content", "")),
            category=str(data.get("category") or "other"),
            fact_type=fact_type,
            created_at=str(data.get("created_at") or ""),
            source_session=str(data.get("source_session") or ""),
        )


def group_facts(facts: list[MemoryFact]) -> tuple[list[MemoryFact], list[MemoryFact]]:
    """按 fact_type 分组，返回 (当前状态, 历史事实)。"""
    current = [f for f in facts if f.fact_type == FACT_CURRENT]
    historical = [f for f in facts if f.fact_type != FACT_CURRENT]
    return current, historical


def render_facts(
    facts: list[MemoryFact],
    *,
    with_category: bool = False,
    with_time: bool = True,
) -> str:
    """把一组事实渲染为多行文本，按「当前状态 / 历史事实」分组。"""
    current, historical = group_facts(facts)
    parts: list[str] = []

    if current:
        parts.append("【当前状态（近期确认，仍可能变化）】")
        parts.append(_render_lines(current, with_category, with_time))

    if historical:
        parts.append("【历史事实（过去发生，可能已过时，勿当作当前状态）】")
        parts.append(_render_lines(historical, with_category, with_time))

    return "\n".join(parts)


def _render_lines(
    facts: list[MemoryFact],
    with_category: bool,
    with_time: bool,
) -> str:
    lines = []
    for f in facts:
        prefix = f"[{f.category}] " if with_category else ""
        suffix = f"（记录于 {f.created_at}）" if with_time and f.created_at else ""
        lines.append(f"- {prefix}{f.content}{suffix}")
    return "\n".join(lines)
