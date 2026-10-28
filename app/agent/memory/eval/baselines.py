"""离线参考摘要器（不调 API）：harness 的行为锚点与 Step 2 的对照基线。

两个实现均模拟真实摘要器的递归形态（prev_summary + 本轮新信息，
总长受 max_chars 约束，旧摘要最多占一半预算——模拟「300 字内重写」
的真实约束：随轮次推进新信息空间被旧摘要挤占）：

- truncate_summarizer：无提取能力，transcript 头部截断 → 预期事实快速丢失
- extractive_summarizer：按通用信息密度规则抽行 → 预期保留率显著更高

两者都不接触评测集的事实清单（gold），规则词表是通用领域词。
若 harness 无法区分这两个已知行为差异的基线，说明度量管线自身有问题。
"""

from __future__ import annotations

import re

from app.agent.summarizer import build_transcript

# 订单号式实体：字母前缀 + 长数字，或纯长数字（税号等）
_ORDER_RE = re.compile(r"[A-Za-z]{2,6}\d{5,}|\d{6,}")

# 通用客服信息密度词表（非评测集专用）
_DENSITY_KEYWORDS = (
    "订单", "会员", "已同意", "已提交", "已备注", "已核实", "已记录",
    "承诺", "退货", "退款", "补发", "发货", "发票", "优惠券", "质保",
    "上门", "地址", "预算", "喜欢", "优先", "加急",
)


def _assemble(prev_summary: str | None, new_text: str, max_chars: int) -> str:
    """拼装递归摘要：旧摘要占前一半预算，剩余空间给新信息。"""
    parts: list[str] = []
    used = 0
    if prev_summary:
        prev_budget = max_chars // 2
        clipped = prev_summary[:prev_budget]
        parts.append(clipped)
        used += len(clipped) + 1  # 换行符
    remain = max_chars - used
    if remain > 0 and new_text:
        parts.append(new_text[:remain])
    return "\n".join(parts)[:max_chars]


def truncate_summarizer(max_chars: int = 300):
    """朴素截断基线：无提取能力，直接保留 transcript 头部。"""

    def summarize_fn(old_messages: list[dict], prev_summary: str | None) -> str:
        transcript = build_transcript(old_messages)
        return _assemble(prev_summary, transcript, max_chars)

    return summarize_fn


def _score_line(line: str) -> int:
    """行级信息密度打分：实体 > 关键词 > 工具结果。"""
    score = 0
    if _ORDER_RE.search(line):
        score += 3
    hits = sum(1 for kw in _DENSITY_KEYWORDS if kw in line)
    score += min(hits, 3) * 2
    if line.startswith("[工具结果]"):
        score += 1
    return score


def extractive_summarizer(max_chars: int = 300):
    """规则抽取基线：按信息密度选行，在预算内尽量多保留高密度行。"""

    def summarize_fn(old_messages: list[dict], prev_summary: str | None) -> str:
        lines = build_transcript(old_messages).splitlines()
        if not lines:
            return _assemble(prev_summary, "", max_chars)

        prev_budget = max_chars // 2
        remain = max_chars - min(len(prev_summary or ""), prev_budget) - (1 if prev_summary else 0)

        ranked = sorted(
            enumerate(lines), key=lambda pair: (-_score_line(pair[1]), pair[0]),
        )
        chosen: list[tuple[int, str]] = []
        used = 0
        for idx, line in ranked:
            if not line or used + len(line) + 1 > remain:
                continue
            chosen.append((idx, line))
            used += len(line) + 1
        chosen.sort()  # 恢复原始对话顺序
        new_text = "\n".join(line for _, line in chosen)
        return _assemble(prev_summary, new_text, max_chars)

    return summarize_fn


SUMMARIZERS = {
    "truncate": truncate_summarizer,
    "extractive": extractive_summarizer,
}
