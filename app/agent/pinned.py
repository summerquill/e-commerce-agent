"""pinned facts：关键实体锚定——订单号/手机号/邮箱等标识符原文钉住，永不进有损摘要。

实证背景（Step 2 基线，deepseek-v4-flash，12 轮压缩）：
- 递归压缩存在真实漂移：round4 重写可能整条丢弃售后线
- prompt 级"≤300 字"约束不可靠：4/12 轮超限（max 367 字）

机制：压缩前用正则抽取"必须原样保留"的标识符，跨轮合并进 pinned 列表，
随 summary 一起以独立 system 段注入，形成「摘要（有损）+ 锚点（无损）」双通道。
标识符即使被摘要丢掉也能从锚点段找回；语义类承诺（"15 天无理由"等）仍走
摘要通道，由评测分类别呈现各自保留率。

设计取舍：抽取只做确定性正则、不调 LLM——锚点机制自身必须可靠且可复现；
纯数字 ≥6 位兜底手机号/单号，日期 2025-11-03、时间 14:00-18:00、金额 199
天然不命中；紧凑日期（20251103）会被钉住，属可接受误报。
"""

import json
import re
from typing import Optional

# 标识符 token：字母/数字/连字符 ≥8 位，同时含字母与数字
# （排除纯数字、纯单词、2025-11-03、14:00-18:00）。
# Python 的 \w 是 Unicode 感知的，中文字符算 \w，\b 在中文与字母间不成立，
# 必须用显式 lookaround 定边界。
_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9-])"                # 左边界：前面不是 token 字符
    r"(?=[A-Za-z0-9-]*[A-Za-z])"        # 含字母
    r"(?=[A-Za-z0-9-]*[0-9])"           # 含数字
    r"[A-Za-z0-9-]{8,}"
    r"(?![A-Za-z0-9-])"                 # 右边界
)

# 纯数字标识符：≥6 位连续数字（手机号/纯数字单号），不落在 token 内部
_DIGITS_RE = re.compile(r"(?<![0-9])\d{6,}(?![0-9])")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _valid_token(raw: str) -> bool:
    """token 二次校验：去掉首尾连字符后仍需 ≥8 位且字母/数字各 ≥2 个。"""
    tok = raw.strip("-")
    if len(tok) < 8:
        return False
    return (
        len(re.findall(r"[A-Za-z]", tok)) >= 2
        and len(re.findall(r"[0-9]", tok)) >= 2
    )


def _scan_text(text: str) -> list[str]:
    """扫一段文本，按出现顺序返回其中的标识符（可能重复）。"""
    entities: list[str] = []

    token_spans = [
        (m.start(), m.end())
        for m in _TOKEN_RE.finditer(text)
        if _valid_token(m.group())
    ]
    for start, end in token_spans:
        entities.append(text[start:end].strip("-"))

    token_ranges = set()
    for start, end in token_spans:
        token_ranges.update(range(start, end))

    for m in _DIGITS_RE.finditer(text):
        if token_ranges.isdisjoint(range(m.start(), m.end())):
            entities.append(m.group())

    entities.extend(m.group() for m in _EMAIL_RE.finditer(text))
    return entities


def _message_texts(msg: dict) -> list[str]:
    """取一条消息中所有需要扫描的文本（content + tool_calls 参数原文）。"""
    texts = []
    content = msg.get("content")
    if isinstance(content, str) and content:
        texts.append(content)
    for tc in msg.get("tool_calls") or []:
        args = tc.get("function", {}).get("arguments")
        if isinstance(args, str) and args:
            texts.append(args)
    return texts


def extract_pinnable_entities(messages: list[dict]) -> list[str]:
    """从消息列表中按序抽取可钉住的标识符。

    覆盖 user/assistant/tool 的 content 及 assistant tool_calls 的
    arguments JSON 原文（工具调用里的订单号是关键来源）。
    """
    entities: list[str] = []
    for msg in messages:
        for text in _message_texts(msg):
            entities.extend(_scan_text(text))
    return entities


def merge_pinned(
    existing: list[str], new: list[str], max_pinned: int = 30,
) -> list[str]:
    """合并进 pinned 列表：去重保序，超上限时按 FIFO 淘汰最旧的。

    上限防止长会话把 system 段撑爆；淘汰最旧符合"近期订单更常被追问"的
    客服场景直觉。
    """
    seen = set(existing)
    merged = list(existing)
    for item in new:
        if item and item not in seen:
            seen.add(item)
            merged.append(item)
    return merged[-max_pinned:]


def render_pinned_section(pinned: list[str]) -> Optional[str]:
    """渲染锚点 system 段；空列表返回 None（不注入）。"""
    if not pinned:
        return None
    lines = "\n".join(f"- {item}" for item in pinned)
    return (
        "【关键信息锚点】以下是本会话出现过的关键标识符（订单号/手机号/邮箱等），"
        "回答时必须与原文完全一致，禁止改写或省略：\n"
        f"{lines}"
    )
