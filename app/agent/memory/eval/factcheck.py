"""关键事实保留检查：确定性字符串匹配，离线零 LLM 依赖。

订单号、会员等级、商品名等实体类事实用子串匹配即可高置信判定；
语义类事实（承诺、偏好）的子串判定偏宽松，精确归因留给后续
LLM-as-judge 扩展（Step 2 接入真实摘要器时启用）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    """一条埋点关键事实。gold 是压缩后必须仍可检索到的关键字符串。"""

    id: str
    category: str  # order_id / product / identity / promise / preference / other
    gold: str


@dataclass
class FactState:
    """某轮压缩后单个事实的存在状态。"""

    in_summary: bool  # 出现在滚动摘要中
    in_window: bool   # 出现在保留的原始消息窗口中

    @property
    def retained(self) -> bool:
        return self.in_summary or self.in_window

    def to_dict(self) -> dict:
        return {"in_summary": self.in_summary, "in_window": self.in_window}


def message_text(msg: dict) -> str:
    """把单条消息展开为可检索文本。

    包含 tool_calls 的参数——订单号等实体经常只出现在调用参数里。
    """
    parts = [msg.get("content") or ""]
    for tc in msg.get("tool_calls") or []:
        func = tc.get("function", {})
        parts.append(f"{func.get('name', '')} {func.get('arguments', '')}")
    return "\n".join(p for p in parts if p)


def _normalize(text: str) -> str:
    """去除全部空白字符后再匹配。

    根因（LLM 基线实测发现）：摘要器重写时常增删数字/标点两侧空格
    （如「100 元无门槛」→「100元无门槛」），逐字面子串匹配会漏报。
    实体匹配必须对空白不敏感，否则度量本身失真。
    """
    return "".join(text.split())


def check_facts(
    facts: list[Fact],
    summary: str | None,
    window_messages: list[dict],
) -> dict[str, FactState]:
    """检查每个事实在（摘要 ∪ 保留窗口）中是否仍可检索到（空白不敏感）。"""
    summary_norm = _normalize(summary or "")
    window_norm = _normalize(
        "\n".join(message_text(m) for m in window_messages)
    )
    states: dict[str, FactState] = {}
    for fact in facts:
        gold_norm = _normalize(fact.gold)
        states[fact.id] = FactState(
            in_summary=bool(summary) and gold_norm in summary_norm,
            in_window=gold_norm in window_norm,
        )
    return states
