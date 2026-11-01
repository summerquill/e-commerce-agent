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


# 语义等价人工核验的槽位证据映射：gold 类别 → 结构化状态中优先查看的字段。
# 匹配的是值所在 JSON 路径上的任意一段键名（如 identifiers 子树下的中文
# 类型键「订单号」的值，靠路径中的 identifiers 段命中）。None 表示全部。
_CATEGORY_FIELDS: dict[str, tuple[str, ...] | None] = {
    "status": ("status", "next_action", "business", "change_note"),
    "promise": ("promise", "current_sla", "previous_sla", "next_action"),
    "current_promise": ("current_sla", "promise", "change_note"),
    "historical_promise": ("previous_sla", "promise"),
    "preference": ("preferences",),
    "order_id": ("identifiers",),
}


def structured_evidence(
    fact: Fact, state_json: str | None,
) -> list[str]:
    """从结构化状态 JSON 中抽取与该 gold 语义相关的槽位值，供人工核验语义等价。

    背景（v2 实测）：字面子串匹配对措辞改写产生假阴性（摘要写「48小时内发新机」，
    gold 是「48 小时内发出」——信息在场被判丢）。结构化摘要下字段值就是原文
    表述，把类别相关槽位的值列为证据，争议样本由人工判定语义等价，不引入
    LLM judge（确定性优先，judge 单条波动可达 ±0.6）。

    state_json 无法解析时返回空列表（调用方按"无额外证据"处理）。
    """
    import json

    if not state_json:
        return []
    try:
        state = json.loads(state_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(state, dict):
        return []

    fields = _CATEGORY_FIELDS.get(fact.category)
    values: list[str] = []

    def _walk(node, path: tuple[str, ...] = ()) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, path + (str(k),))
        elif isinstance(node, list):
            for item in node:
                _walk(item, path)
        else:
            if fields is None or any(seg in fields for seg in path):
                values.append(str(node))

    _walk(state)
    # 完整槽位证据供人工核验；仅过滤空值与 unknown 占位
    return sorted({v for v in values if v and v != "unknown"})
