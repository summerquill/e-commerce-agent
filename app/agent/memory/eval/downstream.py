"""下游任务检查：只给评测器最终摘要（不给原始对话），让它回答业务问题。

目的：验证摘要是否真的能支持后续任务，而不仅仅是字段/文本看起来完整。
问题取自 v2 对抗集的损伤点（自由文本 round3 丢状态词、round4 拿旧 SLA
当现状）：
  Q1 当前换货状态是什么？      → gold: 换货中
  Q2 最新回电 SLA 是多久？     → gold: 2小时内（24小时是被覆盖的旧值）

判定为规则匹配 + 原始答案全文落盘供人工复核：
- Q1 命中「换货中」为对；答「已完成/已退款/换货」而无状态词为错
- Q2 含「2小时」为对；只含「24小时」为错（旧值当现状，正是要检测的损伤）
- Q2 同时含 2/24 小时时以 latest 出现者为准的简化不采用——直接要求答案
  明确不含矛盾，矛盾情况标记 conflict 人工复核
"""

from __future__ import annotations

DOWNSTREAM_PROMPT = """你是电商客服 Agent。下面是这段对话的最终摘要（你看不到原始对话），基于它回答两个问题。

【摘要】
{summary}

请只输出 JSON：{{"exchange_status": "...", "callback_sla": "..."}}
- exchange_status：当前换货业务的状态（用摘要中的原文表述；摘要没写则答"未知"）
- callback_sla：最新的回电时限承诺（如"24小时内"；摘要没写则答"未知"）"""


def run_downstream_check(client, model: str, final_summary: str) -> dict:
    """只喂最终摘要做下游问答，返回答案 + 规则判定 + 原文。"""
    import json as _json
    import re

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "你是严谨的客服信息核对助手，只依据给定材料回答。"},
            {"role": "user", "content": DOWNSTREAM_PROMPT.format(summary=final_summary)},
        ],
    )
    raw = response.choices[0].message.content or ""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        ans = _json.loads(m.group(0)) if m else {}
    except _json.JSONDecodeError:
        ans = {}
    usage = getattr(response, "usage", None)

    exchange = str(ans.get("exchange_status", ""))
    sla = str(ans.get("callback_sla", ""))
    norm = lambda t: "".join(t.split())  # noqa: E731

    q1_ok = "换货中" in norm(exchange)
    has_2h, has_24h = "2小时" in norm(sla), "24小时" in norm(sla)
    q2_ok = has_2h and not has_24h
    q2_conflict = has_2h and has_24h

    return {
        "exchange_status": exchange,
        "callback_sla": sla,
        "q1_exchange_status_ok": q1_ok,
        "q2_latest_sla_ok": q2_ok,
        "q2_conflict": q2_conflict,
        "raw": raw,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        },
    }
