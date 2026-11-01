"""真实 LLM 摘要器适配器：把生产 summarize() / summarize_structured()
适配到 runner 协议。

协议：(old_messages, prev_summary) -> str，与 baselines 一致。
按 Lessons Learned 约定：显式 timeout + max_retries=0 + 每次调用可见日志，
失败快速抛出（评测是人工触发场景，静默重试只会掩盖通道问题）。

token 账目三口径分离（用户约定）：
- 摘要文本字符数（len，硬截断后）
- 输出 token 数（completion_tokens，推理模型含思维链）
- 累计调用 token 数（prompt_tokens + completion_tokens，真实成本口径）
- reasoning_tokens 单独拆出，不与 JSON/摘要文本长度混同

hard_cap 默认 None（不截断）：复现 Step 2 无兜底基线，复测方差用；
对照生产 pinned 模式时显式传 settings.summary_hard_cap，保持与生产
summarize() 行为一致。
"""

from __future__ import annotations

import json
import time

from openai import OpenAI

from app.agent.summarizer import summarize, summarize_structured
from app.config.settings import settings


def _usage_entry(usage: dict, text_chars: int) -> dict:
    """单轮 token 账目：字符数 + 三类 token 分离。"""
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    return {
        "text_chars": text_chars,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "total_tokens": (pt + ct) if (pt is not None and ct is not None) else None,
    }


def build_llm_summarizer(
    timeout: float = 60.0,
    hard_cap: int | None = None,
    budget_chars: int | None = None,
):
    """构建真实 LLM 自由文本摘要器，client/model 取自 settings（.env）。

    budget_chars 为 prompt 级摘要预算（None=默认 300）；压测时传小值
    （如 100）迫使摘要取舍。
    逐轮账目记录在返回函数的 usage_log 属性上。
    """
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=timeout,
        max_retries=0,
    )
    model = settings.model_name

    def summarize_fn(old_messages: list[dict], prev_summary: str | None) -> str:
        t0 = time.time()
        out, usage = summarize(
            client, model, old_messages, prev_summary,
            hard_cap=hard_cap, **({"budget_chars": budget_chars} if budget_chars else {}),
        )
        entry = _usage_entry(usage, len(out))
        summarize_fn.usage_log.append(entry)
        cap_note = f" cap={hard_cap}" if hard_cap else ""
        budget_note = f" budget={budget_chars}" if budget_chars else ""
        print(
            f"  [llm summarize{budget_note}{cap_note}] model={model} old={len(old_messages)}条 "
            f"prev={len(prev_summary or '')}字 -> {entry['text_chars']}字 "
            f"out_tok={entry['completion_tokens']} total_tok={entry['total_tokens']} "
            f"({time.time() - t0:.1f}s)"
        )
        return out

    summarize_fn.usage_log: list[dict] = []
    return summarize_fn


def build_structured_summarizer(
    timeout: float = 60.0, budget_chars: int = 300, client=None,
):
    """结构化状态摘要器（runner 协议兼容）：(old_messages, prev_state_json) -> state_json。

    - usage_log 逐轮记录 token 账目（三口径分离，见模块 docstring）；
      text_chars 是回退后最终状态的字符数
    - 垃圾输出回退：解析失败时透传上一轮状态（陈旧优于崩坏——丢失整份状态
      比短暂陈旧破坏性更大），parse_ok=False 供报告暴露
    - client 参数供测试注入桩（None 时按 settings 构建真实 client）
    """
    if client is None:
        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=timeout,
            max_retries=0,
        )
    model = settings.model_name

    def summarize_fn(old_messages: list[dict], prev_summary: str | None) -> str:
        t0 = time.time()
        state, raw, usage = summarize_structured(
            client, model, old_messages, prev_summary, budget_chars=budget_chars,
        )
        if state is None:
            out = prev_summary or "{}"
            ok = False
        else:
            out = json.dumps(state, ensure_ascii=False)
            ok = True
        entry = _usage_entry(usage, len(out))
        entry["parse_ok"] = ok
        summarize_fn.usage_log.append(entry)
        summarize_fn.parse_ok.append(ok)
        print(
            f"  [llm structured budget={budget_chars}] model={model} old={len(old_messages)}条 "
            f"prev={len(prev_summary or '')}字 -> {entry['text_chars']}字 "
            f"out_tok={entry['completion_tokens']}(reason={entry['reasoning_tokens']}) "
            f"total_tok={entry['total_tokens']} "
            f"parse={'ok' if ok else 'FALLBACK'} ({time.time() - t0:.1f}s)"
        )
        return out

    summarize_fn.usage_log: list[dict] = []
    summarize_fn.parse_ok: list[bool] = []
    return summarize_fn
