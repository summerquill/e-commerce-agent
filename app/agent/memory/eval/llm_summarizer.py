"""真实 LLM 摘要器适配器：把生产 summarize() 适配到 runner 协议。

协议：(old_messages, prev_summary) -> str，与 baselines 一致。
按 Lessons Learned 约定：显式 timeout + max_retries=0 + 每次调用可见日志，
失败快速抛出（评测是人工触发场景，静默重试只会掩盖通道问题）。

hard_cap 默认 None（不截断）：复现 Step 2 无兜底基线，复测方差用；
对照生产 pinned 模式时显式传 settings.summary_hard_cap，保持与生产
summarize() 行为一致。
"""

from __future__ import annotations

import time

from openai import OpenAI

from app.agent.summarizer import summarize
from app.config.settings import settings


def build_llm_summarizer(timeout: float = 60.0, hard_cap: int | None = None):
    """构建真实 LLM 摘要器，client/model 取自 settings（.env）。"""
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=timeout,
        max_retries=0,
    )
    model = settings.model_name

    def summarize_fn(old_messages: list[dict], prev_summary: str | None) -> str:
        t0 = time.time()
        out = summarize(client, model, old_messages, prev_summary, hard_cap=hard_cap)
        cap_note = f" cap={hard_cap}" if hard_cap else ""
        print(
            f"  [llm summarize{cap_note}] model={model} old={len(old_messages)}条 "
            f"prev={len(prev_summary or '')}字 -> {len(out)}字 "
            f"({time.time() - t0:.1f}s)"
        )
        return out

    return summarize_fn
