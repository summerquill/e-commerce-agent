"""轻量 LLM judge：faithfulness + answer_correctness，每题各 1 次调用。

设计取舍（替代 ragas，2026-10-28）：
- ragas 0.4 的依赖链（datasets>=4 → pyarrow 26 → numpy>=2）与本地
  bge 栈（torch 2.2.2 + numpy 1.26 ABI 锁定）不兼容，无法共存；
- 自写 judge 每题 1 次调用（ragas 每题 3-4 次），prompt 完全可控，
  输出断言明细支持人审；
- 局限：judge 与生成共用同一模型（自评偏差风险），换模型可消除。

client/model 由调用方注入（run_eval 经 rag_adapter 传入默认配置），
judge 失败返回 None，不阻塞评测主流程。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

# free 模型限流（429）常见：重试 3 次，指数退避
_RETRY_DELAYS = (5, 10, 20)

FAITHFULNESS_PROMPT = """\
你是严格的 RAG 评测裁判。给定 question、answer 和若干 context（检索到的知识原文），
请从 answer 中提取所有原子事实断言（assertions），并逐条判断该断言是否被 context 明确支持。

判定规则：
- 只判断「context 是否支持该断言」，不判断回答好坏或风格；
- answer 中与 context 无关的寒暄/复述问题本身的句子不算断言；
- supported=false 时给出 evidence=null，true 时给出支持它的 context 原文片段；
- 只输出 JSON，不要输出其他文字，格式：
{{"assertions": [{{"text": "<断言>", "supported": true, "evidence": "<原文片段>"}}]}}

question: {question}

contexts:
{contexts}

answer: {answer}"""

CORRECTNESS_PROMPT = """\
你是严格的 RAG 评测裁判。给定 question、answer 和 reference（标准答案要点），
请从 reference 中提取关键事实点（facts），并逐条判断 answer 是否覆盖了该事实点。

判定规则：
- 只判断「answer 是否包含该事实点」，不要求措辞一致，语义等价即算 covered；
- 语义相反或数值错误记 contradicted=true（同时 covered=false）；
- 只输出 JSON，不要输出其他文字，格式：
{{"facts": [{{"text": "<事实点>", "covered": true, "contradicted": false}}]}}

question: {question}

reference: {reference}

answer: {answer}"""


def _parse_json(raw: str) -> dict[str, Any] | None:
    """lenient JSON 解析：容忍 markdown fences 与前后杂质，提取第一个 {...}。"""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _chat_json(client: OpenAI, model: str, prompt: str) -> dict[str, Any] | None:
    """单次 judge 调用：温度 0 + 429 退避重试 + lenient JSON 解析；失败返回 None。"""
    resp = None
    for attempt, delay in enumerate((0,) + _RETRY_DELAYS):
        if delay:
            print(f"[judge] 调用失败，{delay}s 后重试 (attempt {attempt + 1}/{len(_RETRY_DELAYS)})", flush=True)
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            break
        except Exception as exc:
            print(f"[judge] 异常: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            if attempt == len(_RETRY_DELAYS):
                return None
    return _parse_json(resp.choices[0].message.content or "")


def judge_faithfulness(
    client: OpenAI, model: str, *, question: str, answer: str, contexts: list[str]
) -> float | None:
    """忠实度 = 被检索 context 支持的断言占比。无断言或解析失败返回 None。"""
    prompt = FAITHFULNESS_PROMPT.format(
        question=question,
        contexts="\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)),
        answer=answer,
    )
    data = _chat_json(client, model, prompt)
    if not data or not isinstance(data.get("assertions"), list):
        return None
    assertions = data["assertions"]
    if not assertions:
        return None
    supported = sum(1 for a in assertions if a.get("supported") is True)
    return round(supported / len(assertions), 4)


def judge_correctness(
    client: OpenAI, model: str, *, question: str, answer: str, reference: str
) -> dict[str, Any] | None:
    """答案正确性 = 覆盖 reference 事实点的占比；附明细供人审。

    返回 {"score": float, "missing": [...], "contradicted": [...]}。
    无事实点或解析失败返回 None。
    """
    prompt = CORRECTNESS_PROMPT.format(
        question=question, reference=reference, answer=answer
    )
    data = _chat_json(client, model, prompt)
    if not data or not isinstance(data.get("facts"), list):
        return None
    facts = data["facts"]
    if not facts:
        return None
    covered = sum(1 for f in facts if f.get("covered") is True)
    return {
        "score": round(covered / len(facts), 4),
        "missing": [f.get("text") for f in facts if f.get("covered") is not True],
        "contradicted": [f.get("text") for f in facts if f.get("contradicted") is True],
    }
