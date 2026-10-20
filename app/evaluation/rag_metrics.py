"""RAG 专用评测指标：检索层（召回/精确）+ 生成层（忠实/完整）。

与现有 metrics.py 分工：
- metrics.py 负责 Agent 端到端（tool_accuracy / intent_match 等）
- 本模块负责 RAG 链路独立评测（检索层 + 生成层）

核心函数：
    recall_at_k(expected_ids, retrieved_ids, k)  → float
    precision_at_k(expected_ids, retrieved_ids, k) → float
    judge_rag_faithfulness(client, model, answer, contexts_text) → (score, reason)
    judge_rag_completeness(client, model, answer, grading_notes) → (score, reason)
"""

from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI


# ──────────────────── 检索层指标 ────────────────────

def recall_at_k(
    expected_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    k: int = 3,
) -> float:
    """召回率：ground truth chunks 中有多少被检索到了（前 k 个里）。

    Recall = |expected ∩ retrieved[:k]| / |expected|

    - expected 为空 → 1.0（negative 用例，期望检索不到任何 chunk，
      但召回率定义是"要的找到了吗"，expected 空的情况 recall 自动满分）
    - retrieved 前 k 个可能有重复 chunk_id，去重后计算
    """
    if not expected_chunk_ids:
        return 1.0
    retrieved_topk = list(dict.fromkeys(retrieved_chunk_ids[:k]))  # 去重保序
    expected_set = set(expected_chunk_ids)
    hit = sum(1 for cid in retrieved_topk if cid in expected_set)
    return hit / len(expected_set)


def precision_at_k(
    expected_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    k: int = 3,
) -> float:
    """精确率：检索到的前 k 个 chunks 中有多少是 ground truth。

    Precision = |expected ∩ retrieved[:k]| / k

    - expected 为空 → 期望 retrieved 也为空。retrieved 前 k 个都不在 expected
      里是正常行为（召回空），此时 precision = 0.0 或 1.0 取决于业务定义。
      这里约定：expected 空且 retrieved 前 k 也空 → 1.0（完美拒绝）；
      expected 空但 retrieved 有结果 → 0.0（不该召回却召回了）
    """
    retrieved_topk = list(dict.fromkeys(retrieved_chunk_ids[:k]))
    if not expected_chunk_ids:
        return 1.0 if not retrieved_topk else 0.0
    expected_set = set(expected_chunk_ids)
    hit = sum(1 for cid in retrieved_topk if cid in expected_set)
    return hit / k


# ──────────────────── 生成层指标 ────────────────────

def judge_rag_faithfulness(
    client: OpenAI,
    model: str,
    answer: str,
    contexts: list[dict],
) -> tuple[float, str]:
    """忠实度 judge：答案是否完全基于检索到的上下文，有没有幻觉。

    Ragas faithfulness 定义 = 1 - hallucination_rate
    这里用 LLM 判断答案中的每个陈述是否能从上下文中推导。

    Returns: (score 0.0~1.0, reason)
    """
    if not answer or not answer.strip():
        return 0.0, "答案为空"

    context_text = "\n\n".join(
        f"[{ctx.get('chunk_id', '?')}] {ctx.get('text', '')}" for ctx in contexts
    ) if contexts else "（未检索到任何上下文）"

    prompt = f"""你是忠实度评判专家。判断下面的答案是否完全基于提供的上下文，有没有编造（幻觉）。

【上下文】
{context_text}

【答案】
{answer}

请逐步分析：
1. 把答案拆成几个独立陈述（句子级）
2. 对每个陈述，判断能否从上下文中直接找到依据（是/否）
3. 如果有陈述无法从上下文推导，就是幻觉

输出 JSON:
{{"statement_count": 总数, "faithful_count": 有依据的数量, "hallucinations": ["幻觉内容1", ...], "overall_score": 0.0-1.0, "reason": "评判理由"}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        score = float(data.get("overall_score", 0.0))
        reason = data.get("reason", "")
        return min(1.0, max(0.0, score)), reason
    except (json.JSONDecodeError, Exception) as e:
        return 0.0, f"忠实度评判解析失败: {e}"


def judge_rag_completeness(
    client: OpenAI,
    model: str,
    answer: str,
    grading_notes: list[str],
) -> tuple[float, str]:
    """答案完整性：答案是否覆盖了所有评分要点。

    和 metrics.py 的 keyword_coverage（代码规则关键词匹配）不同，
    这里让 LLM 做语义覆盖判断——因为评分要点和答案用词可能不同。

    Returns: (score 0.0~1.0, reason)
    """
    if not grading_notes:
        return 1.0, "无评分要点"
    if not answer or not answer.strip():
        return 0.0, "答案为空"

    notes_text = "\n".join(f"- {n}" for n in grading_notes)
    prompt = f"""你是答案完整性评判专家。判断答案是否覆盖了所有评分要点。

【评分要点】
{notes_text}

【答案】
{answer}

对每个评分要点，判断答案是否提到了：
- 完全覆盖（要点全部内容都提到了）→ 1.0
- 部分覆盖（只提到一半/模糊提到）→ 0.5
- 未覆盖（没提到）→ 0.0

输出 JSON:
{{"coverage_per_note": [1.0, 0.5, ...], "overall_score": 0.0-1.0, "reason": "评判理由"}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        score = float(data.get("overall_score", 0.0))
        reason = data.get("reason", "")
        return min(1.0, max(0.0, score)), reason
    except (json.JSONDecodeError, Exception) as e:
        return 0.0, f"完整性评判解析失败: {e}"


# ──────────────────── 聚合辅助 ────────────────────

def _avg(values: list[float | None]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def summarize_metrics(results: list[dict]) -> dict:
    """聚合一批用例的评测结果为 summary dict。

    输入: results 每条包含 recall@k, precision@k, faithfulness, completeness
    输出: 各维度均值 + 按 type 分组均值
    """
    summaries = {}
    for metric_key in ("recall@3", "precision@3", "faithfulness", "completeness"):
        vals = [r.get(metric_key) for r in results]
        summaries[metric_key] = _avg(vals)

    # 按 type 分组
    by_type: dict[str, list[dict]] = {}
    for r in results:
        t = r.get("type", "unknown")
        by_type.setdefault(t, []).append(r)

    summaries["by_type"] = {}
    for t, items in by_type.items():
        summaries["by_type"][t] = {}
        for k in ("recall@3", "precision@3", "faithfulness", "completeness"):
            vals = [i.get(k) for i in items]
            summaries["by_type"][t][k] = _avg(vals)

    summaries["total_cases"] = len(results)
    return summaries
