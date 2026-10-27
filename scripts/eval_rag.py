"""RAG 评测脚本：跑评测集 → 对比 baseline（朴素检索）效果 → 输出报告。

用法:
    python scripts/eval_rag.py                              # 用默认数据集，跑 baseline
    python scripts/eval_rag.py --dataset rag_dataset.json   # 指定数据集
    python scripts/eval_rag.py --top-k 5                    # 改检索数量
    python scripts/eval_rag.py --limit 20                    # 只跑前 N 条（快速验证）
    python scripts/eval_rag.py --no-judge                   # 跳过 LLM judge（快，只看检索层）

评测指标:
    检索层: recall@K, precision@K
    生成层: faithfulness (LLM judge), completeness (LLM judge)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from app.agent.rag.retriever import KnowledgeRetriever  # noqa: E402
from app.agent.rag.embedder import Embedder  # noqa: E402
from app.agent.rag.backends import create_backend  # noqa: E402
from app.agent.rag.chunker import chunk_markdown_dir  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.evaluation.rag_metrics import (  # noqa: E402
    judge_rag_completeness,
    judge_rag_faithfulness,
    precision_at_k,
    recall_at_k,
    summarize_metrics,
)


GENERATION_PROMPT = """你是电商客服。基于下面检索到的知识库内容回答用户问题。

规则：
1. 如果检索到的内容足以回答 → 直接、准确回答
2. 如果检索到的内容不足以回答 → 明确说"抱歉，我暂时无法回答这个问题"，不要编造
3. 回答简洁、友好，符合客服语气

【检索到的知识库内容】
{contexts_text}

【用户问题】
{query}

【你的回答】"""


def _load_retriever() -> KnowledgeRetriever:
    """创建并加载检索器（复用 settings 配置）。"""
    embedder = create_embedder()
    if settings.rag_backend.lower() == "numpy":
        backend = create_backend("numpy", index_path=Path(settings.kb_index_path))
    elif settings.rag_backend.lower() == "chroma":
        backend = create_backend(
            "chroma",
            persist_dir=Path(settings.chroma_persist_dir),
            collection_name=settings.chroma_collection,
        )
    else:
        raise ValueError(f"未知后端: {settings.rag_backend}")

    retriever = KnowledgeRetriever(embedder=embedder, backend=backend)
    retriever.load()
    return retriever


def _load_dataset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["cases"]


def _retrieve(retriever: KnowledgeRetriever, query: str, top_k: int) -> list[dict]:
    """执行检索，返回 [{"chunk_id", "text", "score"}, ...]。"""
    hits = retriever.search(query, top_k=top_k)
    return [
        {
            "chunk_id": h.chunk.chunk_id,
            "text": h.chunk.text,
            "score": h.score,
        }
        for h in hits
    ]


def _generate_answer(
    client: OpenAI, model: str, query: str, contexts: list[dict]
) -> str:
    """基于检索上下文生成答案。"""
    if not contexts:
        return "抱歉，我暂时无法回答这个问题。"

    contexts_text = "\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in contexts
    )
    prompt = GENERATION_PROMPT.format(contexts_text=contexts_text, query=query)
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"（生成失败: {e}）"


def run_eval(args):
    retriever = _load_retriever()
    print(f"[INFO] Retriever loaded, backend={settings.rag_backend}, chunks={retriever.size}")

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    dataset_path = Path(args.dataset)
    cases = _load_dataset(dataset_path)
    if args.limit:
        cases = cases[: args.limit]
    print(f"[INFO] 评测集: {len(cases)} 条, top_k={args.top_k}, judge={'on' if not args.no_judge else 'off'}")

    results = []
    t0 = time.time()

    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] {case['id']} ({case['type']}): {case['query'][:40]}...")

        # 1. 检索
        retrieved = _retrieve(retriever, case["query"], args.top_k)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        expected_ids = case.get("expected_chunk_ids", [])

        # 2. 检索层指标
        recall = recall_at_k(expected_ids, retrieved_ids, k=args.top_k)
        precision = precision_at_k(expected_ids, retrieved_ids, k=args.top_k)

        print(f"  ✓ 检索: recall@{args.top_k}={recall:.2f}, precision@{args.top_k}={precision:.2f}")
        print(f"    expected={expected_ids}")
        print(f"    retrieved={retrieved_ids}")

        # 3. 生成答案
        answer = _generate_answer(client, args.model, case["query"], retrieved)
        print(f"  ✓ 生成: {answer[:60]}...")

        # 4. 生成层指标（LLM judge）
        faith_score, faith_reason = None, ""
        completeness_score, comp_reason = None, ""
        if not args.no_judge:
            faith_score, faith_reason = judge_rag_faithfulness(
                client, args.model, answer, retrieved
            )
            completeness_score, comp_reason = judge_rag_completeness(
                client, args.model, answer, case.get("grading_notes", [])
            )
            print(f"  ✓ 忠实度: {faith_score:.2f} | 完整性: {completeness_score:.2f}")

        results.append({
            "id": case["id"],
            "type": case.get("type", "unknown"),
            "query": case["query"],
            "expected_chunk_ids": expected_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_scores": [r["score"] for r in retrieved],
            "recall@3": recall,
            "precision@3": precision,
            "answer": answer,
            "faithfulness": faith_score,
            "faithfulness_reason": faith_reason,
            "completeness": completeness_score,
            "completeness_reason": comp_reason,
        })

    # 聚合
    summary = summarize_metrics(results)
    elapsed = time.time() - t0

    # 输出
    output = {
        "run_info": {
            "dataset": str(dataset_path),
            "total_cases": len(cases),
            "top_k": args.top_k,
            "backend": settings.rag_backend,
            "embedding_model": settings.embedding_model,
            "judge_enabled": not args.no_judge,
            "elapsed_seconds": round(elapsed, 1),
        },
        "summary": summary,
        "cases": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印 summary
    print(f"\n{'='*60}")
    print(f"[DONE] 耗时 {elapsed:.1f}s → {output_path}")
    print(f"\n📊 总体指标:")
    for metric in ("recall@3", "precision@3", "faithfulness", "completeness"):
        val = summary.get(metric)
        label = metric.ljust(14)
        print(f"  {label}: {val:.3f}" if val is not None else f"  {label}: (skip)")

    print(f"\n📊 按类型分组:")
    for t, metrics in summary.get("by_type", {}).items():
        print(f"  [{t}]")
        for k, v in metrics.items():
            if v is not None:
                print(f"    {k}: {v:.3f}")


def main():
    parser = argparse.ArgumentParser(description="RAG evaluation script")
    parser.add_argument("--dataset", default="app/evaluation/rag_dataset.json")
    parser.add_argument("--model", default=settings.model_name)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge (faster)")
    parser.add_argument("--output", default="app/evaluation/rag_eval_report.json")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
