"""
RAG baseline evaluator.

Features:
- Loads test cases from JSON: {"cases": [...]}
- Calls a project-specific adapter to retrieve chunks and generate answers.
- Computes deterministic retrieval metrics from expected_chunk_ids:
  Hit@K, Recall@K, Precision@K, MRR@K.
- Optionally runs Ragas (pinned to the 0.2.x API in requirements.txt).
- Saves per-case traces and aggregate metrics.

Usage:
  python -m evals.run_eval --dataset evals/datasets/rag_eval.json --top-k 5
  python -m evals.run_eval --dataset evals/datasets/rag_eval.json --top-k 5 --with-ragas

Before running:
  Implement evals/rag_adapter.py to connect retrieve() and generate() to your project.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from .rag_adapter import retrieve, generate


def _chunk_id(item: Any) -> str:
    """Normalize a returned chunk/document into a chunk ID."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # Adjust these keys to match the metadata your retriever returns.
        for key in ("chunk_id", "id", "source_id"):
            if item.get(key) is not None:
                return str(item[key])
        metadata = item.get("metadata") or {}
        for key in ("chunk_id", "id", "source_id"):
            if metadata.get(key) is not None:
                return str(metadata[key])
    # LangChain Document-like object
    metadata = getattr(item, "metadata", {}) or {}
    for key in ("chunk_id", "id", "source_id"):
        if metadata.get(key) is not None:
            return str(metadata[key])
    item_id = getattr(item, "id", None)
    if item_id is not None:
        return str(item_id)
    raise ValueError(
        f"Cannot extract chunk ID from {type(item).__name__}. "
        "Update _chunk_id() to match your retriever's return type."
    )


def _chunk_text(item: Any) -> str:
    """Normalize a returned chunk/document into text."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("page_content", "text", "content", "document"):
            if isinstance(item.get(key), str):
                return item[key]
    text = getattr(item, "page_content", None)
    if isinstance(text, str):
        return text
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text
    return str(item)


def retrieval_metrics(expected_ids: list[str], actual_ids: list[str], k: int) -> dict[str, float | None]:
    """
    expected_ids: gold evidence chunk IDs. Empty means unanswerable / no gold evidence.
    For unanswerable cases, retrieval metrics are None rather than treating empty gold
    as a perfect retrieval result.
    """
    if not expected_ids:
        return {
            f"hit@{k}": None,
            f"recall@{k}": None,
            f"precision@{k}": None,
            f"mrr@{k}": None,
        }

    expected = set(expected_ids)
    ranked = actual_ids[:k]
    relevant_positions = [i + 1 for i, cid in enumerate(ranked) if cid in expected]
    hits = len(set(ranked) & expected)

    return {
        f"hit@{k}": 1.0 if hits > 0 else 0.0,
        f"recall@{k}": hits / len(expected),
        f"precision@{k}": hits / k,
        f"mrr@{k}": 1.0 / min(relevant_positions) if relevant_positions else 0.0,
    }


def safe_mean(values: list[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return round(statistics.mean(clean), 4) if clean else None


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        cases = data.get("cases")
    else:
        cases = None
    if not isinstance(cases, list):
        raise ValueError('Dataset must be {"cases": [...]} or a JSON list.')
    for index, case in enumerate(cases):
        if "id" not in case or "query" not in case:
            raise ValueError(f"Case at index {index} must include 'id' and 'query'.")
        case.setdefault("expected_chunk_ids", [])
        case.setdefault("grading_notes", [])
        case.setdefault("answerable", True)
    return cases


def run_eval(
    dataset_path: Path,
    output_dir: Path,
    top_k: int,
    with_ragas: bool,
    no_judge: bool = False,
) -> None:
    cases = load_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces: list[dict[str, Any]] = []
    ragas_rows: list[dict[str, Any]] = []

    for case in cases:
        case_id = case["id"]
        query = case["query"]
        expected_ids = [str(x) for x in case.get("expected_chunk_ids", [])]

        started = time.perf_counter()
        retrieved = retrieve(query, top_k=top_k)
        retrieval_latency_ms = (time.perf_counter() - started) * 1000

        actual_ids = [_chunk_id(item) for item in retrieved]
        contexts = [_chunk_text(item) for item in retrieved]

        answer: str | None = None
        generation_latency_ms: float | None = None
        if not no_judge:
            started = time.perf_counter()
            answer = generate(query, contexts)
            generation_latency_ms = (time.perf_counter() - started) * 1000

        metrics = retrieval_metrics(expected_ids, actual_ids, top_k)
        trace: dict[str, Any] = {
            "id": case_id,
            "type": case.get("type"),
            "difficulty": case.get("difficulty"),
            "query": query,
            "answerable": bool(case["answerable"]),
            "expected_chunk_ids": expected_ids,
            "retrieved_chunk_ids": actual_ids,
            "retrieval_metrics": metrics,
            "retrieved_contexts": contexts,
            "reference": case.get("reference", ""),
            "grading_notes": case.get("grading_notes", []),
            "response": answer,
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        }
        if not no_judge and generation_latency_ms is not None:
            trace["generation_latency_ms"] = round(generation_latency_ms, 2)
        traces.append(trace)

        if (
            not no_judge
            and answer is not None
            and case.get("answerable", True)
            and case.get("reference")
        ):
            ragas_rows.append({
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": case["reference"],
            })

        gen_label = (
            f"generate={generation_latency_ms:.0f}ms"
            if generation_latency_ms is not None
            else "generate=SKIP"
        )
        print(
            f"[{case_id}] retrieve={retrieval_latency_ms:.0f}ms "
            f"{gen_label} retrieved={actual_ids}"
        )

    trace_path = output_dir / "traces.json"
    trace_path.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_names = [f"hit@{top_k}", f"recall@{top_k}", f"precision@{top_k}", f"mrr@{top_k}"]
    summary = {
        "dataset": str(dataset_path),
        "case_count": len(cases),
        "answerable_case_count": sum(bool(c.get("answerable", True)) for c in cases),
        "top_k": top_k,
        "no_judge": no_judge,
        "retrieval_metrics": {
            name: safe_mean([t["retrieval_metrics"][name] for t in traces])
            for name in metric_names
        },
        "latency_ms": {
            "retrieval_mean": safe_mean([t["retrieval_latency_ms"] for t in traces]),
            "generation_mean": (
                safe_mean([t.get("generation_latency_ms") for t in traces])
                if not no_judge else None
            ),
            "total_mean": (
                safe_mean([
                    t["retrieval_latency_ms"] + (t.get("generation_latency_ms") or 0)
                    for t in traces
                ])
                if not no_judge else safe_mean([t["retrieval_latency_ms"] for t in traces])
            ),
        },
        "notes": [
            "Retrieval metrics exclude cases with empty expected_chunk_ids.",
            "Precision@K uses K as denominator even if fewer than K chunks are returned.",
            "For unanswerable cases, add a separate abstention/unsupported-answer review; empty gold IDs are not scored as perfect retrieval.",
        ]
        + (["--no-judge 已启用：跳过 generate 调用与生成层评测。"] if no_judge else []),
    }

    if with_ragas:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                context_precision,
                context_recall,
                faithfulness,
                answer_correctness,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Ragas dependencies missing. Install requirements.txt or run without --with-ragas."
            ) from exc

        if not ragas_rows:
            print("No answerable cases with reference answers; skipping Ragas.")
        else:
            result = evaluate(
                Dataset.from_list(ragas_rows),
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_correctness,
                ],
            )
            # Ragas 0.2.x EvaluationResult supports to_pandas().
            try:
                ragas_df = result.to_pandas()
                ragas_df.to_csv(output_dir / "ragas_per_case.csv", index=False)
                summary["ragas"] = {
                    str(column): (
                        round(float(ragas_df[column].dropna().mean()), 4)
                        if not ragas_df[column].dropna().empty else None
                    )
                    for column in ragas_df.columns
                    if column in {
                        "context_precision", "context_recall",
                        "faithfulness", "answer_correctness"
                    }
                }
            except Exception:
                summary["ragas"] = {"raw_result": str(result)}
            print("Ragas evaluation completed.")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")
    print(f"Traces:  {trace_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/datasets/rag_eval.json")
    parser.add_argument("--output-dir", default="evals/results/baseline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--with-ragas", action="store_true")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="跳过 generate 调用与生成层评测，只跑检索指标（hit/recall/precision/MRR）",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    run_eval(
        Path(args.dataset),
        Path(args.output_dir),
        args.top_k,
        args.with_ragas,
        args.no_judge,
    )


if __name__ == "__main__":
    main()
