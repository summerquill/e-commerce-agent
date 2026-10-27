"""Rerank 探针实验：dense top-5 候选交给 bge-reranker-base 精排，不改生产 Retriever。

目的：验证 CrossEncoder 能否在相同 Top-5 候选上把真正含答案的 chunk 排进
Top 3；重点观察纯 dense 漏掉的跨文档干扰 case（q_020 / q_034，"破损"同现
于配送说明与退换货政策两份文档，BM25/双信号补位都修不了）。

实验口径：
- 候选来自生产 ChromaBackend（HNSW 近似检索）的 dense top-5；
  41 chunks 小库下与 NumpyBackend 暴力扫描结果一致（此前已实测）
- 重排输入是 chunk 正文全文（含【文档 · 章节】标题前缀），不传裸 chunk ID
- 指标复用 run_eval.retrieval_metrics，与基线/混合检索口径逐位一致
- 逐 case 记录四元组：chunk_id / dense 原排名 / rerank 分数 / rerank 新排名

跑法（仓库根目录执行；模型已缓存后完全离线，不调用任何云 API）：
    EMBEDDING_BACKEND=local python tests/test_reranker.py

首次运行会从 HuggingFace 下载 BAAI/bge-reranker-base（约 1.1GB）到本地缓存。

输出：
- 控制台：逐 case 重排明细 + rerank 前后指标对比
- app/agent/rag/eval/results/rerank-probe/：traces.json + summary.json
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须在 import settings 之前固化实验环境：本地 bge + 纯 dense 候选
os.environ.setdefault("EMBEDDING_BACKEND", "local")
os.environ.setdefault("RETRIEVAL_MODE", "dense")

from app.agent.rag.backends import create_backend  # noqa: E402
from app.agent.rag.embedder import create_embedder  # noqa: E402
from app.agent.rag.eval.run_eval import load_dataset, retrieval_metrics  # noqa: E402
from app.agent.rag.retriever import KnowledgeRetriever  # noqa: E402
from app.config.settings import settings  # noqa: E402

DATASET = ROOT / "app/agent/rag/eval/datasets/rag_eval.example.json"
OUTPUT_DIR = ROOT / "app/agent/rag/eval/results/rerank-probe"
RERANKER_MODEL = "BAAI/bge-reranker-base"


def pick_device() -> str:
    """与 LocalBgeEmbedder 相同的探测规则：mps 仅 Apple Silicon 可信（Intel Mac 会误报）。"""
    try:
        import torch

        if platform.machine() == "arm64" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def aggregate(rows: list[dict], metric: str) -> float | None:
    values = [r[metric] for r in rows if r.get(metric) is not None]
    return round(sum(values) / len(values), 4) if values else None


def main() -> None:
    # ── 1. 生产同款 dense 检索链路（ChromaBackend / HNSW） ──────────────
    retriever = KnowledgeRetriever(
        embedder=create_embedder(),
        backend=create_backend(
            "chroma", persist_dir=str(ROOT / settings.chroma_persist_dir)
        ),
    )
    retriever.load()
    print(f"dense 检索就绪: backend=chroma size={retriever.size}")

    # ── 2. CrossEncoder 精排模型 ───────────────────────────────────────
    from sentence_transformers import CrossEncoder

    device = pick_device()
    print(f"加载 reranker: {RERANKER_MODEL} (device={device}) ...")
    reranker = CrossEncoder(RERANKER_MODEL, device=device, max_length=512)

    # ── 3. 全量 32 条：dense top-5 → rerank 重排 ────────────────────────
    cases = load_dataset(DATASET)
    traces: list[dict] = []
    for case in cases:
        case_id, query = case["id"], case["query"]
        gold = set(case.get("expected_chunk_ids", []))

        hits = retriever.search(query, top_k=5)
        texts = [h.chunk.text for h in hits]  # 正文全文参与打分
        dense_rank = {h.chunk.chunk_id: i + 1 for i, h in enumerate(hits)}

        scores = reranker.predict([(query, t) for t in texts])
        # 分数降序；同分按 chunk_id 保证全序确定性
        order = sorted(
            range(len(hits)),
            key=lambda i: (-float(scores[i]), hits[i].chunk.chunk_id),
        )
        rerank_rank = {old_idx: r for r, old_idx in enumerate(order, start=1)}

        items = [
            {
                "chunk_id": h.chunk.chunk_id,
                "dense_rank": dense_rank[h.chunk.chunk_id],
                "rerank_score": round(float(scores[idx]), 4),
                "rerank_rank": rerank_rank[idx],
                "is_gold": h.chunk.chunk_id in gold,
            }
            for idx, h in enumerate(hits)
        ]
        items.sort(key=lambda it: it["rerank_rank"])

        before = [h.chunk.chunk_id for h in hits]
        after = [it["chunk_id"] for it in items]
        metrics = {
            "before": retrieval_metrics(gold, before, 5)
            | retrieval_metrics(gold, before, 3)
            | retrieval_metrics(gold, before, 1),
            "after": retrieval_metrics(gold, after, 5)
            | retrieval_metrics(gold, after, 3)
            | retrieval_metrics(gold, after, 1),
        }

        traces.append(
            {
                "case_id": case_id,
                "query": query,
                "gold": sorted(gold),
                "items": items,
                "metrics": metrics,
            }
        )

        # 逐 case 打印重排后明细（gold 行标注）
        print(f"\n[{case_id}] {query}")
        for it in items:
            flag = " ★gold" if it["is_gold"] else ""
            print(
                f"  #{it['rerank_rank']}  score={it['rerank_score']:>8.4f}"
                f"  dense#{it['dense_rank']}  {it['chunk_id']}{flag}"
            )

    # ── 4. 汇总指标对比 + 落盘 ─────────────────────────────────────────
    # 口径说明：候选池固定为 dense top-5；before = dense 原序直接截前 k，
    # after = rerank 重排后截前 k。hit@k 对多 gold 过于宽松（任一 gold
    # 命中即满分），rerank 的价值主要看 recall@1 / recall@3 / mrr@3。
    metrics_keys = [
        "recall@1", "hit@3", "recall@3", "precision@3", "mrr@3",
        "hit@5", "recall@5", "mrr@5",
    ]
    summary_rows = {
        k: {
            "before": aggregate([t["metrics"]["before"] for t in traces], k),
            "after": aggregate([t["metrics"]["after"] for t in traces], k),
        }
        for k in metrics_keys
    }
    print("\n" + "=" * 64)
    print("Rerank 前后指标对比（32 条，候选池=dense top-5，unanswerable 不计）")
    print("before = dense 原序直接截前 k；after = rerank 重排后截前 k")
    print(f"{'指标':<14}{'dense(不rerank)':>16}{'rerank后':>14}")
    for k, v in summary_rows.items():
        b, a = v["before"], v["after"]
        b_s = f"{b:>16.4f}" if b is not None else f"{'—':>16}"
        a_s = f"{a:>14.4f}" if a is not None else f"{'—':>14}"
        print(f"{k:<14}{b_s}{a_s}")

    # 重点 case：跨文档干扰（rerank 的预期靶子），打印前后 top-3 明细
    print("\n重点 case 重排明细（dense top-3 → rerank top-3）：")
    for t in traces:
        if t["case_id"] in {"q_020", "q_034", "q_011"}:
            by_rank = {it["dense_rank"]: it for it in t["items"]}
            dense3 = " > ".join(
                f"{by_rank[i]['chunk_id']}{'★' if by_rank[i]['is_gold'] else ''}"
                for i in sorted(by_rank)[:3]
            )
            rerank3 = " > ".join(
                f"{it['chunk_id']}{'★' if it['is_gold'] else ''}"
                for it in t["items"][:3]
            )
            print(f"  {t['case_id']}: {t['query']}")
            print(f"    dense 前3 : {dense3}")
            print(f"    rerank 前3: {rerank3}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {"model": RERANKER_MODEL, "case_count": len(traces), "metrics": summary_rows},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n明细已落盘: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
