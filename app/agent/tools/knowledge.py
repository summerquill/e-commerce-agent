"""知识库检索工具：通过向量检索回答 FAQ、政策类问题。

与查订单/查物流这类「结构化数据查询」工具不同，
search_knowledge 面向非结构化文本（退换货政策、配送说明、FAQ 等），
返回 Top-K 命中片段及其来源，由 LLM 引用回答。

向量后端由 settings.rag_backend 切换：
- numpy ：手写余弦相似度，零依赖，教学透明
- chroma：嵌入式向量数据库，HNSW 索引，工程代表性

检索模式由 settings.retrieval_mode 切换：
- dense ：纯向量语义检索（默认）
- hybrid：dense + BM25 词汇检索，RRF 融合（需先构建 BM25 索引）

为避免每次进程启动都重建索引，单例缓存 Retriever。
"""

from pathlib import Path
from typing import Optional
import time

from app.config.settings import settings
from app.agent.rag.backends import create_backend
from app.agent.rag.embedder import create_embedder
from app.agent.rag.logging_utils import get_rag_logger as _get_rag_logger
from app.agent.rag.retriever import KnowledgeRetriever

_retriever: Optional[KnowledgeRetriever] = None


def _format_search_log(
    query: str,
    top_k: int,
    doc_scope: bool,
    hits: list,
    elapsed_ms: float,
) -> str:
    """把一次检索格式化为可读日志（含每条 chunk 的命中证据）。"""
    lines = [
        (
            f"[search_knowledge] query={query!r} backend={settings.rag_backend} "
            f"mode={settings.retrieval_mode} top_k={top_k} "
            f"doc_scope={doc_scope} hits={len(hits)} "
            f"elapsed={elapsed_ms:.1f}ms"
        )
    ]
    for rank, hit in enumerate(hits, start=1):
        text = " ".join(hit.chunk.text.split())
        preview = text[:80] + ("..." if len(text) > 80 else "")
        lines.append(
            f"  {rank}. score={hit.score:.4f} {hit.chunk.chunk_id} | "
            f"{hit.chunk.doc} · {hit.chunk.section} | "
            f"expanded={getattr(hit, 'expanded', False)}"
        )
        lines.append(f"     text: {preview}")
    return "\n".join(lines)


def _create_backend_from_settings():
    """根据 settings.rag_backend 创建对应后端实例。"""
    name = settings.rag_backend.lower()
    if name == "numpy":
        return create_backend("numpy", index_path=Path(settings.kb_index_path))
    if name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=Path(settings.chroma_persist_dir),
            collection_name=settings.chroma_collection,
        )
    raise ValueError(
        f"未知的 RAG 后端: {settings.rag_backend}（可选: numpy / chroma）"
    )


def _get_retriever():
    """按 settings.retrieval_mode 构建（并缓存）检索器单例。

    - dense ：KnowledgeRetriever，纯向量语义检索
    - hybrid：HybridRetriever，dense 保底 + BM25 sparse 补位，接口不变
    - hybrid_rerank：hybrid 之上注入 Reranker 对最终结果精排
    """
    global _retriever
    if _retriever is None:
        start = time.perf_counter()
        embedder = create_embedder()
        backend = _create_backend_from_settings()
        dense = KnowledgeRetriever(embedder=embedder, backend=backend)

        mode = settings.retrieval_mode.lower()
        if mode == "dense":
            _retriever = dense
        elif mode == "hybrid":
            # 延迟导入：dense 模式无需加载 jieba/rank_bm25
            from app.agent.rag.lexical import LexicalIndex
            from app.agent.rag.retriever import HybridRetriever

            _retriever = HybridRetriever(
                dense_retriever=dense,
                lexical_index=LexicalIndex(Path(settings.kb_bm25_path)),
            )
        elif mode == "hybrid_rerank":
            # 延迟导入：非精排模式无需加载 jieba/rank_bm25/torch
            from app.agent.rag.lexical import LexicalIndex
            from app.agent.rag.reranker import Reranker
            from app.agent.rag.retriever import HybridRetriever

            _retriever = HybridRetriever(
                dense_retriever=dense,
                lexical_index=LexicalIndex(Path(settings.kb_bm25_path)),
                reranker=Reranker(),
            )
        else:
            raise ValueError(
                f"未知检索模式: {settings.retrieval_mode}"
                "（可选: dense / hybrid / hybrid_rerank）"
            )
        _retriever.load()
        index_ref = (
            settings.kb_index_path
            if settings.rag_backend.lower() == "numpy"
            else settings.chroma_persist_dir
        )
        _get_rag_logger().info(
            "[rag.init] backend=%s mode=%s model=%s index=%s elapsed=%.1fms",
            settings.rag_backend,
            settings.retrieval_mode,
            embedder.model,
            index_ref,
            (time.perf_counter() - start) * 1000,
        )
    return _retriever


def reset_retriever() -> None:
    """清空单例缓存（测试或切换后端时使用）。"""
    global _retriever
    _retriever = None


def search_knowledge(query: str, top_k: int = 3) -> dict:
    """检索退换货政策、配送说明、会员权益、FAQ 等知识库内容。

    Returns:
        {
          "success": bool,
          "backend": "numpy" | "chroma",
          "query": str,
          "doc_scope": bool,  # 是否触发文档级聚合（宽泛问题整篇展开）
          "results": [
            {"doc": "...", "section": "...", "score": 0.83, "text": "..."},
            ...
          ],
          "error": "..."  # 仅失败时存在
        }
    """
    if not query or not query.strip():
        _get_rag_logger().warning("[search_knowledge] query 不能为空")
        return {"success": False, "error": "query 不能为空", "query": query, "results": []}

    start = time.perf_counter()
    try:
        retriever = _get_retriever()
    except FileNotFoundError as e:
        _get_rag_logger().error("[search_knowledge] 索引不存在: %s", e)
        return {
            "success": False,
            "error": str(e),
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }
    except Exception as e:
        _get_rag_logger().error("[search_knowledge] 知识库初始化失败: %s", e)
        return {
            "success": False,
            "error": f"知识库初始化失败: {e}",
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }

    top_k = max(1, min(int(top_k or 3), 5))
    # 文档级聚合：宽泛问题（top-k 被同一文档占据）整篇展开；评测脚本不走这里
    search_fn = getattr(retriever, "search_expanded", None) or retriever.search
    hits = search_fn(query, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    doc_scope = any(getattr(h, "expanded", False) for h in hits)
    _get_rag_logger().info(
        _format_search_log(query, top_k, doc_scope, hits, elapsed_ms)
    )

    return {
        "success": True,
        "backend": settings.rag_backend,
        "query": query,
        "doc_scope": doc_scope,
        "results": [
            {
                "doc": h.chunk.doc,
                "section": h.chunk.section,
                "score": round(h.score, 4),
                "text": h.chunk.text,
            }
            for h in hits
        ],
    }
