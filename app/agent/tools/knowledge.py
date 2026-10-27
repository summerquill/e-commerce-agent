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

from app.config.settings import settings
from app.agent.rag.backends import create_backend
from app.agent.rag.embedder import create_embedder
from app.agent.rag.retriever import KnowledgeRetriever

_retriever: Optional[KnowledgeRetriever] = None


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
    """
    global _retriever
    if _retriever is None:
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
        else:
            raise ValueError(
                f"未知检索模式: {settings.retrieval_mode}（可选: dense / hybrid）"
            )
        _retriever.load()
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
          "results": [
            {"doc": "...", "section": "...", "score": 0.83, "text": "..."},
            ...
          ],
          "error": "..."  # 仅失败时存在
        }
    """
    if not query or not query.strip():
        return {"success": False, "error": "query 不能为空", "query": query, "results": []}

    try:
        retriever = _get_retriever()
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": str(e),
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"知识库初始化失败: {e}",
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }

    top_k = max(1, min(int(top_k or 3), 5))
    hits = retriever.search(query, top_k=top_k)

    return {
        "success": True,
        "backend": settings.rag_backend,
        "query": query,
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
