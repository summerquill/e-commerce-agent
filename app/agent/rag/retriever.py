"""知识库检索器：query → 向量化 → 委托后端检索。

设计上 retriever 只负责"问句怎么变向量""结果怎么聚合"，
存储和打分都交给 VectorBackend 实现，对应两套：
- NumpyBackend：手写余弦 + JSON 持久化（教学透明，零依赖）
- ChromaBackend：嵌入式向量数据库 + HNSW（生产代表性）

校验逻辑：加载后比对 backend 持久化的 embedding_model 与当前 Embedder.model，
不一致直接报错——避免"换了 embedding 但还在用老索引"这种隐蔽问题。
"""

from __future__ import annotations

import time

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.embedder import Embedder
from app.agent.rag.logging_utils import get_rag_logger

__all__ = ["KnowledgeRetriever", "HybridRetriever", "RetrievedChunk"]


def _brief_hits(hits: list[RetrievedChunk]) -> str:
    """日志用的命中摘要：chunk_id:score，避免打印整段正文。"""
    return ", ".join(f"{h.chunk.chunk_id}:{h.score:.4f}" for h in hits)

# 文档级聚合检索参数：
# - 触发：top-k 结果中与 top-1 同文档的 chunk 数 >= DOC_SCOPE_MIN_HITS，
#   说明 query 打在整个文档主题上（如"咨询会员权益"），而非某个具体小节
# - 上限：文档 chunk 数超过 DOC_SCOPE_MAX_CHUNKS 时不整篇展开，
#   防止单次工具调用注入过大的上下文（会员权益恰好 10 个 chunk，可展开）
DOC_SCOPE_MIN_HITS = 3
DOC_SCOPE_MAX_CHUNKS = 10


def expand_doc_scope(
    hits: list[RetrievedChunk],
    top_k: int,
    doc_hits_fn,
    *,
    trace: bool = False,
) -> list[RetrievedChunk]:
    """宽泛问题的文档级聚合：top-k 被同一文档占据时，返回整篇文档。

    展开规则：
    1. 该文档全部 chunk（含 query 真实相似度）按分数降序排在最前，
       同分按 chunk_id 排序保证确定性；
    2. 原始 top-k 中不属于该文档的命中保留在后面（跨文档信号不丢弃）；
    3. 按 chunk_id 去重，展开条目标记 expanded=True 供工具层观测。

    未触发（命中过于分散 / 文档过大）时原样返回，行为与普通检索一致。
    trace=True 时记录判定原因，供生产 tool 层排查；评测默认关闭。
    """
    logger = get_rag_logger() if trace else None
    if not hits:
        if logger:
            logger.info("[rag.expand] skipped reason=no_hits")
        return hits
    top1_doc = hits[0].chunk.doc
    if not top1_doc:
        if logger:
            logger.info("[rag.expand] skipped reason=no_doc")
        return hits
    same_doc_in_top = sum(
        1 for h in hits[:top_k] if h.chunk.doc == top1_doc
    )
    if same_doc_in_top < DOC_SCOPE_MIN_HITS:
        if logger:
            logger.info(
                "[rag.expand] skipped reason=too_scattered doc=%s "
                "same_doc_in_top=%d threshold=%d",
                top1_doc, same_doc_in_top, DOC_SCOPE_MIN_HITS,
            )
        return hits

    doc_hits = doc_hits_fn(top1_doc)
    if not doc_hits:
        if logger:
            logger.info(
                "[rag.expand] skipped reason=doc_empty doc=%s", top1_doc,
            )
        return hits
    if len(doc_hits) > DOC_SCOPE_MAX_CHUNKS:
        if logger:
            logger.info(
                "[rag.expand] skipped reason=doc_too_large doc=%s "
                "doc_chunks=%d max=%d",
                top1_doc, len(doc_hits), DOC_SCOPE_MAX_CHUNKS,
            )
        return hits

    doc_ids = {h.chunk.chunk_id for h in doc_hits}
    extras = [h for h in hits if h.chunk.chunk_id not in doc_ids]
    ranked = sorted(doc_hits, key=lambda h: (-h.score, h.chunk.chunk_id))
    for h in ranked:
        h.expanded = True
    result = ranked + extras
    if logger:
        logger.info(
            "[rag.expand] expanded doc=%s same_doc_in_top=%d doc_chunks=%d "
            "hits_before=%d hits_after=%d hits=[%s]",
            top1_doc, same_doc_in_top, len(doc_hits),
            len(hits), len(result), _brief_hits(result),
        )
    return result


class KnowledgeRetriever:
    """对上层暴露统一接口，对下委托给具体 backend。"""

    def __init__(self, embedder: Embedder, backend: VectorBackend):
        self._embedder = embedder
        self._backend = backend
        self._loaded = False

    @property
    def backend(self) -> VectorBackend:
        return self._backend

    @property
    def size(self) -> int:
        return self._backend.size()

    def load(self) -> None:
        if self._loaded:
            return
        self._backend.load()

        expected = self._backend.expected_embedding_model()
        if expected and expected != self._embedder.model:
            raise ValueError(
                f"索引模型({expected}) 与当前 Embedder 模型"
                f"({self._embedder.model}) 不一致，请重建索引。"
            )
        self._loaded = True

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not self._loaded:
            self.load()
        start = time.perf_counter()
        q_vec = self._embedder.encode_one(query)
        encode_ms = (time.perf_counter() - start) * 1000
        search_start = time.perf_counter()
        hits = self._backend.search(q_vec, top_k=top_k)
        search_ms = (time.perf_counter() - search_start) * 1000
        get_rag_logger().info(
            "[rag.dense] backend=%s query=%r top_k=%d "
            "encode=%.1fms search=%.1fms hits=[%s]",
            type(self._backend).__name__, query, top_k,
            encode_ms, search_ms, _brief_hits(hits),
        )
        return hits

    def encode_query(self, query: str) -> list[float]:
        """暴露 query 向量化给组合检索器（HybridRetriever）复用。"""
        return self._embedder.encode_one(query)

    def search_expanded(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """带文档级聚合的检索：生产工具层使用，评测脚本仍走 search()。

        注意 query 会编码两次（骨架 + 文档打分各一次），本地模型约十几 ms，
        换取 search() 零改动以保持评测口径稳定。
        """
        hits = self.search(query, top_k=top_k)
        return expand_doc_scope(
            hits,
            top_k,
            lambda doc: self._backend.search_doc(doc, self.encode_query(query)),
            trace=True,
        )


class HybridRetriever:
    """dense + sparse 双路检索：「dense 保底 + sparse 补位」主从融合。

    组合方式：KnowledgeRetriever（语义）+ LexicalIndex（BM25 词法），
    对外暴露与 KnowledgeRetriever 相同的 load/search 接口，
    上层（knowledge.py / 评测 adapter）无感切换。

    为什么不用标准 RRF 融合：实测回退（32 条评测集 MRR@5 0.887→0.847）。
    RRF 的 1/(k+rank) 加性融合抹平 dense 名次差（第 1 名与第 6 名分差
    仅约 5%），而小库上 BM25 噪声高，"dense 尾部 + BM25 头部"的垃圾
    chunk 凭双份加分即可反超 dense 冠军。详见
    docs/rag-hybrid-retrieval-case-study.md。

    v2 契约（主从结构，sparse 永不洗掉 dense 头部）：
    1. dense 前 top_k 条为骨架，顺序不动；
    2. 补位候选须同时满足两个独立信号：
       a) BM25 名次 <= sparse_top（词汇强命中）
       b) dense 名次 > top_k 且 <= fetch_depth（语义相关但差一名）
       多个候选时按 dense 名次优先取第一个，替换骨架末位，至多补 1 个；
    3. 两路 fetch_depth 之外的 chunk 不参与（垃圾防火墙）。

    返回条目的 score 一律保留 dense 余弦分：排序由结构（补位/精排）决定，
    分值仅作参考。

    reranker（可选）：注入后对融合结果再做一次 CrossEncoder 精排。
    精排只调整顺序、不增删候选——补位负责「候选覆盖」（recall@5），
    rerank 负责「排序质量」（recall@1），分工互补，探针数据见案例文档第 6 节。
    """

    def __init__(
        self,
        dense_retriever: KnowledgeRetriever,
        lexical_index,  # LexicalIndex，鸭子类型避免本模块依赖 jieba
        fetch_depth: int = 10,
        sparse_top: int = 2,
        reranker=None,  # Reranker | None，注入后对最终结果精排
    ):
        self._dense = dense_retriever
        self._lexical = lexical_index
        self._fetch_depth = fetch_depth
        self._sparse_top = sparse_top
        self._reranker = reranker

    @property
    def size(self) -> int:
        return self._dense.size

    def load(self) -> None:
        # lexical 延迟到 search 才自加载也可以，这里 eager load
        # 是为了让"BM25 索引缺失"在初始化时就报错，而不是查询时
        self._dense.load()
        self._lexical.load()

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        dense_start = time.perf_counter()
        dense_hits = self._dense.search(query, top_k=self._fetch_depth)
        dense_ms = (time.perf_counter() - dense_start) * 1000
        bm25_start = time.perf_counter()
        lexical_hits = self._lexical.search(query, top_k=self._fetch_depth)
        bm25_ms = (time.perf_counter() - bm25_start) * 1000

        skeleton = dense_hits[:top_k]
        tail = dense_hits[top_k:]

        bm25_top = lexical_hits[: self._sparse_top]
        bm25_top_ids = {hit.chunk.chunk_id for hit in bm25_top}
        substitute = next(
            (hit for hit in tail if hit.chunk.chunk_id in bm25_top_ids), None
        )
        if substitute is not None:
            skeleton = skeleton[:-1] + [substitute]

        get_rag_logger().info(
            "[rag.hybrid] query=%r fetch_depth=%d sparse_top=%d "
            "dense_fetch=%d bm25_fetch=%d dense_ms=%.1f bm25_ms=%.1f "
            "dense_skeleton=[%s] dense_tail=[%s] bm25_top=[%s] substitute=%s",
            query, self._fetch_depth, self._sparse_top,
            len(dense_hits), len(lexical_hits), dense_ms, bm25_ms,
            _brief_hits(skeleton), _brief_hits(tail), _brief_hits(bm25_top),
            substitute.chunk.chunk_id if substitute is not None else "none",
        )

        if self._reranker is not None:
            skeleton = self._rerank(query, skeleton)
        get_rag_logger().info(
            "[rag.hybrid] final=[%s]", _brief_hits(skeleton),
        )
        return skeleton

    def search_expanded(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """带文档级聚合的检索：融合结果触发条件后整篇展开文档。

        文档打分复用 dense 后端的 search_doc（真实余弦分）；
        展开判断基于融合后的最终名次，补位/精排不改变触发语义。
        """
        hits = self.search(query, top_k=top_k)
        q_vec = self._dense.encode_query(query)
        return expand_doc_scope(
            hits, top_k,
            lambda doc: self._dense.backend.search_doc(doc, q_vec),
            trace=True,
        )

    def _rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """CrossEncoder 精排：按 rerank 分数降序重排，不改候选集。

        同分按 chunk_id 排序保证确定性（可复现评测）；
        score 字段仍保留 dense 余弦分，排序依据由 rerank 排名体现。
        """
        start = time.perf_counter()
        scores = self._reranker.score(query, [h.chunk.text for h in items])
        elapsed_ms = (time.perf_counter() - start) * 1000
        ranked = sorted(
            zip(items, scores),
            key=lambda pair: (-pair[1], pair[0].chunk.chunk_id),
        )
        after = ", ".join(
            f"{item.chunk.chunk_id}:rerank={score:.4f}/dense={item.score:.4f}"
            for item, score in ranked
        )
        get_rag_logger().info(
            "[rag.rerank] model=%s candidates=%d elapsed=%.1fms "
            "before=[%s] after=[%s]",
            getattr(self._reranker, "model", type(self._reranker).__name__),
            len(items), elapsed_ms, _brief_hits(items), after,
        )
        return [item for item, _ in ranked]
