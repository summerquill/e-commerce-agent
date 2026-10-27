"""知识库检索器：query → 向量化 → 委托后端检索。

设计上 retriever 只负责"问句怎么变向量""结果怎么聚合"，
存储和打分都交给 VectorBackend 实现，对应两套：
- NumpyBackend：手写余弦 + JSON 持久化（教学透明，零依赖）
- ChromaBackend：嵌入式向量数据库 + HNSW（生产代表性）

校验逻辑：加载后比对 backend 持久化的 embedding_model 与当前 Embedder.model，
不一致直接报错——避免"换了 embedding 但还在用老索引"这种隐蔽问题。
"""

from __future__ import annotations

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.embedder import Embedder

__all__ = ["KnowledgeRetriever", "HybridRetriever", "RetrievedChunk"]


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
        q_vec = self._embedder.encode_one(query)
        return self._backend.search(q_vec, top_k=top_k)


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

    返回条目的 score 一律保留 dense 余弦分：排序由结构决定，分值仅作参考。
    """

    def __init__(
        self,
        dense_retriever: KnowledgeRetriever,
        lexical_index,  # LexicalIndex，鸭子类型避免本模块依赖 jieba
        fetch_depth: int = 10,
        sparse_top: int = 2,
    ):
        self._dense = dense_retriever
        self._lexical = lexical_index
        self._fetch_depth = fetch_depth
        self._sparse_top = sparse_top

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
        dense_hits = self._dense.search(query, top_k=self._fetch_depth)
        lexical_hits = self._lexical.search(query, top_k=self._fetch_depth)

        skeleton = dense_hits[:top_k]
        tail = dense_hits[top_k:]

        bm25_top_ids = {
            hit.chunk.chunk_id for hit in lexical_hits[: self._sparse_top]
        }
        substitute = next(
            (hit for hit in tail if hit.chunk.chunk_id in bm25_top_ids), None
        )
        if substitute is not None:
            skeleton = skeleton[:-1] + [substitute]
        return skeleton
