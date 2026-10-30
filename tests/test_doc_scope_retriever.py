"""文档级聚合检索（doc scope）离线单测。

不调用任何 LLM / Embedding API：
- 向量用手工构造的 2 维向量，通过 StubEmbedder 按查询词查表返回
- 后端复用真实 NumpyBackend（内存 + 临时文件），保证打分逻辑真实
"""

import tempfile
import unittest
from pathlib import Path

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.backends.numpy_backend import NumpyBackend
from app.agent.rag.chunker import Chunk
from app.agent.rag.retriever import (
    DOC_SCOPE_MAX_CHUNKS,
    KnowledgeRetriever,
    HybridRetriever,
    expand_doc_scope,
)
from app.agent.tools import knowledge


# ---------- 测试基建 ----------

class StubEmbedder:
    """按查询词查表返回固定向量；未注册的词返回默认向量。"""

    model = "stub-embedder"

    def __init__(self, mapping: dict[str, list[float]], default: list[float]):
        self._mapping = mapping
        self._default = default

    def encode_one(self, text: str) -> list[float]:
        return self._mapping.get(text, self._default)


def _chunk(chunk_id: str, doc: str, text: str = "") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc=doc, section=chunk_id, text=text or chunk_id)


def _hit(chunk: Chunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=score)


class Counter:
    """记录 doc_hits_fn 是否被调用（未触发时不应调用）。"""

    def __init__(self):
        self.calls = 0

    def __call__(self, doc: str):
        self.calls += 1
        return []


# ---------- expand_doc_scope 纯函数测试 ----------

class TestExpandDocScope(unittest.TestCase):

    def setUp(self):
        self.doc_a = {cid: _chunk(cid, "会员权益") for cid in (
            "会员权益#01-00", "会员权益#02-00", "会员权益#03-00", "会员权益#04-00",
        )}

    def test_empty_hits_unchanged(self):
        counter = Counter()
        self.assertEqual(expand_doc_scope([], 3, counter), [])
        self.assertEqual(counter.calls, 0)

    def test_below_threshold_no_expansion(self):
        hits = [
            _hit(self.doc_a["会员权益#01-00"], 0.9),
            _hit(self.doc_a["会员权益#02-00"], 0.8),
            _hit(_chunk("配送#01-00", "配送"), 0.7),
        ]
        counter = Counter()
        result = expand_doc_scope(hits, 3, counter)
        self.assertEqual([h.chunk.chunk_id for h in result],
                         ["会员权益#01-00", "会员权益#02-00", "配送#01-00"])
        self.assertEqual(counter.calls, 0)  # 未触发就不该去取文档

    def test_expansion_ranks_doc_first_and_keeps_extras(self):
        b1 = _chunk("售后政策#01-00", "售后政策")
        hits = [
            _hit(self.doc_a["会员权益#01-00"], 0.9),
            _hit(self.doc_a["会员权益#02-00"], 0.8),
            _hit(self.doc_a["会员权益#03-00"], 0.7),
            _hit(b1, 0.6),
            _hit(self.doc_a["会员权益#04-00"], 0.5),
        ]

        def doc_hits_fn(doc):
            # 文档打分与原始名次不同（真实余弦），#04 分数最高
            return [
                _hit(self.doc_a["会员权益#04-00"], 0.85),
                _hit(self.doc_a["会员权益#01-00"], 0.75),
                _hit(self.doc_a["会员权益#02-00"], 0.65),
                _hit(self.doc_a["会员权益#03-00"], 0.55),
            ]

        result = expand_doc_scope(hits, 5, doc_hits_fn)
        ids = [h.chunk.chunk_id for h in result]
        # 整篇文档按打分降序在前，跨文档命中 b1 保留在后，#04 去重不重复
        self.assertEqual(ids, [
            "会员权益#04-00", "会员权益#01-00",
            "会员权益#02-00", "会员权益#03-00", "售后政策#01-00",
        ])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(h.expanded for h in result[:4]))
        self.assertFalse(result[4].expanded)

    def test_oversized_doc_not_expanded(self):
        hits = [_hit(_chunk(f"大文档#{i:02d}-00", "大文档"), 0.9 - i * 0.01)
                for i in range(1, 6)]
        big_doc = [_hit(_chunk(f"大文档#{i:02d}-00", "大文档"), 0.5)
                   for i in range(1, DOC_SCOPE_MAX_CHUNKS + 2)]  # 11 个
        result = expand_doc_scope(hits, 5, lambda doc: big_doc)
        self.assertEqual([h.chunk.chunk_id for h in result],
                         [h.chunk.chunk_id for h in hits])
        self.assertFalse(any(h.expanded for h in result))

    def test_equal_scores_sorted_by_chunk_id(self):
        hits = [_hit(self.doc_a[f"会员权益#{i:02d}-00"], 0.9) for i in (1, 2, 3)]
        doc_hits = [
            _hit(self.doc_a[f"会员权益#{i:02d}-00"], 0.5) for i in (3, 1, 2, 4)
        ]
        result = expand_doc_scope(hits, 3, lambda doc: doc_hits)
        self.assertEqual(
            [h.chunk.chunk_id for h in result],
            ["会员权益#01-00", "会员权益#02-00", "会员权益#03-00", "会员权益#04-00"],
        )


# ---------- KnowledgeRetriever.search_expanded（真实 NumpyBackend）----------

class TestSearchExpandedNumpy(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.backend = NumpyBackend(Path(self.tmp.name) / "kb_index.json")
        chunks = [
            _chunk("会员权益#01-00", "会员权益"), _chunk("会员权益#02-00", "会员权益"),
            _chunk("会员权益#03-00", "会员权益"), _chunk("会员权益#04-00", "会员权益"),
            _chunk("售后政策#01-00", "售后政策"), _chunk("售后政策#02-00", "售后政策"),
        ]
        vectors = {
            "会员权益#01-00": [1.0, 0.0],
            "会员权益#02-00": [0.95, 0.05],
            "会员权益#03-00": [0.9, 0.1],
            "会员权益#04-00": [0.8, 0.2],
            "售后政策#01-00": [0.0, 1.0],
            "售后政策#02-00": [0.1, 0.9],
        }
        self.backend.upsert(
            chunks, [vectors[c.chunk_id] for c in chunks], "stub-embedder",
        )
        self.embedder = StubEmbedder(
            mapping={"宽泛问法": [1.0, 0.3], "聚焦问法": [0.2, 1.0]},
            default=[1.0, 0.0],
        )
        self.retriever = KnowledgeRetriever(self.embedder, self.backend)
        self.retriever.load()

    def tearDown(self):
        self.tmp.cleanup()

    def test_broad_query_expands_full_doc(self):
        result = self.retriever.search_expanded("宽泛问法", top_k=3)
        ids = [h.chunk.chunk_id for h in result]
        # query=[1.0,0.3] 归一化后与 #04 方向夹角最小，真实余弦分最高
        self.assertEqual(ids, [
            "会员权益#04-00", "会员权益#03-00",
            "会员权益#02-00", "会员权益#01-00",
        ])
        self.assertTrue(all(h.expanded for h in result))
        # 分数真实降序
        scores = [h.score for h in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_specific_query_keeps_plain_topk(self):
        plain = self.retriever.search("聚焦问法", top_k=3)
        expanded = self.retriever.search_expanded("聚焦问法", top_k=3)
        self.assertEqual(
            [h.chunk.chunk_id for h in expanded],
            [h.chunk.chunk_id for h in plain],
        )
        self.assertFalse(any(h.expanded for h in expanded))

    def test_search_stays_unexpanded_for_eval(self):
        """评测口径保护：search() 永远不做聚合，只返回 top_k 条。"""
        result = self.retriever.search("宽泛问法", top_k=3)
        self.assertEqual(len(result), 3)
        self.assertFalse(any(h.expanded for h in result))

    def test_deterministic_across_calls(self):
        a = [h.chunk.chunk_id for h in self.retriever.search_expanded("宽泛问法", 3)]
        b = [h.chunk.chunk_id for h in self.retriever.search_expanded("宽泛问法", 3)]
        self.assertEqual(a, b)


# ---------- HybridRetriever.search_expanded ----------

class StubLexicalIndex:
    """无词汇检索能力的桩：load/search 与 LexicalIndex 接口一致。"""

    def load(self) -> None:
        pass

    def search(self, query: str, top_k: int):
        return []


class TestHybridExpanded(unittest.TestCase):

    def test_hybrid_expansion_uses_fused_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = NumpyBackend(Path(tmp) / "kb_index.json")
            chunks = [
                _chunk(f"会员权益#{i:02d}-00", "会员权益") for i in range(1, 5)
            ] + [_chunk("售后政策#01-00", "售后政策")]
            vectors = [
                [1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0],
            ]
            backend.upsert(chunks, vectors, "stub-embedder")
            embedder = StubEmbedder({}, default=[1.0, 0.3])
            dense = KnowledgeRetriever(embedder, backend)
            hybrid = HybridRetriever(dense_retriever=dense, lexical_index=StubLexicalIndex())

            result = hybrid.search_expanded("宽泛问法", top_k=5)
            ids = [h.chunk.chunk_id for h in result]
            # 整篇文档在前，跨文档命中（售后政策）保留在末尾
            self.assertEqual(ids, [
                "会员权益#04-00", "会员权益#03-00",
                "会员权益#02-00", "会员权益#01-00", "售后政策#01-00",
            ])
            self.assertTrue(all(h.expanded for h in result[:4]))
            self.assertFalse(result[4].expanded)


# ---------- 工具层接线 ----------

class TestToolWiring(unittest.TestCase):

    def setUp(self):
        self._original = knowledge._retriever

    def tearDown(self):
        knowledge._retriever = self._original

    def test_tool_uses_search_expanded_and_reports_flag(self):
        expanded_hit = _hit(_chunk("会员权益#01-00", "会员权益"), 0.9)
        expanded_hit.expanded = True
        stub = type("StubRetriever", (), {})()
        stub.search_expanded = lambda query, top_k: [expanded_hit]
        knowledge._retriever = stub

        result = knowledge.search_knowledge("随便问问", top_k=3)
        self.assertTrue(result["success"])
        self.assertTrue(result["doc_scope"])
        self.assertEqual(result["results"][0]["section"], "会员权益#01-00")

    def test_tool_falls_back_to_search(self):
        plain_hit = _hit(_chunk("退换货政策#01-00", "退换货政策"), 0.8)
        stub = type("StubRetriever", (), {})()
        stub.search = lambda query, top_k: [plain_hit]  # 无 search_expanded
        knowledge._retriever = stub

        result = knowledge.search_knowledge("退货运费", top_k=3)
        self.assertTrue(result["success"])
        self.assertFalse(result["doc_scope"])
        self.assertEqual(result["results"][0]["doc"], "退换货政策")


if __name__ == "__main__":
    unittest.main()
