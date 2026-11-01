"""RAG 分阶段日志离线单测：不加载真实 embedding / reranker 模型。"""

import tempfile
import unittest
from pathlib import Path

from app.config.settings import settings
from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk
from app.agent.rag.logging_utils import reset_rag_logger
from app.agent.rag.retriever import KnowledgeRetriever, HybridRetriever


def _chunk(chunk_id: str, doc: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc=doc, section=chunk_id, text=chunk_id)


def _hit(chunk_id: str, score: float, doc: str = "会员权益") -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(chunk_id, doc), score=score)


class _StubEmbedder:
    model = "stub-embedding"

    def encode_one(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _StubBackend:
    def __init__(self, hits, doc_hits=None):
        self._hits = hits
        self._doc_hits = doc_hits or []

    def load(self):
        pass

    def expected_embedding_model(self):
        return "stub-embedding"

    def size(self):
        return len(self._hits)

    def search(self, query_vector, top_k):
        return list(self._hits[:top_k])

    def search_doc(self, doc, query_vector):
        return list(self._doc_hits)


class _StubDense:
    """HybridRetriever 用的 dense 桩。"""

    def __init__(self, hits, doc_hits=None):
        self._hits = hits
        self.backend = _StubBackend(hits, doc_hits)

    @property
    def size(self):
        return len(self._hits)

    def load(self):
        pass

    def search(self, query, top_k):
        return list(self._hits[:top_k])

    def encode_query(self, query):
        return [1.0, 0.0]


class _StubLexical:
    def __init__(self, hits):
        self._hits = hits

    def load(self):
        pass

    def search(self, query, top_k):
        return list(self._hits[:top_k])


class _StubReranker:
    model = "stub-reranker"

    def score(self, query, texts):
        # 固定反向打分，便于断言 before/after 确实变化
        return [float(len(texts) - i) for i in range(len(texts))]


class TestRagStageLogs(unittest.TestCase):
    def setUp(self):
        self._original_log_path = settings.rag_log_path
        self.tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmp.name) / "rag.log"
        settings.rag_log_path = str(self.log_path)
        reset_rag_logger()

    def tearDown(self):
        reset_rag_logger()
        settings.rag_log_path = self._original_log_path
        self.tmp.cleanup()

    def _log(self) -> str:
        return self.log_path.read_text(encoding="utf-8")

    def test_dense_stage_logs_candidates_and_latency(self):
        backend = _StubBackend([
            _hit("会员权益#01-00", 0.9),
            _hit("会员权益#02-00", 0.8),
        ])
        retriever = KnowledgeRetriever(_StubEmbedder(), backend)
        retriever.search("会员权益", top_k=2)

        content = self._log()
        self.assertIn("[rag.dense]", content)
        self.assertIn("query='会员权益'", content)
        self.assertIn("encode=", content)
        self.assertIn("search=", content)
        self.assertIn("会员权益#01-00:0.9000", content)
        self.assertNotIn("[rag.rerank]", content)

    def test_hybrid_logs_bm25_substitute_and_rerank(self):
        dense_hits = [
            _hit("会员权益#01-00", 0.90),
            _hit("会员权益#02-00", 0.80),
            _hit("会员权益#03-00", 0.70),
            _hit("会员权益#04-00", 0.60),
            _hit("会员权益#05-00", 0.50),
        ]
        lexical_hits = [
            _hit("会员权益#04-00", 12.0),
            _hit("会员权益#05-00", 9.0),
        ]
        retriever = HybridRetriever(
            dense_retriever=_StubDense(dense_hits),
            lexical_index=_StubLexical(lexical_hits),
            reranker=_StubReranker(),
        )
        retriever.search("退货运费", top_k=3)

        content = self._log()
        self.assertIn("[rag.hybrid]", content)
        self.assertIn("bm25_top=[", content)
        self.assertIn("substitute=会员权益#04-00", content)
        self.assertIn("[rag.rerank]", content)
        self.assertIn("model=stub-reranker", content)
        self.assertIn("before=[", content)
        self.assertIn("after=[", content)
        self.assertIn("[rag.hybrid] final=[", content)

    def test_expand_logs_trigger_and_final_count(self):
        doc_hits = [
            _hit("会员权益#01-00", 0.90),
            _hit("会员权益#02-00", 0.80),
            _hit("会员权益#03-00", 0.70),
            _hit("会员权益#04-00", 0.60),
        ]
        backend = _StubBackend(
            hits=doc_hits[:3],
            doc_hits=doc_hits,
        )
        retriever = KnowledgeRetriever(_StubEmbedder(), backend)
        retriever.search_expanded("会员权益", top_k=3)

        content = self._log()
        self.assertIn("[rag.expand] expanded", content)
        self.assertIn("same_doc_in_top=3", content)
        self.assertIn("doc_chunks=4", content)
        self.assertIn("hits_after=4", content)

    def test_expand_logs_skip_reason_when_scattered(self):
        backend = _StubBackend([
            _hit("会员权益#01-00", 0.9, doc="会员权益"),
            _hit("配送说明#01-00", 0.8, doc="配送说明"),
            _hit("退换货政策#01-00", 0.7, doc="退换货政策"),
        ])
        retriever = KnowledgeRetriever(_StubEmbedder(), backend)
        retriever.search_expanded("随便问问", top_k=3)

        content = self._log()
        self.assertIn("[rag.expand] skipped reason=too_scattered", content)
        self.assertIn("same_doc_in_top=1", content)


if __name__ == "__main__":
    unittest.main()
