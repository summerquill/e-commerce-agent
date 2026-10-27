"""Hybrid 检索离线单测（不需要 API key，不触网，秒级）。

覆盖四层：
1. tokenize：jieba 切词的确定性与纯标点过滤
2. LexicalIndex：BM25 索引 build/save/load/search 全链路（含 roundtrip 一致性）
3. HybridRetriever：双信号补位规则（BM25 top-2 ∩ dense 尾部才补位）、
   骨架保序、至多补 1、fetch_depth 垃圾防火墙、top_k 边界、load 委托
   —— dense/sparse 两路均用 stub，钉死「dense 保底 + sparse 补位」契约
4. reranker 注入（补位后精排）：只改顺序不改候选集、score 保留 dense 分、
   同分按 chunk_id 稳定、精排输入=补位后 top-5 全文、knowledge.py
   hybrid_rerank 装配路由——reranker 用 stub，不加载真实模型

跑法（在项目 .venv 中，仓库根目录执行）：
    python -m unittest tests.test_hybrid_retriever -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.rag.backends.base import RetrievedChunk  # noqa: E402
from app.agent.rag.chunker import Chunk  # noqa: E402
from app.agent.rag.lexical import LexicalIndex, tokenize  # noqa: E402


# ──────────────────── 工具 ────────────────────

def _chunk(cid: str, text: str | None = None) -> Chunk:
    return Chunk(chunk_id=cid, doc="d", section="s", text=text or f"【d · s】{cid} 正文内容")


def _dense_eight() -> list[Chunk]:
    """D0..D7 共 8 条：top_k=5 时骨架=D0..D4，尾部候选池=D5..D7。"""
    return [_chunk(f"D{i}") for i in range(8)]


class _StubSearcher:
    """固定返回预设 chunk 列表的检索器 stub（query 无关，score 按名次递减）。"""

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self.load_called = False
        self.last_top_k: int | None = None

    def load(self) -> None:
        self.load_called = True

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.last_top_k = top_k
        return [
            RetrievedChunk(chunk=c, score=float(len(self._chunks) - i))
            for i, c in enumerate(self._chunks[:top_k])
        ]


# ──────────────────── tokenize ────────────────────

class TestTokenize(unittest.TestCase):

    def test_deterministic(self):
        q = "成长值达到5000可以升级黑卡会员吗？"
        self.assertEqual(tokenize(q), tokenize(q))

    def test_filters_punctuation_and_whitespace(self):
        for tok in tokenize("你好， 世界！退款。？"):
            self.assertTrue(tok.strip())
            self.assertTrue(any(ch.isalnum() for ch in tok))

    def test_keeps_numbers_and_chinese(self):
        toks = tokenize("满5000升级")
        self.assertIn("5000", toks)
        self.assertTrue(any("满" in t or "升级" in t for t in toks))


# ──────────────────── LexicalIndex ────────────────────

class TestLexicalIndex(unittest.TestCase):

    CORPUS = [
        _chunk("d#00-00", "【会员权益 · 等级体系】成长值达到5000可升级黑卡会员"),
        _chunk("d#01-00", "【退换货政策 · 退货流程】商品签收后7天内可申请退货退款"),
        _chunk("d#02-00", "【配送说明 · 破损处理】包裹破损请拍照联系客服补偿"),
        _chunk("d#03-00", "【促销活动 · 签到】每日签到领取优惠券"),
        _chunk("d#04-00", "【配送说明 · 范围】配送范围覆盖全国大部分城市"),
    ]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "kb_bm25.json"
        self.index = LexicalIndex(self.path)
        self.index.build(list(self.CORPUS), embedding_model="bge-test")

    def test_search_ranks_keyword_hit_first(self):
        hits = self.index.search("成长值5000黑卡门槛", top_k=3)
        self.assertEqual(hits[0].chunk.chunk_id, "d#00-00")
        self.assertGreater(hits[0].score, 0.0)

    def test_save_load_roundtrip_same_results(self):
        self.index.save()
        reloaded = LexicalIndex(self.path)
        reloaded.load()
        for query in ("成长值5000黑卡门槛", "包裹破损怎么补偿", "完全无关的查询词组"):
            a = self.index.search(query, top_k=5)
            b = reloaded.search(query, top_k=5)
            self.assertEqual(
                [(h.chunk.chunk_id, h.score) for h in a],
                [(h.chunk.chunk_id, h.score) for h in b],
                f"roundtrip 后检索结果不一致: {query}",
            )

    def test_saved_payload_contains_chunks_and_corpus(self):
        self.index.save()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["embedding_model"], "bge-test")
        self.assertEqual(len(data["chunks"]), len(self.CORPUS))
        self.assertEqual(len(data["tokenized_corpus"]), len(self.CORPUS))

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            LexicalIndex(Path(self.tmp.name) / "nope.json").load()

    def test_search_respects_top_k(self):
        self.assertEqual(len(self.index.search("退款", top_k=2)), 2)


# ──────────────── HybridRetriever（dense 保底 + sparse 补位契约） ────────────────

class TestHybridRetriever(unittest.TestCase):
    """dense/sparse 双路均 stub 化，验证「dense 保底 + sparse 补位」契约。

    stub score 语义：_StubSearcher 返回 score=float(len-i)，即按名次递减。
    因此可用 score 断言区分结果条目来自哪一路（dense 分 vs lexical 分）。
    """

    def _hybrid(self, dense: list[Chunk], lexical: list[Chunk], **kwargs):
        from app.agent.rag.retriever import HybridRetriever

        d, s = _StubSearcher(dense), _StubSearcher(lexical)
        return HybridRetriever(dense_retriever=d, lexical_index=s, **kwargs), d, s

    def test_double_signal_substitutes_last_slot(self):
        # D5：dense 第 6 名（尾部）+ BM25 第 2 名（强命中）→ 双信号成立
        # → 替换骨架末位 D4，前 4 名保序不动
        hybrid, _, _ = self._hybrid(
            dense=_dense_eight(),
            lexical=[_chunk("X"), _chunk("D5")],
        )
        hits = hybrid.search("q", top_k=5)
        self.assertEqual(
            [h.chunk.chunk_id for h in hits], ["D0", "D1", "D2", "D3", "D5"]
        )

    def test_substitute_keeps_dense_score(self):
        # 返回条目的 score 一律是 dense 余弦分：
        # D5 在 dense stub 中排第 6（i=5）→ score=8-5=3.0；
        # 若误用 lexical 分则为 1.0，可借此区分来源。
        hybrid, _, _ = self._hybrid(
            dense=_dense_eight(),
            lexical=[_chunk("X"), _chunk("D5")],
        )
        hits = hybrid.search("q", top_k=5)
        by_id = {h.chunk.chunk_id: h.score for h in hits}
        self.assertEqual(by_id["D5"], 3.0)

    def test_bm25_weak_hit_no_substitute(self):
        # D5 在 dense 尾部但 BM25 仅第 3 名（未进 top-2）→ 词汇证据不足，不补位
        hybrid, _, _ = self._hybrid(
            dense=_dense_eight(),
            lexical=[_chunk("X"), _chunk("Y"), _chunk("D5")],
        )
        hits = hybrid.search("q", top_k=5)
        self.assertEqual(
            [h.chunk.chunk_id for h in hits], ["D0", "D1", "D2", "D3", "D4"]
        )

    def test_bm25_hit_inside_skeleton_no_change(self):
        # BM25 第 1 名 X 已在 dense 骨架内（第 3 名）→ 不产生补位，结果原样
        dense = [_chunk("D0"), _chunk("D1"), _chunk("X"), _chunk("D2"),
                 _chunk("D3"), _chunk("D4"), _chunk("D5"), _chunk("D6")]
        hybrid, _, _ = self._hybrid(dense=dense, lexical=[_chunk("X"), _chunk("Y")])
        hits = hybrid.search("q", top_k=5)
        self.assertEqual(
            [h.chunk.chunk_id for h in hits], ["D0", "D1", "X", "D2", "D3"]
        )

    def test_bm25_hit_beyond_fetch_depth_no_substitute(self):
        # Z：BM25 第 1 名但 dense 第 11 名（超出 fetch_depth=10 候选池）
        # → 垃圾防火墙，不参与补位
        dense = [_chunk(f"D{i}") for i in range(10)] + [_chunk("Z"), _chunk("W")]
        hybrid, d, _ = self._hybrid(dense=dense, lexical=[_chunk("Z"), _chunk("Y")])
        hits = hybrid.search("q", top_k=5)
        self.assertEqual(d.last_top_k, 10)
        self.assertEqual(
            [h.chunk.chunk_id for h in hits], ["D0", "D1", "D2", "D3", "D4"]
        )

    def test_at_most_one_substitute_prefers_dense_rank(self):
        # D5（dense 第 6）、D6（dense 第 7）同时双信号，且 D6 的 BM25 名次更高
        # → 至多补 1 个，且按 dense 名次优先取 D5；D6 不得出现
        hybrid, _, _ = self._hybrid(
            dense=_dense_eight(),
            lexical=[_chunk("D6"), _chunk("D5")],
        )
        hits = hybrid.search("q", top_k=5)
        self.assertEqual(
            [h.chunk.chunk_id for h in hits], ["D0", "D1", "D2", "D3", "D5"]
        )

    def test_top_k_generalization(self):
        # top_k=3：骨架=D0..D2，尾部候选=D3..，替换位=第 3 名
        # D3：dense 第 4 名 + BM25 第 2 名 → 替换 D2
        hybrid, _, _ = self._hybrid(
            dense=_dense_eight(),
            lexical=[_chunk("X"), _chunk("D3")],
        )
        hits = hybrid.search("q", top_k=3)
        self.assertEqual([h.chunk.chunk_id for h in hits], ["D0", "D1", "D3"])

    def test_top_k_exceeds_candidates_returns_all_dense(self):
        # top_k 超过候选总量 → 返回 dense 全量，尾部无候选即无补位
        dense = [_chunk(f"D{i}") for i in range(6)]
        hybrid, _, _ = self._hybrid(dense=dense, lexical=[_chunk("X")])
        hits = hybrid.search("q", top_k=8)
        self.assertEqual(
            [h.chunk.chunk_id for h in hits], ["D0", "D1", "D2", "D3", "D4", "D5"]
        )

    def test_non_positive_top_k_returns_empty(self):
        hybrid, _, _ = self._hybrid(dense=_dense_eight(), lexical=[_chunk("X")])
        self.assertEqual(hybrid.search("q", top_k=0), [])
        self.assertEqual(hybrid.search("q", top_k=-1), [])

    def test_both_empty_returns_empty(self):
        hybrid, _, _ = self._hybrid(dense=[], lexical=[])
        self.assertEqual(hybrid.search("q", top_k=5), [])

    def test_load_delegates_to_both_routes(self):
        hybrid, d, s = self._hybrid(dense=[_chunk("A")], lexical=[_chunk("A")])
        hybrid.load()
        self.assertTrue(d.load_called)
        self.assertTrue(s.load_called)

    def test_fetch_depth_passed_to_both_routes(self):
        dense = [_chunk(f"D{i}") for i in range(12)]
        lexical = [_chunk(f"L{i}") for i in range(12)]
        hybrid, d, s = self._hybrid(dense=dense, lexical=lexical)
        hybrid.search("q", top_k=5)
        self.assertEqual(d.last_top_k, 10)
        self.assertEqual(s.last_top_k, 10)


# ──────────────── HybridRetriever × reranker（补位后精排契约） ────────────────

class _StubReranker:
    """按预设 {chunk_id: score} 打分的 reranker stub（不加载真实模型）。

    chunk 正文格式固定为「【d · s】{cid} 正文内容」，据此从 text 反查 id；
    记录每次调用入参，供断言精排只作用于最终 top-k、不改候选集。
    """

    def __init__(self, scores: dict[str, float]):
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        out = []
        for t in texts:
            cid = next(c for c in self._scores if f"】{c} " in t)
            out.append(self._scores[cid])
        return out


class TestHybridRerank(unittest.TestCase):
    """reranker 注入后的精排契约：只改顺序、不改候选集、score 保留 dense 分。"""

    def _hybrid(self, dense: list[Chunk], lexical: list[Chunk], reranker=None):
        from app.agent.rag.retriever import HybridRetriever

        d, s = _StubSearcher(dense), _StubSearcher(lexical)
        return HybridRetriever(
            dense_retriever=d, lexical_index=s, reranker=reranker
        )

    def test_rerank_reorders_by_score(self):
        # stub 让 D3 得最高分 → D3 从第 4 位提到第 1，候选集不变
        rer = _StubReranker(
            {"D3": 5.0, "D0": 4.0, "D1": 3.0, "D2": 2.0, "D4": 1.0}
        )
        hybrid = self._hybrid(
            dense=[_chunk(f"D{i}") for i in range(8)], lexical=[], reranker=rer
        )
        hits = hybrid.search("q", top_k=5)
        self.assertEqual(
            [h.chunk.chunk_id for h in hits], ["D3", "D0", "D1", "D2", "D4"]
        )

    def test_rerank_runs_after_substitution(self):
        # 方案 A 全链路：D5 双信号补进第 5 位 → rerank 把它提到第 1
        rer = _StubReranker(
            {"D5": 9.0, "D0": 4.0, "D1": 3.0, "D2": 2.0, "D3": 1.0}
        )
        hybrid = self._hybrid(
            dense=_dense_eight(), lexical=[_chunk("X"), _chunk("D5")],
            reranker=rer,
        )
        hits = hybrid.search("q", top_k=5)
        self.assertEqual(
            [h.chunk.chunk_id for h in hits], ["D5", "D0", "D1", "D2", "D3"]
        )

    def test_rerank_score_field_keeps_dense_score(self):
        # score 字段保留 dense 余弦分：D5 补位后 dense 分=3.0，
        # rerank 提到第 1 也不改写（排序依据由排名体现，与 v2 契约同构）
        rer = _StubReranker(
            {"D5": 9.0, "D0": 4.0, "D1": 3.0, "D2": 2.0, "D3": 1.0}
        )
        hybrid = self._hybrid(
            dense=_dense_eight(), lexical=[_chunk("X"), _chunk("D5")],
            reranker=rer,
        )
        by_id = {h.chunk.chunk_id: h.score for h in hybrid.search("q", top_k=5)}
        self.assertEqual(by_id["D5"], 3.0)

    def test_rerank_tie_broken_by_chunk_id(self):
        # 全体同分 → 按 chunk_id 升序，保证确定性（可复现评测）
        rer = _StubReranker({f"D{i}": 1.0 for i in range(5)})
        hybrid = self._hybrid(
            dense=[_chunk(f"D{i}") for i in range(8)], lexical=[], reranker=rer
        )
        hits = hybrid.search("q", top_k=5)
        self.assertEqual(
            [h.chunk.chunk_id for h in hits], ["D0", "D1", "D2", "D3", "D4"]
        )

    def test_rerank_receives_final_topk_texts(self):
        # 精排输入必须是「补位后」的最终 top-5 全文，而非 dense 原始候选
        rer = _StubReranker(
            {"D0": 1.0, "D1": 1.0, "D2": 1.0, "D3": 1.0, "D5": 1.0}
        )
        hybrid = self._hybrid(
            dense=_dense_eight(), lexical=[_chunk("X"), _chunk("D5")],
            reranker=rer,
        )
        hybrid.search("q", top_k=5)
        self.assertEqual(len(rer.calls), 1)
        query, texts = rer.calls[0]
        self.assertEqual(query, "q")
        self.assertEqual(len(texts), 5)
        joined = "".join(texts)
        self.assertIn("】D5 ", joined)          # 补位 chunk 以正文参与精排
        self.assertNotIn("D4", joined)           # 被补位换掉的 D4 不该出现在精排输入里

    def test_none_reranker_keeps_v2_behavior(self):
        # reranker=None（dense/hybrid 模式）：零精排，v2 补位契约原样
        hybrid = self._hybrid(
            dense=_dense_eight(), lexical=[_chunk("X"), _chunk("D5")]
        )
        hits = hybrid.search("q", top_k=5)
        self.assertEqual(
            [h.chunk.chunk_id for h in hits], ["D0", "D1", "D2", "D3", "D5"]
        )


# ──────────────── knowledge.py 装配路由（hybrid_rerank 模式） ────────────────

class TestKnowledgeAssembly(unittest.TestCase):
    """settings.retrieval_mode=hybrid_rerank 时必须装配 Reranker 进 HybridRetriever。"""

    def test_hybrid_rerank_assembles_reranker(self):
        from unittest import mock

        from app.agent.tools import knowledge

        knowledge.reset_retriever()
        self.addCleanup(knowledge.reset_retriever)
        fake_retriever = mock.MagicMock()
        with mock.patch.object(knowledge.settings, "retrieval_mode", "hybrid_rerank"), \
             mock.patch.object(knowledge, "create_embedder", return_value=mock.MagicMock()), \
             mock.patch.object(knowledge, "_create_backend_from_settings",
                               return_value=mock.MagicMock()), \
             mock.patch("app.agent.rag.lexical.LexicalIndex", return_value="LEX"), \
             mock.patch("app.agent.rag.reranker.Reranker", return_value="RER") as rer_cls, \
             mock.patch("app.agent.rag.retriever.HybridRetriever",
                        return_value=fake_retriever) as hr_cls:
            got = knowledge._get_retriever()
        self.assertIs(got, fake_retriever)
        kwargs = hr_cls.call_args.kwargs
        self.assertEqual(kwargs["lexical_index"], "LEX")
        self.assertEqual(kwargs["reranker"], "RER")
        rer_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
