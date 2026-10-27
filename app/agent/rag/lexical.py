"""BM25 词汇检索索引：jieba 分词 + rank_bm25.BM25Okapi。

与 VectorBackend（dense 语义检索）互补的一路：
- dense 擅长同义改写（"运费"≈"邮费"），但对专有名词/数字门槛等
  词汇信号不敏感；
- BM25 精确匹配查询词与文档词（如"5000"、"成长值"、"破损"），
  恰好补 dense 的短板。

设计要点：
- 语料 = chunk.text 全文（含【文档 · 章节】前缀），标题词天然入索引，
  不需要额外的字段加权。
- 分词必须确定性：同一文本任何时刻切出的 token 序列一致，
  才能保证索引可复现（build 与 load 不重复分词，语料落盘）。
- 持久化采用 JSON（与 NumpyBackend 同风格，教学透明）：
  存 chunks 原文 + 切好的 token 语料，load 时仅用已存的语料重建
  BM25Okapi，不重新分词——避免 jieba 版本差异导致索引漂移。
- BM25 分值无上界、不可跨查询比较，补位阶段只消费名次（top-2 强命中信号），
  不消费绝对分值。
- embedding_model 仅作来源记录（built 时用了哪个 embedder），
  不做一致性校验：BM25 与 embedding 模型无关，切换模型不影响本索引。
"""

from __future__ import annotations

import json
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk

__all__ = ["LexicalIndex", "tokenize"]


def tokenize(text: str) -> list[str]:
    """jieba 切词，过滤空白与纯标点 token，保证确定性。"""
    return [
        tok
        for tok in jieba.lcut(text)
        if tok.strip() and any(ch.isalnum() for ch in tok)
    ]


class LexicalIndex:
    """BM25 词汇索引：build(内存) / save+load(持久化) / search(查询)。"""

    def __init__(self, index_path: Path):
        self._index_path = Path(index_path)
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._embedding_model: str = ""

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Chunk], embedding_model: str = "") -> None:
        """对给定 chunks 构建 BM25 索引（覆盖式，与 backend.upsert 对齐）。"""
        self._chunks = list(chunks)
        self._embedding_model = embedding_model
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self._chunks])

    def save(self) -> None:
        """持久化 chunks 原文 + token 语料 + 来源元数据。"""
        payload = {
            "embedding_model": self._embedding_model,
            "chunks": [c.to_dict() for c in self._chunks],
            "tokenized_corpus": [
                tokenize(c.text) for c in self._chunks
            ],
        }
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def load(self) -> None:
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"BM25 索引不存在: {self._index_path}\n"
                "请先运行 `python app/scripts/build_kb_index.py` 构建索引。"
            )
        data = json.loads(self._index_path.read_text(encoding="utf-8"))
        self._embedding_model = data.get("embedding_model", "")
        self._chunks = [Chunk(**c) for c in data["chunks"]]
        self._bm25 = BM25Okapi(data["tokenized_corpus"])

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """返回 BM25 分值降序的 Top-K 命中。"""
        if self._bm25 is None:
            self.load()
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in ranked[: max(0, top_k)]
        ]
