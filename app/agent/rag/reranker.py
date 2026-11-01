"""重排器封装：CrossEncoder 精排（BAAI/bge-reranker-base 默认）。

与双塔检索（query、chunk 各自编码成向量再算余弦）不同，
CrossEncoder 把 (query, chunk) 拼接后送入模型做逐 token 交叉注意力，
能区分「主题相同」与「答案所在」——探针实验（32 条）：
recall@1 0.7204 → 0.8387、MRR@5 0.8871 → 0.9677，
详见 docs/rag-hybrid-retrieval-case-study.md 第 6 节。

代价是每对 (query, chunk) 都要跑一次前向，无法预计算索引，
只能用于小候选集（top-5）精排，不能替代召回层。
设备探测与 embedder 共用 pick_torch_device：Apple Silicon 走
MPS，Intel Mac 强制 CPU（torch 在 Intel Mac 会误报 mps available）。
"""

from __future__ import annotations

import time

from app.config.settings import settings
from app.agent.rag.logging_utils import get_rag_logger


class Reranker:
    """CrossEncoder 精排封装（构造即加载模型，随上层单例缓存）。"""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
    ):
        # 延迟 import：dense/hybrid 模式无需加载 torch/sentence-transformers
        from sentence_transformers import CrossEncoder

        from app.agent.rag.embedder import pick_torch_device

        self._model_name = model_name or settings.local_reranker_model
        self._max_length = max_length
        self._batch_size = batch_size
        start = time.perf_counter()
        resolved_device = device or pick_torch_device()
        self._model = CrossEncoder(
            self._model_name,
            max_length=max_length,
            device=resolved_device,
        )
        get_rag_logger().info(
            "[rag.rerank.init] model=%s device=%s elapsed=%.1fms",
            self._model_name, resolved_device,
            (time.perf_counter() - start) * 1000,
        )

    @property
    def model(self) -> str:
        return self._model_name

    def score(self, query: str, texts: list[str]) -> list[float]:
        """对 (query, text) 对逐一打分，返回与 texts 等长的分数列表。"""
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self._model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]
