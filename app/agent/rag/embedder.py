"""向量化封装：支持远程 OpenAI Embeddings 和本地 sentence-transformers 两套后端。

两套实现共用 encode / encode_one 接口，由 create_embedder() 工厂
按 settings.embedding_backend 路由：
- remote：OpenAI Embeddings（text-embedding-3-small 等），走 aihubmix 代理
- local ：sentence-transformers 本地推理（BAAI/bge-small-zh-v1.5）

向量维度：
- remote text-embedding-3-small: 1536
- local  bge-small-zh-v1.5       : 512

索引持久化时会把 embedding_model 一起写入，加载时由 Retriever 校验
一致性，防止「换了 embedding 还在用旧索引」的静默脏检索。
"""

from __future__ import annotations

from typing import Iterable, Protocol

from app.config.settings import settings


class EmbedderProtocol(Protocol):
    """统一向量化接口。"""

    @property
    def model(self) -> str: ...

    def encode(self, texts: Iterable[str]) -> list[list[float]]: ...

    def encode_one(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Remote: OpenAI Embeddings
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """OpenAI Embeddings 同步封装。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
    ):
        # 延迟 import：没装 openai 也不影响只用本地模型
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._batch_size = batch_size

    @property
    def model(self) -> str:
        return self._model

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(item.embedding for item in resp.data)
        return out

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# Local: sentence-transformers
# ---------------------------------------------------------------------------


class LocalBgeEmbedder:
    """本地 sentence-transformers 封装（BAAI/bge-small-zh-v1.5 为默认）。

    进程内单例缓存模型，避免重复加载（首次 encode 触发下载，之后完全离线）。
    macOS M 系列自动走 MPS 加速，CPU 环境自动 fallback。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        if device is None:
            try:
                import platform

                import torch
                # mps 仅 Apple Silicon 可用；torch 在 Intel Mac 上会误报 mps available
                if (
                    platform.machine() == "arm64"
                    and torch.backends.mps.is_available()
                ):
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name

    @property
    def model(self) -> str:
        return self._model_name

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        # normalize_embeddings=True 让输出单位化，和余弦相似度假设一致
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def create_embedder(
    backend: str | None = None,
    **overrides,
) -> EmbedderProtocol:
    """按 settings.embedding_backend 创建 embedder 实例。

    backend: "remote" | "local"，为 None 时读 settings.embedding_backend
    overrides: 覆盖特定后端的构造参数（如 model_name / api_key）
    """
    backend = (backend or settings.embedding_backend).lower()
    if backend == "remote":
        return OpenAIEmbedder(
            api_key=overrides.get("api_key", settings.openai_api_key),
            base_url=overrides.get("base_url", settings.openai_base_url),
            model=overrides.get("model", settings.embedding_model),
            batch_size=overrides.get("batch_size", 64),
        )
    if backend == "local":
        return LocalBgeEmbedder(
            model_name=overrides.get("model_name", settings.local_embedding_model),
            device=overrides.get("device", None),
        )
    raise ValueError(
        f"未知的 embedding_backend: {backend}（可选: remote / local）"
    )


# 向后兼容：老代码里 from ... import Embedder 仍能跑（默认 remote）
def Embedder(*args, **kwargs):  # noqa: N802
    """旧接口兼容：等价于 create_embedder("remote", ...)。"""
    return OpenAIEmbedder(*args, **kwargs)
