"""
Project adapter: connect the evaluator to your actual RAG implementation.

Keep project-specific imports and return-value conversion in this file so
evals/run_eval.py does not depend on the internal architecture of the app.

Required contract:
  retrieve(query, top_k) -> list of Document-like objects or dicts
  generate(query, contexts) -> str

Each retrieved item must expose a stable chunk ID, ideally in metadata["chunk_id"].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 允许从任意工作目录导入项目包（与 scripts/ 下脚本的引导方式一致）
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI  # noqa: E402

from app.agent.tools.knowledge import _get_retriever  # noqa: E402
from app.config.settings import settings  # noqa: E402
from scripts.eval_rag import _generate_answer  # noqa: E402

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """复用 settings 的 OpenAI 配置，进程内只创建一次。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _client


def retrieve(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """复用应用真实的检索链路：settings.rag_backend → KnowledgeRetriever → backend.search。

    返回顺序即后端给出的相关性降序；chunk_id 由 chunker 生成，重建索引后保持稳定。
    """
    retriever = _get_retriever()
    hits = retriever.search(query, top_k=top_k)
    return [
        {
            "chunk_id": hit.chunk.chunk_id,
            "text": hit.chunk.text,
            "metadata": {
                "chunk_id": hit.chunk.chunk_id,
                "doc": hit.chunk.doc,
                "section": hit.chunk.section,
            },
            "score": hit.score,
        }
        for hit in hits
    ]


def generate(query: str, contexts: list[str]) -> str:
    """复用 scripts/eval_rag.py 的生成逻辑（同一套 GENERATION_PROMPT 与模型配置）。

    contexts 只携带文本，没有 chunk_id，这里用 ctx_N 作占位标识；
    生成阶段只需要文本内容，不影响答案。
    """
    context_dicts = [
        {"chunk_id": f"ctx_{i + 1}", "text": text}
        for i, text in enumerate(contexts)
    ]
    return _generate_answer(_get_client(), settings.model_name, query, context_dicts)
