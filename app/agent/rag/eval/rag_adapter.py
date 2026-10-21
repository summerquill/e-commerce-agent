"""
Project adapter: connect the evaluator to your actual RAG implementation.

Keep project-specific imports and return-value conversion in this file so
evals/run_eval.py does not depend on the internal architecture of the app.

Required contract:
  retrieve(query, top_k) -> list of Document-like objects or dicts
  generate(query, contexts) -> str

Each retrieved item must expose a stable chunk ID, ideally in metadata["chunk_id"].
"""

from typing import Any


def retrieve(query: str, top_k: int = 5) -> list[Any]:
    # TODO 1: Import and initialize your existing retriever/pipeline here.
    # Example shape only (replace with the actual class/method in your repo):
    #
    # from app.agent.rag.retriever import Retriever
    # retriever = Retriever(...)
    # docs = retriever.retrieve(query, top_k=top_k)
    # return docs
    #
    # IMPORTANT: Do not create a second retrieval implementation for evaluation.
    # Call the same retrieval code used by the application.
    raise NotImplementedError(
        "Implement retrieve() using your existing RAG retriever. "
        "Return items with stable chunk IDs in metadata['chunk_id']."
    )


def generate(query: str, contexts: list[str]) -> str:
    # TODO 2: Call your existing answer-generation logic with this query and contexts.
    # If your current pipeline only exposes end-to-end ask(query), use that initially,
    # but then retrieved contexts should be captured from the pipeline trace instead
    # of running retrieval twice.
    #
    # Example shape only:
    # return answer_generator.generate(query=query, contexts=contexts)
    raise NotImplementedError(
        "Implement generate() using your existing generation logic, "
        "or temporarily return a fixed string if testing retrieval metrics only."
    )
