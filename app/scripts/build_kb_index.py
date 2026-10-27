"""离线构建知识库向量索引。

用法：
  # 默认走 settings.rag_backend（dense + BM25 一起构建）
  python app/scripts/build_kb_index.py

  # 显式指定后端，便于一份知识库同时构建两种后端的索引做对比
  python app/scripts/build_kb_index.py --backend numpy
  python app/scripts/build_kb_index.py --backend chroma

  # 仅构建 BM25 词汇索引：chunks 取自现有 dense 索引（不再调用 embedding API），
  # 保证两路索引的 chunk_id 完全对齐
  python app/scripts/build_kb_index.py --only-bm25

流程：
  1. 扫描 knowledge/ 下的所有 .md 文件，按二级标题切分。
  2. 调用 OpenAI Embeddings 将每个 chunk 向量化。
  3. 通过 backend.upsert 持久化索引（NumpyBackend 写 JSON / ChromaBackend 写 PersistentClient）。
  4. 用同一次切分的同一批 chunks 构建 BM25 词汇索引（chunk_id 与 dense 完全一致）。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.agent.rag.backends import create_backend  # noqa: E402
from app.agent.rag.chunker import Chunk, chunk_markdown_dir  # noqa: E402
from app.agent.rag.embedder import create_embedder  # noqa: E402
from app.agent.rag.lexical import LexicalIndex  # noqa: E402


def _build_backend(name: str):
    if name == "numpy":
        return create_backend(
            "numpy",
            index_path=ROOT / settings.kb_index_path,
        ), str(ROOT / settings.kb_index_path)
    if name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=ROOT / settings.chroma_persist_dir,
            collection_name=settings.chroma_collection,
        ), f"{ROOT / settings.chroma_persist_dir} (collection={settings.chroma_collection})"
    raise ValueError(f"未知后端: {name}")


def _build_bm25(chunks: list[Chunk], embedding_model: str) -> None:
    """用给定 chunks 构建 BM25 词汇索引（与 dense 索引共享同一批 chunk_id）。"""
    target = ROOT / settings.kb_bm25_path
    lexical = LexicalIndex(target)
    lexical.build(chunks, embedding_model=embedding_model)
    lexical.save()
    print(f"   BM25 索引已写入: {target}（{lexical.size} 条）")


def _only_bm25() -> None:
    """仅构建 BM25：chunks 从现有 dense 索引提取，不调用 embedding API。"""
    index_path = ROOT / settings.kb_index_path
    if not index_path.exists():
        print(f"❌ 未找到 dense 索引: {index_path}\n   请先构建 dense 索引（去掉 --only-bm25）。")
        sys.exit(1)

    data = json.loads(index_path.read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in data["chunks"]]
    print("=" * 60)
    print("  并夕夕 · 仅构建 BM25 词汇索引")
    print(f"  chunk 来源: {index_path}（当前生效的 dense 索引，chunk_id 天然对齐）")
    print(f"  索引目标  : {ROOT / settings.kb_bm25_path}")
    print(f"  Embedding : 无需（BM25 不依赖向量，来源记录 = {data.get('embedding_model', '')}）")
    print("=" * 60)

    print(f"\n[1/1] jieba 分词并构建 BM25...")
    _build_bm25(chunks, data.get("embedding_model", ""))
    print("\n🎉 BM25 索引构建完成。")


def main():
    parser = argparse.ArgumentParser(description="构建知识库向量索引")
    parser.add_argument(
        "--backend",
        choices=["numpy", "chroma"],
        default=settings.rag_backend,
        help=f"向量后端（默认: {settings.rag_backend}）",
    )
    parser.add_argument(
        "--only-bm25",
        action="store_true",
        help="仅构建 BM25 词汇索引（chunks 取自现有 dense 索引，不调用 embedding API）",
    )
    args = parser.parse_args()

    if args.only_bm25:
        _only_bm25()
        return

    kb_dir = ROOT / settings.kb_dir

    if not kb_dir.exists():
        print(f"❌ 知识库目录不存在: {kb_dir}")
        sys.exit(1)

    backend, target_desc = _build_backend(args.backend)

    print("=" * 60)
    print("  并夕夕 · 知识库索引构建")
    print(f"  后端      : {args.backend}")
    print(f"  源目录    : {kb_dir}")
    print(f"  索引目标  : {target_desc}")
    _emb_name = (
        settings.embedding_model
        if settings.embedding_backend == "remote"
        else settings.local_embedding_model
    )
    print(f"  Embedding : {settings.embedding_backend} / {_emb_name}")
    print("=" * 60)

    print("\n[1/4] 扫描并切分 markdown 文档...")
    chunks = chunk_markdown_dir(kb_dir)
    if not chunks:
        print("❌ 未发现任何文档，请检查 knowledge/ 目录")
        sys.exit(1)

    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc] = by_doc.get(c.doc, 0) + 1
    for doc, n in by_doc.items():
        print(f"   - {doc}: {n} chunk")
    print(f"   合计 {len(chunks)} 个 chunk")

    print(f"\n[2/4] 调用向量化...")
    embedder = create_embedder()
    print(f"   backend={settings.embedding_backend}  model={embedder.model}")
    vectors = embedder.encode([c.text for c in chunks])
    dim = len(vectors[0]) if vectors else 0
    print(f"   完成，向量维度 = {dim}")

    print(f"\n[3/4] 写入 {args.backend} 索引...")
    backend.upsert(
        chunks=chunks,
        vectors=vectors,
        embedding_model=embedder.model,
    )
    print(f"   已写入 {backend.size()} 条向量")

    print(f"\n[4/4] 构建 BM25 词汇索引（复用 [1/4] 的同一批 chunks，chunk_id 对齐）...")
    _build_bm25(chunks, embedder.model)

    print("\n🎉 索引构建完成。")


if __name__ == "__main__":
    main()
