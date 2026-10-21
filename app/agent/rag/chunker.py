"""按 Markdown 标题层级切分知识库文档。

切分策略：
- 以 `## ` 二级标题为一级切分边界，每个二级章节作为一个 chunk 分组。
- 二级章节内若存在 `### ` 三级标题，则按三级标题继续切分，
  避免“一个 chunk 装多个子主题”导致向量语义被稀释（如会员权益四个等级、
  FAQ 多个问题、特殊场景多个分支混在同一块里）。
- 三级标题之前的章节导语（若有）单独成块，与三级小节并列编号。
- 文档开头到第一个二级标题之间的内容（含一级标题和导语）作为 "概览" chunk。
- 单个 chunk 超过 ~1200 字时按段落进一步切分，避免 chunk 过长。

chunk_id 规则（与旧版保持兼容）：
- 常规：`{doc}#{h2_idx:02d}-{h3_idx:02d}`
  - 只有二级标题时 h3_idx 固定为 00
  - 有三级标题时按 00、01... 编号
- 超长段落再切分时追加一段：`{doc}#{h2_idx:02d}-{h3_idx:02d}-{part:02d}`

每个 chunk 保留：
- doc：文档名（不含扩展名），如 "退换货政策"
- section：章节标题（有三级标题时为 "二级 · 三级"），如 "四、特殊场景 · 4.2 拒收退款"
- text：chunk 全文（含标题前缀）
- chunk_id：稳定的字符串 id，便于增量更新与评测对齐
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHUNK_CHARS = 1200


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_markdown_dir(kb_dir: Path) -> list[Chunk]:
    """扫描目录下所有 .md 文件，逐一切分并汇总。"""
    chunks: list[Chunk] = []
    for md_path in sorted(kb_dir.glob("*.md")):
        chunks.extend(_chunk_one_file(md_path))
    return chunks


def _chunk_one_file(md_path: Path) -> list[Chunk]:
    raw = md_path.read_text(encoding="utf-8")
    doc_name = md_path.stem

    out: list[Chunk] = []
    for h2_idx, h3_idx, section, body in _split_by_headings(raw):
        text = body.strip()
        if not text:
            continue

        if len(text) <= MAX_CHUNK_CHARS:
            out.append(_make_chunk(doc_name, section, text, h2_idx, h3_idx))
            continue

        for part_idx, piece in enumerate(_split_long(text)):
            out.append(
                _make_chunk(doc_name, section, piece, h2_idx, h3_idx, part_idx)
            )
    return out


def _split_by_headings(raw: str) -> list[tuple[int, int, str, str]]:
    """按 H2/H3 切分，返回 [(h2_idx, h3_idx, section, body), ...]。

    h2_idx 从 0 开始；若文档含 H1，则 0 号预留给“概览”块，
    以保持既有 chunk_id（如 `会员权益#01-00`）稳定。
    """
    lines = raw.splitlines()
    blocks: list[tuple[int, int, str, str]] = []

    h1_title = ""
    h2_title = ""
    h3_title = ""
    h2_idx = -1
    h3_idx = 0
    buf: list[str] = []

    def flush() -> None:
        nonlocal h3_idx
        text = "\n".join(buf).strip()
        if not text:
            return

        if not h2_title:
            section = f"{h1_title} · 概览" if h1_title else "概览"
            blocks.append((0, 0, section, text))
            return

        section = f"{h2_title} · {h3_title}" if h3_title else h2_title
        blocks.append((h2_idx, h3_idx, section, text))
        h3_idx += 1

    for line in lines:
        if line.startswith("# "):
            h1_title = line[2:].strip()
            if h2_idx < 0:
                h2_idx = 0  # 给概览块预留 0 号
            continue
        if line.startswith("## "):
            flush()
            buf.clear()
            h2_title = line[3:].strip()
            h3_title = ""
            h3_idx = 0
            h2_idx = h2_idx + 1 if h2_idx >= 0 else 0
            continue
        if line.startswith("### "):
            flush()
            buf.clear()
            h3_title = line[4:].strip()
            continue
        buf.append(line)

    flush()
    return blocks


def _split_long(text: str) -> list[str]:
    """按段落贪心打包到 MAX_CHUNK_CHARS。"""
    paragraphs = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if buf and buf_len + len(p) + 2 > MAX_CHUNK_CHARS:
            pieces.append("\n\n".join(buf))
            buf = [p]
            buf_len = len(p)
        else:
            buf.append(p)
            buf_len += len(p) + 2
    if buf:
        pieces.append("\n\n".join(buf))
    return pieces


def _make_chunk(
    doc: str,
    section: str,
    text: str,
    h2_idx: int,
    h3_idx: int,
    part_idx: int | None = None,
) -> Chunk:
    chunk_id = f"{doc}#{h2_idx:02d}-{h3_idx:02d}"
    if part_idx is not None:
        chunk_id += f"-{part_idx:02d}"
    body = f"【{doc} · {section}】\n{text}"
    return Chunk(chunk_id=chunk_id, doc=doc, section=section, text=body)
