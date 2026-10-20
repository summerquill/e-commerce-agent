"""RAG 评测集生成脚本：从知识库自动生成评测题（待人工过滤）。

用法:
    python scripts/gen_rag_evalset.py --dry-run
    python scripts/gen_rag_evalset.py --output app/evaluation/rag_dataset.json
    python scripts/gen_rag_evalset.py --direct-per-chunk 3 --multihop 20 --negative 10

生成策略（四类问题叠加）：
    direct      —— 单 chunk 直问，每个 chunk 生成 N 条
    paraphrase  —— direct 问题的口语改写，每条改写 N 条
    multihop    —— 跨 2+ chunk 的多跳问题，全库范围内生成 N 条
    negative    —— 知识库盲区/不支持类问题，生成 N 条

每条输出:
    {id, type, query, expected_chunk_ids, grading_notes, difficulty}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from app.agent.rag.chunker import chunk_markdown_dir  # noqa: E402
from app.config.settings import settings  # noqa: E402


# ─────────────────────── LLM Prompts ───────────────────────

DIRECT_Q_PROMPT = """你是电商客服知识库评测专家。基于下面的知识片段，生成 {n} 个用户可能会问的问题。

【知识片段】
{chunk_text}

要求:
1. 问题必须能通过该片段的内容直接回答
2. 问题要自然、贴近真实用户咨询场景
3. 难度分布：{easy_ratio}% 简单（事实直问），{medium_ratio}% 中等（需理解归纳）
4. 每个问题附上 2-3 个评分要点（grading_notes），用于后续评测答案完整性

输出 JSON 数组，每个元素:
{{"query": "问题", "grading_notes": ["要点1", "要点2"], "difficulty": "easy|medium"}}"""

PARAPHRASE_PROMPT = """把下面的问题改写成 {n} 种不同的口语化表达，模拟不同用户的说话习惯。

原始问题: "{query}"

要求:
1. 保持含义不变，但换一种说法
2. 可以更口语化、更模糊（真实用户不会精确表述）
3. 不要加引号或解释
4. 难度跟随原始问题

输出 JSON 数组: ["改写1", "改写2", ...]"""

MULTIHOP_PROMPT = """你是电商客服知识库评测专家。基于整个知识库，生成 {n} 个需要**跨 2 个以上知识片段**才能回答的多跳问题。

【知识库片段列表】
{all_chunks_text}

要求:
1. 问题的答案需要从 2+ 个不同片段综合得出（如"X 和 Y 有什么区别"、"从 A 到 B 的流程是什么"）
2. 必须标注 expected_chunk_ids（数组，包含所有需要的 chunk_id）
3. 附上 grading_notes（评分要点）

输出 JSON 数组，每个元素:
{{"query": "多跳问题", "expected_chunk_ids": ["chunk_id_1", "chunk_id_2"], "grading_notes": ["要点"], "difficulty": "hard"}}"""

NEGATIVE_PROMPT = """你是电商客服知识库评测专家。基于下面的知识库，生成 {n} 个问题，这些问题的答案**不在知识库中**（或知识库明确说"不支持"）。

【知识库片段列表】
{all_chunks_text}

negative 用例分两类:
A) 知识库明确说不支持（如"暂不支持货到付款"）→ expected_chunk_ids 应该命中相关片段，grading_notes 要点要覆盖"不支持"结论
B) 知识库完全没有覆盖（如保修期限、品牌售后电话）→ expected_chunk_ids 为空数组

要求:
1. 两类混合生成，各占约 50%
2. 问题要真实合理，不能是荒谬问题
3. 标注 negative_type: "explicit_no" | "not_covered"

输出 JSON 数组，每个元素:
{{"query": "问题", "expected_chunk_ids": [], "grading_notes": ["要点"], "difficulty": "medium", "negative_type": "explicit_no"}}"""


# ─────────────────────── 主逻辑 ───────────────────────

def _strip_markdown_code_fence(text: str) -> str:
    """剥离 markdown 代码块包裹（```json ... ```）。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉开头的 ```json 或 ```
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        # 去掉结尾的 ```
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _llm_json(client: OpenAI, model: str, system_prompt: str, user_content: str) -> list | dict:
    """调 LLM 并解析 JSON 返回，失败返回空列表。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt + " 重要：只输出纯 JSON，不要 markdown 代码块包裹（不要 ```json ... ```）。"},
                {"role": "user", "content": user_content},
            ],
        )
        raw = resp.choices[0].message.content or "[]"
        raw = _strip_markdown_code_fence(raw)

        # 尝试直接解析
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # 宽容解析：尝试截取首个 {...} 或 [...] 片段
            for open_c, close_c in (("[", "]"), ("{", "}")):
                start = raw.find(open_c)
                end = raw.rfind(close_c)
                if start != -1 and end > start:
                    try:
                        data = json.loads(raw[start:end + 1])
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                print(f"  [WARN] 无法解析为 JSON: {raw[:80]}...")
                return []

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ("questions", "items", "queries", "results", "data", "answers"):
                if k in data and isinstance(data[k], list):
                    return data[k]
            return data
        return {}
    except Exception as e:
        print(f"  [WARN] LLM 调用失败: {e}")
        return []


def _load_chunks(kb_dir: str) -> list:
    """加载知识库 chunks。"""
    chunks = chunk_markdown_dir(Path(kb_dir))
    print(f"[INFO] 加载 {len(chunks)} 个 chunks from {kb_dir}")
    return chunks


def _format_chunk_for_prompt(chunk) -> str:
    """把 chunk 格式化成 LLM 可读文本。"""
    return f"[{chunk.chunk_id}] {chunk.text}"


def _build_all_chunks_text(chunks: list) -> str:
    """全量 chunks 列表文本（用于 multihop / negative prompt）。"""
    lines = []
    for c in chunks:
        lines.append(f"[{c.chunk_id}] 来源:{c.doc} 章节:{c.section}")
        lines.append(c.text[:200] + "..." if len(c.text) > 200 else c.text)
        lines.append("")
    return "\n".join(lines)


def generate_direct(client: OpenAI, model: str, chunks: list, n_per_chunk: int) -> list[dict]:
    """对每个 chunk 生成 n_per_chunk 个 direct 问题。"""
    results = []
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] 生成 direct 问题: {chunk.chunk_id}")
        prompt = DIRECT_Q_PROMPT.format(
            n=n_per_chunk,
            chunk_text=_format_chunk_for_prompt(chunk),
            easy_ratio=60,
            medium_ratio=40,
        )
        items = _llm_json(client, model, "你是评测数据生成专家", prompt)
        if not isinstance(items, list):
            continue
        for item in items[:n_per_chunk]:
            q = item.get("query", "").strip()
            if not q:
                continue
            notes = item.get("grading_notes", [])
            if isinstance(notes, str):
                notes = [notes]
            results.append({
                "type": "direct",
                "query": q,
                "expected_chunk_ids": [chunk.chunk_id],
                "grading_notes": notes,
                "difficulty": item.get("difficulty", "easy"),
            })
    return results


def generate_paraphrase(client: OpenAI, model: str, direct_items: list[dict], n_per: int) -> list[dict]:
    """对每个 direct 问题生成 n_per 个口语改写。"""
    results = []
    for i, item in enumerate(direct_items):
        print(f"  [{i+1}/{len(direct_items)}] 生成 paraphrase: {item['query'][:20]}...")
        prompt = PARAPHRASE_PROMPT.format(n=n_per, query=item["query"])
        items = _llm_json(client, model, "你是语言改写专家", prompt)
        if not isinstance(items, list):
            continue
        for rewrite in items[:n_per]:
            if isinstance(rewrite, str):
                results.append({
                    "type": "paraphrase",
                    "query": rewrite.strip(),
                    "expected_chunk_ids": item["expected_chunk_ids"],
                    "grading_notes": item["grading_notes"],
                    "difficulty": item["difficulty"],
                    "original_query": item["query"],
                })
    return results


def generate_multihop(client: OpenAI, model: str, chunks: list, n: int) -> list[dict]:
    """生成跨 chunk 的多跳问题。"""
    print(f"  生成 multihop 问题 (目标 {n} 条)")
    prompt = MULTIHOP_PROMPT.format(
        n=n,
        all_chunks_text=_build_all_chunks_text(chunks),
    )
    items = _llm_json(client, model, "你是评测数据生成专家", prompt)
    results = []
    if isinstance(items, list):
        for item in items[:n]:
            if not isinstance(item, dict):
                continue
            q = item.get("query", "").strip()
            if not q:
                continue
            expected = item.get("expected_chunk_ids", [])
            if isinstance(expected, str):
                expected = [expected]
            notes = item.get("grading_notes", [])
            if isinstance(notes, str):
                notes = [notes]
            if len(expected) < 2:
                # 跨 chunk 至少要 2 个
                continue
            results.append({
                "type": "multihop",
                "query": q,
                "expected_chunk_ids": expected,
                "grading_notes": notes,
                "difficulty": "hard",
            })
    return results


def generate_negative(client: OpenAI, model: str, chunks: list, n: int) -> list[dict]:
    """生成 negative 问题（知识库盲区）。"""
    print(f"  生成 negative 问题 (目标 {n} 条)")
    prompt = NEGATIVE_PROMPT.format(
        n=n,
        all_chunks_text=_build_all_chunks_text(chunks),
    )
    items = _llm_json(client, model, "你是评测数据生成专家", prompt)
    results = []
    if isinstance(items, list):
        for item in items[:n]:
            if not isinstance(item, dict):
                continue
            q = item.get("query", "").strip()
            if not q:
                continue
            expected = item.get("expected_chunk_ids", [])
            if isinstance(expected, str):
                expected = [expected]
            notes = item.get("grading_notes", [])
            if isinstance(notes, str):
                notes = [notes]
            results.append({
                "type": "negative",
                "query": q,
                "expected_chunk_ids": expected,
                "grading_notes": notes,
                "difficulty": item.get("difficulty", "medium"),
                "negative_type": item.get("negative_type", "not_covered"),
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate RAG evaluation dataset")
    parser.add_argument("--kb-dir", default=settings.kb_dir)
    parser.add_argument("--model", default=settings.model_name)
    parser.add_argument("--direct-per-chunk", type=int, default=2)
    parser.add_argument("--paraphrase-per-direct", type=int, default=1)
    parser.add_argument("--multihop", type=int, default=25)
    parser.add_argument("--negative", type=int, default=15)
    parser.add_argument("--output", default="app/evaluation/rag_dataset.json")
    parser.add_argument("--dry-run", action="store_true", help="仅预览数量，不调 LLM")
    args = parser.parse_args()

    chunks = _load_chunks(args.kb_dir)

    if args.dry_run:
        n_direct = len(chunks) * args.direct_per_chunk
        n_paraphrase = n_direct * args.paraphrase_per_direct
        print(f"\n[DRY-RUN] 预估产出:")
        print(f"  direct:     {len(chunks)} chunks × {args.direct_per_chunk} = ~{n_direct}")
        print(f"  paraphrase: {n_direct} direct × {args.paraphrase_per_direct} = ~{n_paraphrase}")
        print(f"  multihop:   ~{args.multihop}")
        print(f"  negative:   ~{args.negative}")
        print(f"  ─────────────────────")
        print(f"  合计:       ~{n_direct + n_paraphrase + args.multihop + args.negative}")
        return

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    all_items = []

    print("\n[1/4] 生成 direct 问题...")
    direct = generate_direct(client, args.model, chunks, args.direct_per_chunk)
    all_items.extend(direct)
    print(f"  ✓ {len(direct)} 条 direct")

    print("\n[2/4] 生成 paraphrase 改写...")
    paraphrases = generate_paraphrase(client, args.model, direct, args.paraphrase_per_direct)
    all_items.extend(paraphrases)
    print(f"  ✓ {len(paraphrases)} 条 paraphrase")

    print("\n[3/4] 生成 multihop 问题...")
    multihop = generate_multihop(client, args.model, chunks, args.multihop)
    all_items.extend(multihop)
    print(f"  ✓ {len(multihop)} 条 multihop")

    print("\n[4/4] 生成 negative 问题...")
    negative = generate_negative(client, args.model, chunks, args.negative)
    all_items.extend(negative)
    print(f"  ✓ {len(negative)} 条 negative")

    # 去重 + 编号
    seen = set()
    deduped = []
    for item in all_items:
        key = item["query"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    # 编号
    for i, item in enumerate(deduped):
        item["id"] = f"q_{i+1:03d}"

    # 输出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"cases": deduped}, f, ensure_ascii=False, indent=2)

    # 统计
    from collections import Counter
    type_counts = Counter(item["type"] for item in deduped)
    diff_counts = Counter(item.get("difficulty", "?") for item in deduped)
    print(f"\n{'='*50}")
    print(f"[DONE] 总计 {len(deduped)} 条 → {output_path}")
    print(f"  类型分布: {dict(type_counts)}")
    print(f"  难度分布: {dict(diff_counts)}")
    print(f"\n⚠️  请人工过滤后再使用（删除重复/垃圾题、核对 expected_chunk_ids）")


if __name__ == "__main__":
    main()
