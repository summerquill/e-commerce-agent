"""压缩评估 CLI（离线，不调 API）。

用法（仓库根目录，.venv 环境）：
    python -m app.agent.memory.eval.run_eval --summarizer extractive
    python -m app.agent.memory.eval.run_eval --summarizer truncate --threshold 4 --keep-recent 2

默认 threshold=4 / keep_recent=2：离线脚本体量下能触发多轮压缩，
展示衰减曲线；Step 2 接入真实 LLM 摘要器时应使用生产配置
（threshold=10 / keep_recent=3）并加长对话脚本。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agent.memory.eval.baselines import SUMMARIZERS
from app.agent.memory.eval.runner import aggregate, evaluate

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="上下文压缩评估（离线 harness）")
    parser.add_argument(
        "--dataset",
        default=str(BASE_DIR / "datasets" / "compression_eval.json"),
        help="评测集路径",
    )
    parser.add_argument(
        "--summarizer", choices=sorted(SUMMARIZERS) + ["llm"], default="extractive",
        help="摘要器：离线基线（truncate/extractive）或真实 LLM（llm，调 API）",
    )
    parser.add_argument("--threshold", type=int, default=4, help="压缩触发阈值（消息条数）")
    parser.add_argument("--keep-recent", type=int, default=2, help="压缩时保留最近 N 条")
    parser.add_argument("--max-chars", type=int, default=300, help="摘要长度上限（字符）")
    parser.add_argument(
        "--pin-entities", action="store_true",
        help="启用 pinned facts 锚点（对齐生产机制，事实检查对锚点+摘要合并上下文）",
    )
    parser.add_argument(
        "--hard-cap", type=int, default=0,
        help="LLM 摘要硬截断字数（0=不截断；生产为 settings.summary_hard_cap）",
    )
    parser.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.summarizer == "llm":
        # 懒加载：离线基线模式不引入真实 client
        # --max-chars 对 llm 摘要器是 prompt 级预算（默认 300，压测可调小）
        from app.agent.memory.eval.llm_summarizer import build_llm_summarizer
        summarize_fn = build_llm_summarizer(
            hard_cap=args.hard_cap or None, budget_chars=args.max_chars,
        )
    else:
        summarize_fn = SUMMARIZERS[args.summarizer](max_chars=args.max_chars)
    results = evaluate(
        dataset, summarize_fn, args.threshold, args.keep_recent,
        pin_entities=args.pin_entities,
    )
    agg = aggregate(results)

    print(f"\n== 压缩评估 | summarizer={args.summarizer} "
          f"threshold={args.threshold} keep_recent={args.keep_recent} "
          f"pin={args.pin_entities} hard_cap={args.hard_cap or 'off'} ==")
    print(f"{'script':<16} {'rounds':>6} {'ratio':>7} {'retention':>10} {'summary-reliant':>16}")
    for r in results:
        print(f"{r.script_id:<16} {r.n_rounds:>6} "
              f"{r.compression_ratio:>6.2f}x {r.retention:>10.2%} "
              f"{r.summary_reliant_retention:>16.2%}")
    print("\n== 汇总 ==")
    print(f"final_retention_mean          = {agg['mean_final_retention']}")
    print(f"summary_reliant_retention_mean= {agg['mean_summary_reliant_retention']}")
    print(f"compression_ratio_mean        = {agg['mean_compression_ratio']}")
    print(f"decay_curve                   = {agg['decay_curve']}")
    if agg.get("retention_by_category"):
        by_cat = {c: v["retention"] for c, v in agg["retention_by_category"].items()}
        print(f"retention_by_category         = "
              f"{json.dumps(by_cat, ensure_ascii=False)}")

    out_path = (
        Path(args.out) if args.out
        else BASE_DIR / "results" / (
            f"compression-{args.summarizer}-t{args.threshold}-k{args.keep_recent}"
            f"{'-pin' if args.pin_entities else ''}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": vars(args),
        "aggregate": agg,
        "scripts": [r.to_dict() for r in results],
    }
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n✅ 结果已写入 {out_path}")


if __name__ == "__main__":
    main()
