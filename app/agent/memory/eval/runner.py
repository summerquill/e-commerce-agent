"""压缩模拟 runner：复放多轮对话，触发与生产一致的压缩逻辑并度量事实保留。

summarize_fn 协议：(old_messages: list[dict], prev_summary: str | None) -> str
- 离线：baselines.py 的规则摘要器（不调 API）
- 在线（Step 2）：适配真实 summarize(client, model, ...) 的薄包装

触发与切分逻辑复用生产代码 app/agent/summarizer.split_history，
保证「评测的就是生产逻辑」；触发时机对齐 EcomAgent.chat()——
每个对话回合（assistant 无 tool_calls 的收尾消息）结束后检查阈值。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agent.memory.eval.factcheck import Fact, FactState, check_facts, message_text
from app.agent.summarizer import split_history


@dataclass
class RoundRecord:
    """一次压缩事件的快照与当时的事实保留状态。

    summary / input_messages 为该轮摘要全文与送入摘要器的原始老消息，
    随结果 JSON 落盘，保证评测全程可回溯复现。
    """

    round: int
    n_compressed: int
    n_window: int
    summary_chars: int
    window_chars: int
    facts: dict[str, FactState]
    summary: str = ""
    input_messages: list[dict] = field(default_factory=list)

    def retention(self) -> float:
        if not self.facts:
            return 1.0
        return sum(s.retained for s in self.facts.values()) / len(self.facts)

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "n_compressed": self.n_compressed,
            "n_window": self.n_window,
            "summary_chars": self.summary_chars,
            "window_chars": self.window_chars,
            "facts": {k: v.to_dict() for k, v in self.facts.items()},
            "retention": round(self.retention(), 4),
            "summary": self.summary,
            "input_messages": self.input_messages,
        }


@dataclass
class ScriptResult:
    """单个对话脚本跑完后的完整结果。"""

    script_id: str
    original_chars: int
    final_summary: str | None
    final_window: list[dict]
    rounds: list[RoundRecord] = field(default_factory=list)
    final_facts: dict[str, FactState] = field(default_factory=dict)

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    @property
    def retention(self) -> float:
        """最终状态的事实保留率（摘要 ∪ 窗口）。"""
        if not self.final_facts:
            return 1.0
        return sum(s.retained for s in self.final_facts.values()) / len(self.final_facts)

    @property
    def summary_reliant_retention(self) -> float:
        """摘要依赖型事实的保留率：已被压出窗口、只能靠 summary 挽回的事实。

        这是衡量摘要质量的核心指标——窗口内的事实不考验摘要能力。
        """
        reliant = {fid: s for fid, s in self.final_facts.items() if not s.in_window}
        if not reliant:
            return 1.0
        return sum(s.retained for s in reliant.values()) / len(reliant)

    @property
    def summary_reliant_ids(self) -> list[str]:
        return sorted(fid for fid, s in self.final_facts.items() if not s.in_window)

    @property
    def compression_ratio(self) -> float:
        """压缩比 = 原始全部消息字符数 /（最终 summary + 窗口消息字符数）。"""
        final_chars = len(self.final_summary or "") + sum(
            len(message_text(m)) for m in self.final_window
        )
        if final_chars == 0:
            return 1.0
        return self.original_chars / final_chars

    def to_dict(self) -> dict:
        return {
            "script_id": self.script_id,
            "n_rounds": self.n_rounds,
            "original_chars": self.original_chars,
            "final_summary_chars": len(self.final_summary or ""),
            "final_summary": self.final_summary,
            "final_window_chars": sum(len(message_text(m)) for m in self.final_window),
            "final_window": self.final_window,
            "compression_ratio": round(self.compression_ratio, 4),
            "retention": round(self.retention, 4),
            "summary_reliant_retention": round(self.summary_reliant_retention, 4),
            "summary_reliant_ids": self.summary_reliant_ids,
            "final_facts": {k: v.to_dict() for k, v in self.final_facts.items()},
            "rounds": [r.to_dict() for r in self.rounds],
        }


def run_script(
    script: dict,
    summarize_fn,
    threshold: int,
    keep_recent: int,
) -> ScriptResult:
    """复放单个对话脚本，逐回合触发压缩并记录事实保留状态。

    script 格式见 datasets/compression_eval.json：
    {"id", "messages": [...], "facts": [{"id", "category", "gold"}]}
    """
    facts = [
        Fact(id=f["id"], category=f["category"], gold=f["gold"])
        for f in script["facts"]
    ]
    messages: list[dict] = []
    summary: str | None = None
    rounds: list[RoundRecord] = []
    original_chars = sum(len(message_text(m)) for m in script["messages"])

    for msg in script["messages"]:
        messages.append(msg)
        # 回合边界：assistant 无 tool_calls 且有内容（对齐 chat() 一轮结束的位置）
        if not (
            msg.get("role") == "assistant"
            and not msg.get("tool_calls")
            and msg.get("content")
        ):
            continue
        if len(messages) <= threshold:
            continue
        split_result = split_history(messages, keep_recent)
        if split_result is None:
            continue
        old, recent = split_result
        summary = summarize_fn(old, summary)
        messages = recent
        rounds.append(RoundRecord(
            round=len(rounds) + 1,
            n_compressed=len(old),
            n_window=len(messages),
            summary_chars=len(summary),
            window_chars=sum(len(message_text(m)) for m in messages),
            facts=check_facts(facts, summary, messages),
            summary=summary,
            input_messages=list(old),
        ))

    final_facts = check_facts(facts, summary, messages)
    return ScriptResult(
        script_id=script["id"],
        original_chars=original_chars,
        final_summary=summary,
        final_window=messages,
        rounds=rounds,
        final_facts=final_facts,
    )


def evaluate(
    dataset: dict,
    summarize_fn,
    threshold: int,
    keep_recent: int,
) -> list[ScriptResult]:
    """跑完数据集里的全部脚本。"""
    return [
        run_script(script, summarize_fn, threshold, keep_recent)
        for script in dataset["scripts"]
    ]


def aggregate(results: list[ScriptResult]) -> dict:
    """跨脚本聚合：均值指标 + 衰减曲线（按压缩轮次对齐）。"""
    if not results:
        return {}

    max_rounds = max(r.n_rounds for r in results)
    decay_curve = {}
    for i in range(1, max_rounds + 1):
        rets = [r.rounds[i - 1].retention() for r in results if r.n_rounds >= i]
        decay_curve[f"round_{i}"] = round(sum(rets) / len(rets), 4)

    return {
        "n_scripts": len(results),
        "mean_final_retention": round(
            sum(r.retention for r in results) / len(results), 4),
        "mean_summary_reliant_retention": round(
            sum(r.summary_reliant_retention for r in results) / len(results), 4),
        "mean_compression_ratio": round(
            sum(r.compression_ratio for r in results) / len(results), 4),
        "total_rounds": sum(r.n_rounds for r in results),
        "decay_curve": decay_curve,
    }
