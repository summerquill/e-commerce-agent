"""上下文压缩评估 harness 的离线单测（不需要 API key，秒级）。

覆盖：
- split_history 切分边界（tool 消息不孤立 / 窗口过大不压缩）
- build_transcript 渲染契约（tool_calls 参数保留、tool 结果截断）
- 事实匹配（含 tool_calls 参数中的实体）
- runner 多轮触发与窗口维护
- 双基线摘要器的区分度（harness 自检：extractive 必须优于 truncate）
- EcomAgent._compress_history 与 runner 的等价性（评测的就是生产逻辑）

跑法（在项目 .venv 中，仓库根目录执行）：
    python -m unittest tests.test_compression_eval -v
"""

import contextlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.memory.eval.baselines import (  # noqa: E402
    extractive_summarizer,
    truncate_summarizer,
)
from app.agent.memory.eval.factcheck import Fact, check_facts  # noqa: E402
from app.agent.memory.eval.runner import aggregate, run_script  # noqa: E402
from app.agent.summarizer import build_transcript, split_history  # noqa: E402

DATASET_PATH = (
    ROOT / "app" / "agent" / "memory" / "eval" / "datasets" / "compression_eval.json"
)


def _load_dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _script_by_id(dataset: dict, script_id: str) -> dict:
    return next(s for s in dataset["scripts"] if s["id"] == script_id)


def _constant_summarizer(text: str = "固定摘要"):
    return lambda old_messages, prev_summary: text


# ──────────────────── split_history ────────────────────

class TestSplitHistory(unittest.TestCase):

    def test_split_avoids_tool_orphan(self):
        """切分点落在 tool 消息上时必须回退：工具结果与其 tool_calls 同侧。"""
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "assistant", "content": "a1"},
        ]
        old, recent = split_history(messages, keep_recent=2)
        # 直接切分点是 index 2（tool）→ 回退到 1，tool 结果与 tool_calls 同入 recent
        self.assertEqual(old, [messages[0]])
        self.assertEqual(recent[0]["role"], "assistant")
        self.assertIn({"role": "tool", "tool_call_id": "c1", "content": "r1"}, recent)
        self.assertEqual(old + recent, messages)

    def test_none_when_window_too_big(self):
        messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
        self.assertIsNone(split_history(messages, keep_recent=5))
        self.assertIsNone(split_history([], keep_recent=2))


# ──────────────────── build_transcript ────────────────────

class TestBuildTranscript(unittest.TestCase):

    def test_tool_calls_and_truncation(self):
        long_result = "x" * 500
        messages = [
            {"role": "user", "content": "查订单 SO202511020015"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "query_order",
                                          "arguments": "{\"order_id\": \"SO202511020015\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": long_result},
            {"role": "assistant", "content": "已查到"},
        ]
        transcript = build_transcript(messages)
        self.assertIn("用户：查订单 SO202511020015", transcript)
        # tool_calls 的参数必须保留（订单号常在参数里）
        self.assertIn("[调用工具 query_order({\"order_id\": \"SO202511020015\"})]", transcript)
        # tool 结果截断到 200 字符 + 省略号
        self.assertIn("[工具结果] " + "x" * 200 + "...", transcript)
        self.assertIn("客服：已查到", transcript)


# ──────────────────── 事实匹配 ────────────────────

class TestCheckFacts(unittest.TestCase):

    def test_fact_in_tool_args_counts(self):
        facts = [Fact(id="order", category="order_id", gold="SO202511020015")]
        window = [
            {"role": "user", "content": "帮我查下订单"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "query_order",
                                          "arguments": "{\"order_id\": \"SO202511020015\"}"}}]},
        ]
        states = check_facts(facts, summary=None, window_messages=window)
        self.assertTrue(states["order"].in_window)
        self.assertFalse(states["order"].in_summary)
        self.assertTrue(states["order"].retained)

    def test_matching_ignores_whitespace(self):
        """回归用例（测试发现）：LLM 摘要把「100 元无门槛」重写成
        「100元无门槛」，空白敏感匹配会漏报已保留的事实。"""
        facts = [Fact(id="c", category="promise", gold="100 元无门槛")]
        states = check_facts(
            facts, summary="客服已申请并审核通过100元无门槛券", window_messages=[])
        self.assertTrue(states["c"].in_summary)
        states2 = check_facts(
            facts, summary=None,
            window_messages=[{"role": "user", "content": "给了张 100 元\n无门槛 的券"}])
        self.assertTrue(states2["c"].in_window)
        # 语义不同的事实仍不可误报
        states3 = check_facts(facts, summary="发放了 50 元优惠券", window_messages=[])
        self.assertFalse(states3["c"].in_summary)

    def test_summary_or_window(self):
        facts = [Fact(id="a", category="other", gold="钻石会员"),
                 Fact(id="b", category="other", gold="不上架")]
        states = check_facts(facts, summary="用户是钻石会员", window_messages=[
            {"role": "user", "content": "这个商品不上架了"},
        ])
        self.assertTrue(states["a"].in_summary and not states["a"].in_window)
        self.assertTrue(states["b"].in_window and not states["b"].in_summary)
        self.assertTrue(states["a"].retained and states["b"].retained)


# ──────────────────── runner ────────────────────

class TestRunner(unittest.TestCase):

    def test_multi_round_trigger_and_window(self):
        dataset = _load_dataset()
        script = _script_by_id(dataset, "ret_flow_001")
        result = run_script(script, _constant_summarizer(), threshold=4, keep_recent=2)

        self.assertEqual(result.n_rounds, 3)
        # 每轮压缩 4 条、窗口回到 2 条
        for r in result.rounds:
            self.assertEqual(r.n_compressed, 4)
            self.assertEqual(r.n_window, 2)
        # 最终窗口 = 脚本最后两条消息
        self.assertEqual(result.final_window, script["messages"][-2:])
        # 固定摘要不含事实：早期事实只能靠窗口 → 全部丢失；late 事实留在窗口
        self.assertFalse(result.final_facts["order_id"].retained)
        self.assertTrue(result.final_facts["notice_channel"].in_window)
        # 衰减曲线逐轮记录
        self.assertEqual([r.round for r in result.rounds], [1, 2, 3])

    def test_no_compression_below_threshold(self):
        script = {"id": "tiny", "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "您好"},
        ], "facts": [{"id": "greet", "category": "other", "gold": "您好"}]}
        result = run_script(script, _constant_summarizer("x"), threshold=10, keep_recent=3)
        self.assertEqual(result.n_rounds, 0)
        self.assertEqual(result.retention, 1.0)
        self.assertEqual(result.compression_ratio, 1.0)

    def test_aggregate_decay_curve(self):
        dataset = _load_dataset()
        results = [
            run_script(s, truncate_summarizer(), threshold=4, keep_recent=2)
            for s in dataset["scripts"]
        ]
        agg = aggregate(results)
        self.assertEqual(agg["n_scripts"], 4)
        self.assertTrue(agg["total_rounds"] > 0)
        # 衰减曲线键与实际轮次对齐
        self.assertEqual(
            sorted(agg["decay_curve"].keys()),
            sorted(f"round_{i}" for i in range(1, max(r.n_rounds for r in results) + 1)),
        )


# ──────────────────── 双基线区分度（harness 自检）────────────────────

class TestBaselineDiscrimination(unittest.TestCase):

    def test_extractive_beats_truncate_synthetic(self):
        """合成脚本：承诺事实深埋在 transcript 截断窗之外，
        truncate 必丢，extractive 靠关键词抽行保留。"""
        pad_a = "你好，我在你们平台买了挺多东西的，" + "这次想处理一下售后问题，" * 12
        pad_b = "之前买的商品整体还不错，" + "不过这次遇到了一些状况需要反馈，" * 10
        script = {
            "id": "synthetic",
            "messages": [
                {"role": "user", "content": f"{pad_a}订单号 SO202511010001，帮我看看。"},
                {"role": "assistant", "content": f"{pad_b}已收到您的反馈，稍等为您核实。"},
                {"role": "user", "content": "好的，麻烦尽快。"},
                {"role": "assistant", "content": "与您确认好了，工程师已排期，明天上午上门安装。"},
                {"role": "user", "content": "那大概几点？"},
                {"role": "assistant", "content": "上午 10 点到 12 点之间，请保持电话畅通。"},
                {"role": "user", "content": "好的谢谢。"},
                {"role": "assistant", "content": "不客气，有其他问题随时联系我们。"},
            ],
            "facts": [
                {"id": "order", "category": "order_id", "gold": "SO202511010001"},
                {"id": "promise", "category": "promise", "gold": "上门安装"},
            ],
        }
        truncate = run_script(script, truncate_summarizer(max_chars=300), threshold=4, keep_recent=2)
        extractive = run_script(script, extractive_summarizer(max_chars=300), threshold=4, keep_recent=2)

        # 两个基线都触发了一轮压缩，两个事实都依赖摘要挽回
        self.assertEqual(truncate.n_rounds, 1)
        self.assertEqual(extractive.n_rounds, 1)
        self.assertEqual(set(truncate.summary_reliant_ids), {"order", "promise"})

        # truncate 丢失深埋承诺，extractive 保留；订单号两者都在 transcript 头部
        self.assertFalse(truncate.final_facts["promise"].retained)
        self.assertTrue(extractive.final_facts["promise"].in_summary)
        self.assertGreater(
            extractive.summary_reliant_retention,
            truncate.summary_reliant_retention,
        )

    def test_extractive_not_worse_than_truncate_on_dataset(self):
        dataset = _load_dataset()
        agg_truncate = aggregate([
            run_script(s, truncate_summarizer(), threshold=4, keep_recent=2)
            for s in dataset["scripts"]
        ])
        agg_extractive = aggregate([
            run_script(s, extractive_summarizer(), threshold=4, keep_recent=2)
            for s in dataset["scripts"]
        ])
        self.assertGreaterEqual(
            agg_extractive["mean_summary_reliant_retention"],
            agg_truncate["mean_summary_reliant_retention"],
        )


# ──────────────────── 与生产 EcomAgent 的等价性 ────────────────────

class _StubSummaryClient:
    """summarize() 所需的最小 client 桩：返回固定摘要文本。"""

    def __init__(self, text: str = "测试摘要文本"):
        self.text = text
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        msg = types.SimpleNamespace(content=self.text, tool_calls=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class TestAgentEquivalence(unittest.TestCase):

    def test_compress_history_matches_split(self):
        """EcomAgent._compress_history 的切分/保留行为与 runner 完全一致。"""
        from app.agent.chat import EcomAgent
        from app.config.settings import settings

        dataset = _load_dataset()
        script = _script_by_id(dataset, "ret_flow_001")
        msgs = script["messages"][:6]  # 超过 threshold=4，触发一轮压缩

        with tempfile.TemporaryDirectory() as td:
            session_path = os.path.join(td, "session.json")
            with contextlib.ExitStack() as stack:
                for attr, value in (
                    ("session_path", session_path),
                    ("memory_enabled", False),
                    ("skills_enabled", False),
                    ("mcp_enabled", False),
                    ("history_threshold", 4),
                    ("history_keep_recent", 2),
                ):
                    stack.enter_context(patch.object(settings, attr, value))
                agent = EcomAgent()

            # 必须显式替换为桩 client：EcomAgent 构造时会用真实 key 建
            # OpenAI client，不替换会导致测试意外打出真实 LLM 调用
            agent.client = _StubSummaryClient()

            agent.raw_messages = [dict(m) for m in msgs]
            agent.summary = None
            agent._compress_history()

            self.assertEqual(agent.summary, "测试摘要文本")
            expected_old, expected_recent = split_history(msgs, keep_recent=2)
            self.assertEqual(agent.raw_messages, expected_recent)
            self.assertEqual(len(expected_old), 4)


# ──────────────────── trace 落盘 ────────────────────

class TestTracePersistence(unittest.TestCase):

    def test_round_trace_landed_in_to_dict(self):
        """每轮摘要全文 + 送入摘要器的原始老消息必须随结果落盘（可回溯复现）。"""
        dataset = _load_dataset()
        script = _script_by_id(dataset, "ret_flow_001")
        result = run_script(script, _constant_summarizer("固定摘要"), threshold=4, keep_recent=2)
        d = result.to_dict()
        self.assertEqual(len(d["rounds"]), result.n_rounds)
        for r in d["rounds"]:
            self.assertEqual(r["summary"], "固定摘要")
            self.assertEqual(len(r["input_messages"]), r["n_compressed"])
        self.assertEqual(d["final_summary"], "固定摘要")
        self.assertEqual(d["final_window"], result.final_window)

    def test_prod_dataset_late_facts_in_window_at_t10k3(self):
        """生产配置（t=10/k=3）下：各长脚本触发 4 轮压缩，
        late 控制组事实必须落在最终保留窗口（harness 自校验）。"""
        prod_path = ROOT / "app" / "agent" / "memory" / "eval" / "datasets" / "compression_eval_prod.json"
        dataset = json.loads(prod_path.read_text(encoding="utf-8"))
        for script in dataset["scripts"]:
            result = run_script(script, _constant_summarizer("固定摘要"), threshold=10, keep_recent=3)
            late = script["facts"][-1]
            self.assertEqual(result.n_rounds, 4, msg=script["id"])
            self.assertTrue(
                result.final_facts[late["id"]].in_window,
                msg=f"{script['id']} late fact '{late['gold']}' not in window",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
