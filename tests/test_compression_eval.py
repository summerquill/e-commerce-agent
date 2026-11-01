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
from app.agent.pinned import (  # noqa: E402
    _scan_text,
    extract_pinnable_entities,
    merge_pinned,
    render_pinned_section,
)
from app.agent.storage import load_session, save_session  # noqa: E402
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

    def test_pinned_wired_through_agent(self):
        """生产链路：压缩后实体入 pinned、_build_messages 注入锚点段、
        summarize 的 hard_cap 生效、reset 清空、会话往返保留。"""
        from app.agent.chat import EcomAgent
        from app.config.settings import settings

        dataset = _load_dataset()
        script = _script_by_id(dataset, "ret_flow_001")
        msgs = [dict(m) for m in script["messages"][:6]]  # 含订单号，触发一轮压缩

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
                    ("summary_hard_cap", 50),
                ):
                    stack.enter_context(patch.object(settings, attr, value))
                agent = EcomAgent()

                # 必须显式替换为桩 client（构造即建真实 client 的陷阱，见上例注释）
                # 注意：_compress_history 运行时才读 settings.summary_hard_cap，
                # 压缩调用与断言必须留在 patch 窗口内（测试发现：ExitStack 提前
                # 退出会让 hard_cap 回落到默认 400）
                agent.client = _StubSummaryClient("x" * 500)  # 超长摘要触发硬截断
                agent.raw_messages = msgs
                agent.summary = None
                agent._compress_history()

                # hard_cap 截断生效（prompt 级约束不可靠的实证 → 代码级兜底）
                self.assertEqual(len(agent.summary), 50)
                self.assertIn("SO202511020015", agent.pinned_facts)

                built = agent._build_messages()
                anchors = [
                    i for i, m in enumerate(built)
                    if m["role"] == "system" and "关键信息锚点" in m["content"]
                ]
                self.assertEqual(len(anchors), 1)
                self.assertIn("SO202511020015", built[anchors[0]]["content"])
                # 锚点段位于摘要之后、原始消息之前
                idx_summary = next(
                    i for i, m in enumerate(built)
                    if m["role"] == "system"
                    and m["content"].startswith("以下是此前对话的摘要"))
                self.assertGreater(anchors[0], idx_summary)
                self.assertLess(anchors[0], len(built) - len(agent.raw_messages))

                # 会话往返：pinned 随 save_session 落盘、重建后恢复
                agent.save()
                self.assertEqual(load_session(session_path)["pinned_facts"],
                                 agent.pinned_facts)

            agent.reset()
            self.assertEqual(agent.pinned_facts, [])


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


# ──────────────────── pinned facts ────────────────────

class TestPinnedExtraction(unittest.TestCase):

    def test_extract_entities_positive(self):
        """中文相邻的订单号/手机/邮箱/纯数字单号/tool args 都要抽到。"""
        msgs = [
            {"role": "user", "content":
                "订单ORD-20240115-001没到，手机13812345678，邮箱a.b@test.cn"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "query_order",
                              "arguments": "{\"order_id\": \"SO202511020015\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "运单号TRK881234567890CN"},
        ]
        entities = extract_pinnable_entities(msgs)
        for expected in ("ORD-20240115-001", "13812345678", "a.b@test.cn",
                         "SO202511020015", "TRK881234567890CN"):
            self.assertIn(expected, entities)

    def test_extract_entities_negative(self):
        """时间区间/日期/金额/弱 token（仅 1 个数字）/纯单词不得入锚。"""
        text = ("明天 14:00-18:00 送达，2025-11-03 下单，199 元，"
                "wifi6password 是密码，python 不错")
        self.assertEqual(_scan_text(text), [])

    def test_digits_inside_token_not_double_extracted(self):
        """ORD-20240115-001 内部的 20240115 不得再以纯数字身份入锚。"""
        entities = extract_pinnable_entities(
            [{"role": "user", "content": "订单号ORD-20240115-001"}])
        self.assertEqual(entities, ["ORD-20240115-001"])

    def test_merge_dedup_and_fifo_cap(self):
        merged = merge_pinned(
            ["A1-22334455"], ["A1-22334455", "B2-33445566"], max_pinned=2)
        self.assertEqual(merged, ["A1-22334455", "B2-33445566"])
        merged = merge_pinned(merged, ["C3-44556677"], max_pinned=2)
        self.assertEqual(merged, ["B2-33445566", "C3-44556677"])  # FIFO 淘汰最旧

    def test_render_section(self):
        self.assertIsNone(render_pinned_section([]))
        section = render_pinned_section(["SO202511020015", "13812345678"])
        self.assertIn("SO202511020015", section)
        self.assertIn("13812345678", section)


class TestRunnerPinMode(unittest.TestCase):

    def _run(self, pin_entities: bool):
        dataset = _load_dataset()
        script = _script_by_id(dataset, "ret_flow_001")
        return run_script(
            script, _constant_summarizer("摘要不含任何事实"),
            threshold=4, keep_recent=2, pin_entities=pin_entities)

    def test_pin_rescues_entity_from_lossy_summary(self):
        """锚点兜底：固定摘要丢弃一切，订单号仍应从锚点段找回。
        pin 模式下 in_summary 语义为『在持久上下文（锚点∪摘要）中』。"""
        result = self._run(pin_entities=True)
        state = result.final_facts["order_id"]
        self.assertTrue(state.retained)
        self.assertTrue(state.in_summary)
        self.assertIn("SO202511020015", result.final_pinned)
        self.assertGreater(result.pinned_chars, 0)
        self.assertTrue(all("SO202511020015" in r.pinned for r in result.rounds))

    def test_no_pin_entity_lost(self):
        """对照组：pin 关闭时同一脚本同一摘要器，订单号随摘要丢失。"""
        result = self._run(pin_entities=False)
        self.assertFalse(result.final_facts["order_id"].retained)
        self.assertEqual(result.final_pinned, [])
        self.assertEqual(result.pinned_chars, 0)

    def test_pin_cost_counted_in_compression_ratio(self):
        """锚点不是免费的：pin 模式的压缩比必须严格低于关闭时。"""
        self.assertLess(
            self._run(True).compression_ratio,
            self._run(False).compression_ratio,
        )

    def test_aggregate_category_pooling(self):
        """分类别聚合为跨脚本 pooling：加权均值必须等于总体 pooling 值。"""
        dataset = _load_dataset()
        results = [
            run_script(s, truncate_summarizer(), threshold=4, keep_recent=2)
            for s in dataset["scripts"]
        ]
        agg = aggregate(results)
        self.assertTrue(agg["retention_by_category"])
        n_total = sum(v["total"] for v in agg["retention_by_category"].values())
        pooled = sum(
            s.retained for r in results for s in r.final_facts.values()) / n_total
        weighted = sum(
            v["retention"] * v["total"]
            for v in agg["retention_by_category"].values()) / n_total
        self.assertAlmostEqual(weighted, pooled, places=4)


class TestStoragePinnedFacts(unittest.TestCase):

    def test_pinned_facts_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "session.json")
            save_session(path, [{"role": "user", "content": "hi"}], "摘要",
                         pinned_facts=["SO202511020015"])
            self.assertEqual(load_session(path)["pinned_facts"], ["SO202511020015"])

    def test_legacy_session_without_pinned(self):
        """旧格式会话文件无 pinned_facts 键，加载降级为 None 不报错。"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "session.json")
            save_session(path, [], None)
            self.assertIsNone(load_session(path)["pinned_facts"])


# ──────────────────── 结构化状态摘要（Step：自由文本 → 槽位 JSON） ────────────────────

_STATE_JSON = json.dumps({
    "cases": [
        {"business": "破壁机换货",
         "identifiers": {"订单号": "SO202511150102", "售后单号": "EX2025111502"},
         "status": "换货中", "next_action": "等待旧机仓检",
         "promise": "仓检通过后48小时内发新机"},
        {"business": "配件购买",
         "identifiers": {"订单号": "SO202511150120"},
         "status": "已完成", "next_action": "unknown", "promise": "unknown"},
    ],
    "commitments": [
        {"business": "电饭煲质量投诉", "current_sla": "2小时内回电",
         "previous_sla": "24小时内回电", "change_note": "已加急"},
    ],
    "preferences": ["换货新机优先使用顺丰"],
}, ensure_ascii=False)


class _StubStructuredClient:
    """summarize_structured 桩：返回预设 payload，带 usage token。"""

    def __init__(self, payload: str, completion_tokens: int = 42):
        self.payload = payload
        self.completion_tokens = completion_tokens
        self.captured: list[dict] = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.captured.append(kwargs)
        msg = types.SimpleNamespace(content=self.payload, tool_calls=None)
        usage = types.SimpleNamespace(completion_tokens=self.completion_tokens)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg)], usage=usage)


class TestParseStateJson(unittest.TestCase):

    def test_plain_json(self):
        from app.agent.summarizer import _parse_state_json
        self.assertEqual(_parse_state_json('{"a": 1}'), {"a": 1})

    def test_tolerates_fences_and_noise(self):
        """lenient 解析：容忍 markdown fences 与前后杂质。"""
        from app.agent.summarizer import _parse_state_json
        raw = f"好的，以下是状态：\n```json\n{_STATE_JSON}\n```\n以上。"
        self.assertEqual(_parse_state_json(raw), json.loads(_STATE_JSON))

    def test_garbage_returns_none(self):
        from app.agent.summarizer import _parse_state_json
        self.assertIsNone(_parse_state_json("抱歉，我无法输出"))
        self.assertIsNone(_parse_state_json(""))
        self.assertIsNone(_parse_state_json("[1, 2, 3]"))  # 非 dict


class TestStructuredSummarizerAdapter(unittest.TestCase):

    def _fn(self, payload):
        from app.agent.memory.eval.llm_summarizer import build_structured_summarizer
        return build_structured_summarizer(
            client=_StubStructuredClient(payload, completion_tokens=77))

    def test_ok_path_records_tokens_and_parse_ok(self):
        fn = self._fn(_STATE_JSON)
        out = fn([{"role": "user", "content": "hi"}], None)
        self.assertEqual(json.loads(out), json.loads(_STATE_JSON))
        entry = fn.usage_log[0]
        self.assertEqual(entry["completion_tokens"], 77)
        self.assertIs(entry["parse_ok"], True)
        self.assertEqual(fn.parse_ok, [True])

    def test_garbage_falls_back_to_prev_state(self):
        """解析失败时透传上一轮状态（陈旧优于崩坏），并标记 parse_ok=False。"""
        fn = self._fn("抱歉，我无法输出")
        prev = _STATE_JSON
        out = fn([{"role": "user", "content": "hi"}], prev)
        self.assertEqual(out, prev)
        self.assertEqual(fn.parse_ok, [False])

    def test_garbage_first_round_falls_back_to_empty_object(self):
        fn = self._fn("垃圾")
        self.assertEqual(fn([{"role": "user", "content": "hi"}], None), "{}")

    def test_prev_state_passed_to_prompt(self):
        """递归传递：上一轮状态 JSON 进入下一轮请求报文。"""
        stub = _StubStructuredClient(_STATE_JSON)
        from app.agent.memory.eval.llm_summarizer import build_structured_summarizer
        fn = build_structured_summarizer(client=stub)
        fn([{"role": "user", "content": "hi"}], _STATE_JSON)
        user_content = stub.captured[0]["messages"][-1]["content"]
        self.assertIn("【此前状态】", user_content)
        self.assertIn("SO202511150102", user_content)
        # JSON mode 开启
        self.assertEqual(stub.captured[0]["response_format"], {"type": "json_object"})


class TestStructuredEvidence(unittest.TestCase):

    def test_status_fact_gets_status_slot_values(self):
        """状态类 gold 的证据取自 status/next_action/business/change_note 槽位。"""
        from app.agent.memory.eval.factcheck import structured_evidence
        fact = Fact(id="a_exchange_status", category="status", gold="换货中")
        ev = structured_evidence(fact, _STATE_JSON)
        self.assertIn("换货中", ev)          # 案例状态
        self.assertIn("已完成", ev)          # D 单
        self.assertIn("等待旧机仓检", ev)
        self.assertIn("已加急", ev)          # 变更说明

    def test_promise_fact_gets_sla_slots(self):
        from app.agent.memory.eval.factcheck import structured_evidence
        fact = Fact(id="c_eta", category="promise", gold="3个工作日")
        ev = structured_evidence(fact, _STATE_JSON)
        self.assertIn("仓检通过后48小时内发新机", ev)
        self.assertIn("2小时内回电", ev)
        self.assertIn("24小时内回电", ev)     # 降级后的历史承诺也在证据里

    def test_literals_already_matching_gold_are_filtered(self):
        """证据包含全部相关槽位值（含字面已命中的），供完整人工核验。"""
        from app.agent.memory.eval.factcheck import structured_evidence
        fact = Fact(id="x", category="status", gold="换货中")
        ev = structured_evidence(fact, _STATE_JSON)
        self.assertIn("换货中", ev)

    def test_order_id_fact_gets_identifiers_subtree(self):
        """编号类 gold 的证据取自 identifiers 子树（任意中文类型键）。"""
        from app.agent.memory.eval.factcheck import structured_evidence
        fact = Fact(id="x", category="order_id", gold="不存在")
        ev = structured_evidence(fact, _STATE_JSON)
        self.assertIn("SO202511150102", ev)
        self.assertIn("EX2025111502", ev)
        self.assertIn("SO202511150120", ev)
        self.assertNotIn("换货中", ev)       # 状态不在编号证据里

    def test_unknown_and_unparsable_filtered(self):
        from app.agent.memory.eval.factcheck import structured_evidence
        fact = Fact(id="x", category="status", gold="换货中")
        ev = structured_evidence(fact, _STATE_JSON)
        self.assertNotIn("unknown", ev)
        self.assertEqual(structured_evidence(fact, "不是JSON"), [])
        self.assertEqual(structured_evidence(fact, None), [])


class TestRunnerStructuredRoundtrip(unittest.TestCase):
    """runner 跑结构化摘要器：JSON 状态递归传递 + factcheck 对 JSON 字面工作。"""

    def test_state_json_passthrough_and_factcheck(self):
        dataset = _load_dataset()
        script = _script_by_id(dataset, "ret_flow_001")
        seen_prev: list[str | None] = []

        def fake_summarize(old, prev):
            seen_prev.append(prev)
            return json.dumps({
                "active_cases": [
                    {"entity": "退货", "order_id": "SO202511020015",
                     "status": "处理中", "pending_action": "unknown",
                     "promise": "unknown"}],
                "updated_commitments": [], "preferences": [],
            }, ensure_ascii=False)

        result = run_script(script, fake_summarize, threshold=4, keep_recent=2)
        # 首轮 prev=None，之后每轮都收到上一轮 JSON（递归传递）
        self.assertIsNone(seen_prev[0])
        for p in seen_prev[1:]:
            self.assertIn("SO202511020015", p)
        # factcheck 对 JSON 串字面工作：订单号 gold 命中
        self.assertTrue(result.final_facts["order_id"].in_summary)
        # 摘要落盘的是 JSON 状态
        json.loads(result.final_summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
