"""记忆结构化模型的离线单测（不调用任何 LLM / Embedding API）。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.agent.memory.extraction import _parse_facts
from app.agent.memory.fact import FACT_CURRENT, FACT_HISTORICAL, MemoryFact, group_facts, render_facts
from app.agent.memory.long_term import LongTermMemory
from app.agent.memory.short_term import ShortTermMemory
from app.agent.tools import memory_tool


class TestMemoryFact(unittest.TestCase):

    def test_legacy_string_becomes_historical(self):
        fact = MemoryFact.from_dict("用户曾咨询过会员权益")
        self.assertEqual(fact.content, "用户曾咨询过会员权益")
        self.assertEqual(fact.fact_type, FACT_HISTORICAL)
        self.assertEqual(fact.category, "other")

    def test_unknown_fact_type_falls_back_to_historical(self):
        fact = MemoryFact.from_dict({"content": "x", "fact_type": "unknown"})
        self.assertEqual(fact.fact_type, FACT_HISTORICAL)

    def test_group_and_render_keeps_state_and_history_apart(self):
        facts = [
            MemoryFact(content="用户是钻石会员", fact_type=FACT_CURRENT, created_at="2026-10-30T10:00:00"),
            MemoryFact(content="用户上次咨询过退货", fact_type=FACT_HISTORICAL, created_at="2026-10-29T10:00:00"),
        ]
        current, historical = group_facts(facts)
        self.assertEqual([f.content for f in current], ["用户是钻石会员"])
        self.assertEqual([f.content for f in historical], ["用户上次咨询过退货"])

        text = render_facts(facts)
        self.assertIn("【当前状态", text)
        self.assertIn("【历史事实", text)
        self.assertIn("用户是钻石会员", text)
        self.assertIn("用户上次咨询过退货", text)


class TestShortTermMemory(unittest.TestCase):

    def test_roundtrip_and_prompt_section(self):
        stm = ShortTermMemory()
        stm.facts = [
            MemoryFact(content="用户是钻石会员", category="identity", fact_type=FACT_CURRENT),
            MemoryFact(content="用户之前问过运费", category="issue", fact_type=FACT_HISTORICAL, created_at="2026-10-29T10:00:00"),
        ]
        restored = ShortTermMemory.from_dict(stm.to_dict())
        self.assertEqual(restored.facts, stm.facts)

        section = restored.build_prompt_section()
        self.assertIn("【当前状态", section)
        self.assertIn("【历史事实", section)

    def test_from_dict_accepts_legacy_strings(self):
        stm = ShortTermMemory.from_dict({"facts": ["用户喜欢红色", "用户上次咨询过会员"]})
        self.assertEqual(len(stm.facts), 2)
        self.assertTrue(all(f.fact_type == FACT_HISTORICAL for f in stm.facts))


class TestLongTermMemory(unittest.TestCase):

    def test_save_load_roundtrip_with_fact_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            ltm = LongTermMemory(user_id="u1", memory_dir=tmp)
            ltm.facts = [
                MemoryFact(content="用户是黄金会员", category="identity", fact_type=FACT_CURRENT, source_session="s1"),
                MemoryFact(content="用户退过一次款", category="behavior", fact_type=FACT_HISTORICAL, source_session="s1"),
            ]
            ltm.save()

            loaded = LongTermMemory(user_id="u1", memory_dir=tmp)
            loaded.load()
            self.assertEqual([f.content for f in loaded.facts], ["用户是黄金会员", "用户退过一次款"])
            self.assertEqual(loaded.facts[0].fact_type, FACT_CURRENT)
            self.assertEqual(loaded.facts[1].fact_type, FACT_HISTORICAL)

    def test_load_accepts_legacy_records_without_fact_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u2.json"
            path.write_text(json.dumps({
                "user_id": "u2",
                "facts": [{"content": "用户喜欢蓝色", "category": "preference", "created_at": "2026-01-01"}],
                "interaction_summaries": [],
            }, ensure_ascii=False), encoding="utf-8")

            loaded = LongTermMemory(user_id="u2", memory_dir=tmp)
            loaded.load()
            self.assertEqual(loaded.facts[0].fact_type, FACT_HISTORICAL)


class TestFactParsing(unittest.TestCase):

    def test_json_payload_is_parsed(self):
        raw = '{"facts": [{"content": "用户是钻石会员", "category": "identity", "fact_type": "current"}]}'
        facts = _parse_facts(raw, default_fact_type=FACT_HISTORICAL)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].fact_type, FACT_CURRENT)
        self.assertTrue(facts[0].created_at)

    def test_markdown_fence_and_legacy_lines(self):
        raw = "```\n- 用户喜欢红色\n- 用户上次问过会员\n```"
        facts = _parse_facts(raw, default_fact_type=FACT_HISTORICAL)
        self.assertEqual([f.content for f in facts], ["用户喜欢红色", "用户上次问过会员"])
        self.assertTrue(all(f.fact_type == FACT_HISTORICAL for f in facts))


class TestRecallToolOutput(unittest.TestCase):

    def test_returns_structured_facts(self):
        original = memory_tool._memory_manager
        try:
            stm = ShortTermMemory()
            stm.facts = [MemoryFact(content="用户是钻石会员", fact_type=FACT_CURRENT)]
            ltm = SimpleNamespace(
                facts=[MemoryFact(content="用户退过一次款", fact_type=FACT_HISTORICAL)],
                interaction_summaries=[],
            )
            memory_tool._memory_manager = SimpleNamespace(
                memory_enabled=True, stm=stm, ltm=ltm,
            )

            result = memory_tool.recall_user_memory()
            self.assertTrue(result["success"])
            self.assertEqual(result["short_term_facts"][0]["fact_type"], FACT_CURRENT)
            self.assertEqual(result["short_term_facts"][0]["content"], "用户是钻石会员")
            self.assertEqual(result["long_term_facts"][0]["fact_type"], FACT_HISTORICAL)
        finally:
            memory_tool._memory_manager = original


if __name__ == "__main__":
    unittest.main()
