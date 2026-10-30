"""短期记忆：会话内事实提取与管理。

从当前对话中提取用户关键信息（身份、偏好、情绪等），
注入 system prompt 增强 Agent 的上下文感知能力。

facts 使用统一的 MemoryFact 结构，区分「当前状态」与「历史事实」；
旧版会话文件里的 list[str] 通过 MemoryFact.from_dict 自动迁移。
"""

from __future__ import annotations

from openai import OpenAI

from app.agent.memory.extraction import extract_short_term_facts
from app.agent.memory.fact import MemoryFact, render_facts


class ShortTermMemory:
    """会话内短期记忆：从对话中提取结构化事实，增强上下文感知。"""

    def __init__(self):
        self.facts: list[MemoryFact] = []

    def update(self, client: OpenAI, model: str, recent_messages: list[dict]) -> None:
        """从最近的对话消息中提取/更新事实。"""
        self.facts = extract_short_term_facts(
            client, model, recent_messages, self.facts,
        )

    def build_prompt_section(self) -> str | None:
        """生成注入 system prompt 的短期记忆片段。"""
        if not self.facts:
            return None
        facts_text = render_facts(
            self.facts, with_category=False, with_time=True,
        )
        return (
            "以下是本次对话中提取的用户信息（短期记忆）。"
            "历史事实不能当作当前状态：\n"
            f"{facts_text}"
        )

    def reset(self) -> None:
        self.facts = []

    def to_dict(self) -> dict:
        return {"facts": [f.to_dict() for f in self.facts]}

    @classmethod
    def from_dict(cls, data: dict) -> ShortTermMemory:
        stm = cls()
        stm.facts = [
            MemoryFact.from_dict(item) for item in data.get("facts", [])
        ]
        return stm
