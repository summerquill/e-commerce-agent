"""Plan-and-Execute Agent 入口。

对外接口与 EcomAgent / MultiAgentOrchestrator 一致（main.py 用 settings 开关切换）。
HITL 循环藏在 chat() 内部：遇 interrupt 调 input() 问用户 → Command(resume=...) 恢复，
因此 main.py 无需改动。
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from openai import OpenAI

from app.agent.storage import delete_session, load_session, save_session
from app.agent.tools.manager import ToolManager
from app.config.settings import settings
from app.plan_execute.graph import PlanExecuteGraph
from app.schemas.response import CustomerServiceResponse, IntentType

# Command 来自 langgraph（延迟导入,避免 langgraph 未安装时影响模块加载）
try:
    from langgraph.types import Command
except ImportError:  # pragma: no cover
    Command = None  # type: ignore


class PlanExecuteAgent:
    """Plan-and-Execute 模式入口（LangGraph 图编排）。"""

    def __init__(self, session_path: Optional[str] = None):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.session_path = session_path or settings.session_path

        # 现有组件复用（与 EcomAgent 一致）
        self.tool_manager = ToolManager(
            use_mcp=settings.mcp_enabled,
            mcp_server_url=settings.mcp_server_url,
        )

        from app.agent.memory import MemoryManager
        self.memory_manager = MemoryManager(
            client=self.client,
            model=self.model,
            user_id=settings.memory_user_id,
            memory_dir=settings.memory_dir,
            memory_enabled=settings.memory_enabled,
            max_ltm_facts=settings.max_ltm_facts,
        )
        if settings.memory_enabled:
            from app.agent.tools.memory_tool import set_memory_manager
            set_memory_manager(self.memory_manager)

        from app.agent.skills import SkillManager
        self.skill_manager = SkillManager(
            skills_dir=settings.skills_dir,
            enabled=settings.skills_enabled,
        )
        if settings.skills_enabled:
            from app.agent.tools.skill_tool import set_skill_manager
            set_skill_manager(self.skill_manager)

        # 会话历史
        self.raw_messages: list[dict] = []
        self.summary: Optional[str] = None
        loaded = load_session(self.session_path)
        if loaded:
            self.summary = loaded["summary"]
            self.raw_messages = loaded["messages"]
            if loaded.get("short_term_memory"):
                self.memory_manager.restore_stm(loaded["short_term_memory"])

        # plan-execute 专属：构建图
        self.graph = PlanExecuteGraph(
            client=self.client,
            model=self.model,
            tool_manager=self.tool_manager,
            skill_manager=self.skill_manager,
        ).build()

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    def chat(self, user_input: str) -> CustomerServiceResponse:
        """处理用户输入：invoke 图 → 循环处理 interrupt → 返回结构化结果。

        HITL 循环藏在内部，main.py 不感知。
        """
        self.raw_messages.append({"role": "user", "content": user_input})

        config = {"configurable": {"thread_id": f"turn-{uuid4().hex}"}}
        self.graph.invoke(self._build_input(user_input), config)

        # invoke-resume 循环：一次执行可能触发多个 interrupt
        while True:
            st = self.graph.get_state(config)
            if not st.next:
                break
            iv = self._read_interrupt(st)
            if iv is None:
                break
            resume_val = self._handle_interrupt(iv)
            self.graph.invoke(Command(resume=resume_val), config)

        final = self.graph.get_state(config).values.get("final_response", "")
        result = self._extract_structured_response(final)
        self._post_turn_save(result)
        return result

    # ──────────────────── HITL ────────────────────

    @staticmethod
    def _read_interrupt(state):
        """从图状态中读出 interrupt value，无则返回 None。"""
        try:
            return state.tasks[0].interrupts[0].value
        except (IndexError, AttributeError):
            return None

    def _handle_interrupt(self, iv: dict) -> dict:
        """根据 interrupt 类型与用户交互，返回 resume 值。"""
        iv_type = iv.get("type")
        if iv_type == "confirm":
            print(f"\n⚠️  确认操作: {iv.get('action')}  (y/n)")
            approved = input("👤 你: ").strip().lower() in ("y", "yes", "是")
            return {"approved": approved}
        if iv_type == "clarify":
            print(f"\n❓ {iv.get('question')}")
            text = input("👤 你: ").strip()
            return {"text": text}
        # 未知类型：放行
        return {}

    # ──────────────────── 输入/输出 ────────────────────

    def _build_input(self, user_input: str) -> dict:
        """构造图的初始状态。"""
        messages: list[dict] = [{"role": "user", "content": user_input}]
        # TODO: 注入 MemoryManager.build_memory_prompt_sections() + summary 作跨轮上下文
        return {
            "user_query": user_input,
            "messages": messages,
            "plan": [],
            "plan_iteration": 0,
            "step_results": [],
            "current_step_idx": 0,
            "review_decision": "",
            "clarify_question": "",
            "clarified_answer": "",
            "final_response": "",
        }

    def _extract_structured_response(self, text: str) -> CustomerServiceResponse:
        """从最终回复提取结构化元数据。

        TODO: 复用 EcomAgent.beta.chat.completions.parse 逻辑。
        """
        return CustomerServiceResponse(
            intent=IntentType.OTHER,
            confidence=0.5,
            reply=text,
            requires_human=False,
            follow_up_question=None,
        )

    def _post_turn_save(self, result: CustomerServiceResponse) -> None:
        """轮末：更新短期记忆 + 持久化会话。"""
        self.raw_messages.append(
            {"role": "assistant", "content": result.model_dump_json(ensure_ascii=False)}
        )
        self.memory_manager.update_short_term(self.raw_messages[-6:])
        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
        )

    # ──────────────────── 生命周期 ────────────────────

    def reset(self):
        self.raw_messages = []
        self.summary = None
        self.memory_manager.reset_short_term()
        delete_session(self.session_path)

    def save(self) -> None:
        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
        )

    def close(self):
        self.memory_manager.consolidate_to_long_term(self.raw_messages, self.summary)
        self.tool_manager.close()
