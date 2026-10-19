"""Plan-and-Execute 图控制流离线单测（不需要 API key，秒级）。

与 tests/ 下其他 test_*.py 的区别：
- 那些脚本真实调用 API，验证 prompt 质量与端到端效果（需 .env 配好 key，人工跑）
- 本文件 mock 掉 LLM（脚本化响应队列），配合真实本地 ToolManager，
  只验证图编排的控制流：路由、interrupt/resume、replan、工具执行次数

跑法（在项目 .venv 中，仓库根目录执行）：
    python -m unittest tests.test_plan_execute_graph -v
或直接：
    python tests/test_plan_execute_graph.py
"""

import json
import sys
import types
import unittest
from collections import deque
from pathlib import Path

# 确保项目根目录在 sys.path 中（与 tests/ 下其他测试一致）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langgraph.types import Command  # noqa: E402

from app.agent.tools.manager import ToolManager  # noqa: E402
from app.plan_execute.graph import PlanExecuteGraph  # noqa: E402
from app.plan_execute.nodes.executor import HIGH_RISK_TOOLS  # noqa: E402


# ──────────────────── mock OpenAI 客户端 ────────────────────

class _ToolCall:
    def __init__(self, name, args):
        self.id = f"call_{name}"
        self.function = types.SimpleNamespace(
            name=name, arguments=json.dumps(args, ensure_ascii=False))


class _Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Response:
    def __init__(self, message):
        self.choices = [types.SimpleNamespace(message=message)]


class ScriptedClient:
    """脚本化 mock client：按节点 system prompt 识别节点，从对应队列弹出预设响应。

    队列元素：
    - 字符串 → 作为 message.content（planner/review 的 JSON 或 executor 的无工具回复）
    - {"__toolcall__": name, "args": {...}} → 模拟 function call
    """

    def __init__(self, planner=(), executor=(), review=(), generate=()):
        self._queues = {
            "planner": deque(planner),
            "executor": deque(executor),
            "review": deque(review),
            "generate": deque(generate),
        }
        self.executed_tools: list[tuple[str, dict]] = []
        # 捕获每次实际发给 SDK 的 messages（报文契约回归用）
        self.captured_request_messages: list[list] = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.captured_request_messages.append(kwargs["messages"])
        system_prompt = kwargs["messages"][0]["content"]
        if "规划器" in system_prompt:
            kind = "planner"
        elif "步骤执行器" in system_prompt:
            kind = "executor"
        elif "执行审查器" in system_prompt:
            kind = "review"
        else:
            kind = "generate"
        item = self._queues[kind].popleft()
        if isinstance(item, dict) and "__toolcall__" in item:
            return _Response(_Message(tool_calls=[
                _ToolCall(item["__toolcall__"], item["args"])]))
        return _Response(_Message(content=item))


# ──────────────────── 组装 / 运行辅助 ────────────────────

def initial_state(query: str) -> dict:
    return {
        "user_query": query,
        "messages": [{"role": "user", "content": query}],
        "plan": [],
        "plan_iteration": 0,
        "step_results": [],
        "current_step_idx": 0,
        "review_decision": "",
        "pending_clarification": None,
        "final_response": "",
    }


def build_graph(client: ScriptedClient):
    """真实本地 ToolManager + spy 包裹 execute_tool 记录实际调用。"""
    tm = ToolManager(use_mcp=False)
    real_execute = tm.execute_tool

    def spy(name, args):
        client.executed_tools.append((name, args))
        return real_execute(name, args)

    tm.execute_tool = spy

    graph = PlanExecuteGraph(
        client=client,
        model="fake-model",
        tool_manager=tm,
        skill_manager=None,
        memory_manager=None,
    ).build()
    return graph


def invoke_to_end(graph, state, resumes=None, seen_interrupts=None):
    """invoke + interrupt/resume 循环，返回最终图状态 values。

    resumes: {interrupt 类型: deque([resume 值, ...])}，按中断顺序消费。
    seen_interrupts: 可选 list，依次收集每次 interrupt 的 payload 供断言。
    """
    resumes = resumes or {}
    config = {"configurable": {"thread_id": "test-thread"}}
    graph.invoke(state, config)
    while True:
        st = graph.get_state(config)
        if not st.next:
            break
        interrupt_value = st.tasks[0].interrupts[0].value
        if seen_interrupts is not None:
            seen_interrupts.append(interrupt_value)
        resume_value = resumes[interrupt_value["type"]].popleft()
        graph.invoke(Command(resume=resume_value), config)
    return graph.get_state(config).values


def plan_one_step(desc, tool_hint=None):
    return json.dumps(
        {"steps": [{"id": "step_1", "description": desc, "tool_hint": tool_hint}]},
        ensure_ascii=False,
    )


def message_contents(messages) -> list[str]:
    """取对话记录文本内容（兼容 dict 与 LangChain Message 对象）。"""
    return [
        m["content"] if isinstance(m, dict) else m.content
        for m in messages
    ]


# ──────────────────── 测试用例 ────────────────────

class PlanExecuteGraphFlowTest(unittest.TestCase):
    """覆盖 10 类控制流路径。"""

    def setUp(self):
        # confirm 用例临时开启高风险工具，tearDown 兜底清空避免污染其他用例
        self.addCleanup(HIGH_RISK_TOOLS.clear)

    def test_tool_success_path(self):
        """A: planner → executor 工具调用成功 → generate。"""
        client = ScriptedClient(
            planner=[plan_one_step("查询订单详情", "query_order")],
            executor=[{"__toolcall__": "query_order",
                       "args": {"order_id": "ORD-20240115-001"}}],
            generate=["您的订单状态是已发货～"],
        )
        values = invoke_to_end(build_graph(client), initial_state("查下订单 ORD-20240115-001"))

        result = values["step_results"][0]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool"], "query_order")
        self.assertIn("已发货", values["final_response"])
        self.assertIn(
            ("query_order", {"order_id": "ORD-20240115-001"}),
            client.executed_tools,
        )

    def test_clarify_then_retry_same_step(self):
        """B: 缺信息 → clarify interrupt → 用户回答 → 重试同一步成功。

        回归点：追问不带 options 时 payload 无 options 键（向后兼容）。
        """
        client = ScriptedClient(
            planner=[plan_one_step("查询物流", "query_logistics")],
            executor=[
                '{"needs_clarify": true, "question": "请问订单号是多少？"}',
                {"__toolcall__": "query_logistics",
                 "args": {"order_id": "ORD-20240115-001"}},
            ],
            generate=["物流信息查到啦"],
        )
        seen = []
        values = invoke_to_end(
            build_graph(client),
            initial_state("我的订单到哪了"),
            resumes={"clarify": deque([{"text": "ORD-20240115-001"}])},
            seen_interrupts=seen,
        )

        self.assertEqual(
            [r["status"] for r in values["step_results"]],
            ["needs_clarify", "success"],
        )
        self.assertEqual(values["step_results"][-1]["tool"], "query_logistics")
        # 问答进对话记录（messages），pending 已清空
        self.assertIn("ORD-20240115-001", " ".join(message_contents(values["messages"])))
        self.assertIsNone(values["pending_clarification"])
        self.assertNotIn("options", seen[0])

    def test_failed_then_replan_recovers(self):
        """C: 最后一步 failed → review LLM 判 replan → 新计划执行成功。

        回归点：failed 也推进 idx，"全部完成"判断不得吃掉 failed 的 replan 机会。
        """
        client = ScriptedClient(
            planner=[
                plan_one_step("按订单号查询", "query_order"),
                plan_one_step("列出用户订单", "list_user_orders"),
            ],
            executor=[
                {"__toolcall__": "query_order", "args": {"order_id": "ORD-NOPE"}},
                {"__toolcall__": "list_user_orders", "args": {}},
            ],
            review=['{"decision": "replan", "reason": "订单号查不到，先列订单"}'],
            generate=["已为您列出订单"],
        )
        values = invoke_to_end(build_graph(client), initial_state("查 ORD-NOPE"))

        self.assertEqual(values["plan_iteration"], 2)
        self.assertEqual(len(values["plan"]), 1)
        self.assertIn("列出", values["plan"][0]["description"])
        self.assertEqual(
            [r["status"] for r in values["step_results"]],
            ["failed", "success"],
        )

    def test_confirm_rejected_skips_tool(self):
        """D: 高风险工具审批被拒 → skipped，工具绝不执行。"""
        HIGH_RISK_TOOLS.add("apply_refund")
        client = ScriptedClient(
            planner=[plan_one_step("申请退款", "apply_refund")],
            executor=[{"__toolcall__": "apply_refund",
                       "args": {"order_id": "ORD-20240120-002", "reason": "不想要了"}}],
            generate=["好的，暂不操作退款"],
        )
        values = invoke_to_end(
            build_graph(client),
            initial_state("帮我退款"),
            resumes={"confirm": deque([{"approved": False}])},
        )

        self.assertEqual(values["step_results"][-1]["status"], "skipped")
        self.assertEqual(client.executed_tools, [])

    def test_confirm_approved_executes_once(self):
        """E: 审批通过 → 成功执行，且 resume 重放下工具只执行一次（幂等回归）。"""
        HIGH_RISK_TOOLS.add("apply_refund")
        client = ScriptedClient(
            planner=[plan_one_step("申请退款", "apply_refund")],
            executor=[{"__toolcall__": "apply_refund",
                       "args": {"order_id": "ORD-20240120-002", "reason": "不想要了"}}],
            generate=["退款已发起"],
        )
        values = invoke_to_end(
            build_graph(client),
            initial_state("帮我退款"),
            resumes={"confirm": deque([{"approved": True}])},
        )

        self.assertEqual(values["step_results"][-1]["status"], "success")
        self.assertEqual(
            client.executed_tools,
            [("apply_refund",
              {"order_id": "ORD-20240120-002", "reason": "不想要了"})],
        )

    def test_direct_reply_no_tool(self):
        """F: 无需工具的步骤（闲聊）→ direct_reply 成功，不调任何工具。"""
        client = ScriptedClient(
            planner=[plan_one_step("直接回复用户的问候")],
            executor=['{"direct_reply": "你好呀，我是小夕，有什么可以帮你？"}'],
            generate=["你好呀～"],
        )
        values = invoke_to_end(build_graph(client), initial_state("你好"))

        result = values["step_results"][0]
        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["tool"])
        self.assertIn("你好呀", result["output"])
        self.assertEqual(client.executed_tools, [])

    def test_all_llm_request_messages_have_role(self):
        """I: 发给 SDK 的每条消息必须是 {role, content} dict。

        回归：state["messages"] 经 add_messages reducer 变成 LangChain Message
        对象，planner 透传前必须转回 OpenAI 格式，否则真实 API 422
        （missing field `role`）。mock 不校验报文格式，故离线显式断言。
        """
        client = ScriptedClient(
            planner=[plan_one_step("直接回复用户的问候")],
            executor=['{"direct_reply": "你好呀～"}'],
            generate=["你好呀"],
        )
        invoke_to_end(build_graph(client), initial_state("你好"))

        self.assertTrue(client.captured_request_messages)
        for req_msgs in client.captured_request_messages:
            for m in req_msgs:
                self.assertIsInstance(m, dict, f"非 dict 消息: {m!r}")
                self.assertIn("role", m, f"缺 role 的消息: {m!r}")

    def test_planner_garbage_output_fallback(self):
        """G: planner 返回非法 JSON → 兜底单步计划，图仍可走通。"""
        client = ScriptedClient(
            planner=["我也不知道该咋办"],
            executor=['{"direct_reply": "抱歉能再说详细些吗？"}'],
            generate=["请补充描述"],
        )
        values = invoke_to_end(build_graph(client), initial_state("???"))

        self.assertEqual(len(values["plan"]), 1)
        self.assertIn("直接回复", values["plan"][0]["description"])
        self.assertEqual(values["step_results"][0]["status"], "success")

    def test_clarify_with_options_payload(self):
        """H: needs_clarify 带 options → interrupt payload 含候选，
        回答后重试同一步成功（结构化追问链路回归）。
        """
        client = ScriptedClient(
            planner=[plan_one_step("申请退款", "apply_refund")],
            executor=[
                '{"needs_clarify": true, "question": "请问退款原因是什么？", '
                '"options": ["不想要了", "质量问题", "七天无理由"]}',
                {"__toolcall__": "apply_refund",
                 "args": {"order_id": "ORD-20240120-002", "reason": "质量问题"}},
            ],
            generate=["退款已发起"],
        )
        seen = []
        values = invoke_to_end(
            build_graph(client),
            initial_state("我要退款"),
            resumes={"clarify": deque([{"text": "质量问题"}])},
            seen_interrupts=seen,
        )

        self.assertEqual(seen[0]["type"], "clarify")
        self.assertEqual(seen[0]["question"], "请问退款原因是什么？")
        self.assertEqual(
            seen[0]["options"], ["不想要了", "质量问题", "七天无理由"])
        self.assertEqual(
            [r["status"] for r in values["step_results"]],
            ["needs_clarify", "success"],
        )
        self.assertIn("质量问题", " ".join(message_contents(values["messages"])))
        self.assertEqual(
            client.executed_tools,
            [("apply_refund",
              {"order_id": "ORD-20240120-002", "reason": "质量问题"})],
        )

    def test_multi_round_clarify_keeps_earlier_answers(self):
        """J: 同一步两轮追问 → 第一轮答案不得被第二轮覆盖。

        回归（manual-test 发现）：旧实现 clarified_answer 为单槽位，
        第二轮回答覆盖第一轮，executor 丢失已选订单信息导致重复追问。
        新实现：问答 append 进 messages 对话记录，executor 最终一次请求
        的上下文必须同时包含两轮回答。
        """
        client = ScriptedClient(
            planner=[plan_one_step("为用户选定的订单发起退款", "apply_refund")],
            executor=[
                '{"needs_clarify": true, "question": "请问要为哪个订单退款？", '
                '"options": ["Nike Air Max 270 运动鞋", "AirPods Pro 2"]}',
                '{"needs_clarify": true, "question": "请问退款原因是什么？", '
                '"options": ["尺码不合适", "不想要了"]}',
                {"__toolcall__": "apply_refund",
                 "args": {"order_id": "ORD-20240115-001", "reason": "尺码不合适"}},
            ],
            generate=["退款已发起"],
        )
        values = invoke_to_end(
            build_graph(client),
            initial_state("我要退款"),
            resumes={"clarify": deque([
                {"text": "Nike Air Max 270 运动鞋"},
                {"text": "尺码不合适"},
            ])},
        )

        self.assertEqual(
            [r["status"] for r in values["step_results"]],
            ["needs_clarify", "needs_clarify", "success"],
        )
        # 最后一轮 executor 请求的对话记录里，两轮回答同时在
        executor_reqs = [
            msgs for msgs in client.captured_request_messages
            if any("【当前步骤】" in str(m.get("content", "")) for m in msgs)
        ]
        self.assertEqual(len(executor_reqs), 3)
        last_dialogue = " ".join(
            str(m.get("content", "")) for m in executor_reqs[-1]
        )
        self.assertIn("Nike Air Max 270 运动鞋", last_dialogue)
        self.assertIn("尺码不合适", last_dialogue)
        # 工具成功执行，答案解析结果落 step_results（结构化上下文）
        self.assertEqual(values["pending_clarification"], None)
        self.assertEqual(
            client.executed_tools,
            [("apply_refund",
              {"order_id": "ORD-20240115-001", "reason": "尺码不合适"})],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
