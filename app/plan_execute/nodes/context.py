"""节点共享依赖与 LLM 调用辅助。

节点实现为纯函数 `node(state, deps) -> dict`，deps 由 graph 装配时通过
functools.partial 注入（替代旧的类绑定方法，拆文件后依赖仍显式传递）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from app.plan_execute.state import Step, StepResult


@dataclass
class NodeDeps:
    """各节点共享的运行时依赖（app 级单例，非线程状态）。"""

    client: Any               # OpenAI 兼容客户端
    model: str
    temperature: float
    tool_manager: Any         # ToolManager
    skill_manager: Any = None  # SkillManager（注入 planner）
    memory_manager: Any = None  # MemoryManager（注入 planner 的跨轮上下文）
    # executor 动作解析缓存：confirm interrupt resume 会重放整个节点，
    # 缓存保证 LLM 解析只发生一次（key 含 thread_id+step 定位，天然按轮/步隔离）
    resolve_cache: dict = field(default_factory=dict)


# ──────────────────── LLM 调用辅助 ────────────────────

def chat_completion(
    deps: NodeDeps,
    messages: list[dict],
    *,
    temperature: Optional[float] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[str] = None,
    json_mode: bool = False,
) -> Any:
    """统一封装 chat.completions.create，透传 tools / json_mode。"""
    kwargs: dict[str, Any] = {
        "model": deps.model,
        "messages": messages,
        "temperature": deps.temperature if temperature is None else temperature,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return deps.client.chat.completions.create(**kwargs)


def parse_lenient_json(text: str) -> Optional[dict]:
    """宽容解析 LLM 返回的 JSON：去 markdown 围栏 / 截取首个 {...} 片段。"""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


# ──────────────────── 上下文格式化 ────────────────────

def tool_catalog_text(deps: NodeDeps) -> str:
    """从 tool_manager 的 OpenAI schema 提取 name+description 清单（给 planner）。"""
    lines = []
    for td in deps.tool_manager.tool_definitions:
        fn = td.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def skill_catalog_text(deps: NodeDeps) -> str:
    """skill catalog 文本（planner 注入）；未启用返回空串。"""
    if not deps.skill_manager or not getattr(deps.skill_manager, "enabled", False):
        return ""
    return deps.skill_manager.build_catalog_prompt()


def memory_sections(deps: NodeDeps) -> list[dict]:
    """跨轮记忆 system 消息段（planner 注入）；未启用返回空列表。"""
    if not deps.memory_manager:
        return []
    return deps.memory_manager.build_memory_prompt_sections()


def format_plan(plan: list[Step]) -> str:
    lines = []
    for i, s in enumerate(plan):
        hint = s.get("tool_hint") or "无"
        lines.append(f'{i + 1}. [{s.get("id", f"step_{i+1}")}] {s.get("description", "")}（工具提示: {hint}）')
    return "\n".join(lines)


def format_step_results(results: list[StepResult]) -> str:
    """把已累积的执行结果渲染为 LLM 可读文本（executor 取依赖 / review、generate 评估用）。"""
    if not results:
        return "（暂无）"
    lines = []
    for r in results:
        lines.append(
            f'- 步骤 {r.get("step_id")} | 工具: {r.get("tool")} | 状态: {r.get("status")}\n'
            f'  输入: {json.dumps(r.get("input", {}), ensure_ascii=False)}\n'
            f'  输出: {r.get("output") if r.get("output") is not None else "（无）"}\n'
            f'  错误: {r.get("error") or "（无）"}'
        )
    return "\n".join(lines)


def normalize_steps(raw: Any, user_query: str) -> list[Step]:
    """校验/归一化 planner LLM 输出为 list[Step]，id 强制为 step_N。"""
    items = raw.get("steps") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        items = []

    steps: list[Step] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description", "")).strip()
        if not desc:
            continue
        hint = item.get("tool_hint")
        steps.append(Step(
            id=f"step_{len(steps) + 1}",
            description=desc,
            tool_hint=str(hint) if hint else None,
        ))

    if not steps:
        # LLM 异常输出的兜底：单步直接回复，交由 executor 的 direct_reply 路径
        steps.append(Step(
            id="step_1",
            description=f"根据用户请求直接回复用户: {user_query}",
            tool_hint=None,
        ))
    return steps


def classify_tool_output(raw: str) -> tuple[str, Optional[str]]:
    """根据工具返回 JSON 判断 success/failed。

    本地工具约定：失败返回 {"success": false, "error": ...} 或含 error 键；
    ToolManager 对未知工具同样返回 {"error": ...}。
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return "success", None
    if isinstance(data, dict) and (data.get("success") is False or "error" in data):
        return "failed", str(data.get("error", "工具执行失败"))
    return "success", None
