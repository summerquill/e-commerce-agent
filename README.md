# E-Commerce-Agent 电商客服 Agent 系统

> 声明：本项目基于 [ecom-service-agent](https://github.com/HuaiNan54321/ecom-service-agent) 进行架构升级开发。基线项目已实现 ReAct / Multi-Agent / Memory / Skills / Evaluation 等能力；本人在此基础上引入 **LangGraph 图编排**，新增 **Plan-and-Execute pattern + HITL 人机协作**，详见下文「核心升级」。

一个面向电商客服场景的智能 Agent 系统，支持订单查询、退换货、商品咨询、售后等全链路工单处理。通过可切换的执行范式（ReAct / Multi-Agent / Plan-and-Execute）探索不同 Agent 架构在真实业务下的取舍。

---

## 核心升级：基于 LangGraph 的 Plan-and-Execute

基线项目的 ReAct 循环是"走一步看一步"，难以处理需要全局规划的长链路工单（如"查订单 → 查物流 → 查退货政策"这类多步骤、带依赖的请求）。本次升级引入 LangGraph 状态图编排，实现"先规划、再执行、可重规划、可中断"的 Plan-and-Execute 架构。

### 图结构

```
                         START
                           │
                           ▼
                    ┌─────────────┐
            ┌──────│   planner    │◄──────┐  replan (动态重规划)
            │      └─────────────┘       │
            │            │               │
            │            ▼               │
            │      ┌──────────────┐      │
            │      │   executor   │◄─┐   │
            │      └──────────────┘  │   │
            │        │  ▲             │   │
            │  confirm│  │clarify答   │   │
            │ (内部   │  │     ┌──────┴───┴─┐
            │ interrupt)│  │     │  review   │  ← 路由中枢
            │        │  │  └────►│ (hybrid)  │
            │        ▼  │        └──┬───┬──┬┘
            │     [执行]│           │   │  │
            │           └───────────┘   │  │
            │              (executor⇄review)
            │                              │
            │              ┌───────────────┘
            │              ▼  clarify 触发
            │      ┌──────────────┐
            │      │   clarify    │  (interrupt)
            │      └──────────────┘
            │              │ 回 executor 重试同 step
            ▼
      ┌──────────────┐
      │   generate    │────► END
      └──────────────┘
```

### 设计亮点

- **Dynamic Replanning**：`review` 作为路由中枢，发现计划走不通时回到 `planner` 重新规划——这是 ReAct 做不到的全局重规划能力。
- **HITL 双模式**：区分两种人机中断——`confirm`（高风险操作审批，executor 内部 interrupt）/ `clarify`（缺信息追问，独立节点）。两者检测时机不同：confirm 是执行前本地可知，clarify 是执行后由 review 评估才知信息不足。
- **描述型 Step + 隐式依赖解析**：plan 只给步骤语义描述（不含工具参数），executor 执行时由 LLM 读历史 `step_results` 动态解析 `{tool, args}`，自然处理步骤间依赖，无需显式 DAG 或模板占位符。
- **Review Hybrid 路由**：确定性信号（缺信息/全部完成/超迭代上限）走规则快路径不调 LLM，仅 `failed → 是否 replan` 走 LLM，平衡准确性与成本。
- **可插拔持久化层**：基于 LangGraph `BaseCheckpointSaver` 抽象，dev 用 MemorySaver，prod 可零成本切换 PostgresSaver，支持中断后跨进程恢复。

> 详细设计决策、trade-off 与面试讲点见 [docs/plan-execute-design.md](docs/plan-execute-design.md)。

---

## 系统架构

项目支持三种可切换的 Agent 执行范式，通过 `.env` 开关选择：

| 模式 | 开关 | 特点 | 适用场景 |
|------|------|------|---------|
| ReAct + MCP + RAG | 默认 | 思考-行动-观察循环，工具调用 + 知识检索 | 单轮、即时问答 |
| Multi-Agent | `MULTI_AGENT_ENABLED=true` | Router 路由 + 子 Agent 分工 | 按意图领域隔离 |
| **Plan-and-Execute** | `PLAN_EXECUTE_ENABLED=true` | LangGraph 图编排 + 动态重规划 + HITL | 长链路、多步骤工单 |

### 能力矩阵

- **工具调用**：本地工具 + MCP 远程工具统一调度（`ToolManager`，MCP 失败自动降级本地）
- **知识检索（RAG）**：政策/退货规则检索，支持 numpy / chroma 双后端
- **记忆系统**：短期记忆（STM，会话内）+ 长期记忆（LTM，跨会话事实沉淀）
- **技能模块**：Markdown 形式的可复用流程编排（`app/skills/`）
- **评估体系**：过程层（工具准确率/调用效率）+ 结果层（意图/忠实度/回答质量）双维度离线评测

---

## 项目结构

```
app/
├── agent/            # ReAct 基线 Agent（chat.py / tools / memory / skills / rag）
├── multi_agent/      # Multi-Agent 模式（orchestrator / router / agents）
├── plan_execute/     # ★ LangGraph Plan-and-Execute 升级
│   ├── state.py       #    图状态（AgentState / Step / StepResult）
│   ├── graph.py       #    图构建 + 节点（planner/executor/review/clarify/generate）
│   ├── agent.py       #    PlanExecuteAgent 入口（接口兼容 EcomAgent）
│   ├── checkpointer.py#    持久化工厂（memory / postgres 可切）
│   └── prompts.py     #    各节点 prompt
├── evaluation/       # 离线评估框架（evaluator / sandbox / cases）
├── config/           # 配置（settings.py）
├── prompts/          # 系统 prompt
├── schemas/          # 结构化输出 schema
└── mcp_client/       # MCP 同步封装
main.py               # 入口：按 settings 切换运行模式
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env:
#   OPENAI_API_KEY=...
#   OPENAI_BASE_URL=...        # 兼容 OpenAI 兼容接口
#   PLAN_EXECUTE_ENABLED=true  # 启用 Plan-and-Execute 模式

# 3. 运行
python main.py
```

交互命令：`quit/exit` 退出 · `reset` 重置对话 · `memory` 查看记忆 · `skills` 查看技能。

---

## 技术栈

- **LLM 编排**：LangGraph（状态图 + interrupt + checkpointer）
- **LLM 调用**：OpenAI SDK（function calling / structured output）
- **工具协议**：MCP（Model Context Protocol）
- **向量检索**：ChromaDB / numpy（双后端可切）
- **数据校验**：Pydantic v2
- **语言**：Python 3.10+

---

## License

继承基线项目 License，详见 [LICENSE](LICENSE)。
