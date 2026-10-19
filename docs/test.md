# 测试指南

## 测试分层

| 层级 | 位置 | 依赖 | 用途 |
|------|------|------|------|
| 离线控制流单测 | `tests/test_plan_execute_graph.py` | mock LLM（ScriptedClient）+ 真实本地 ToolManager，**不触网** | 图编排回归：路由、interrupt/resume、replan、工具执行幂等 |
| 真实 API 场景脚本 | `tests/` 下其余 `test_*.py` | `.env` 配好真实 key | prompt 质量、意图识别、多轮上下文等**人工验证**（非自动化断言） |
| 端到端 CLI | `python main.py` | `.env` 真实 key + 模式开关 | Plan-and-Execute 全链路效果验证 |

所有命令都在激活 `.venv` 后、于仓库根目录执行。

***

## 离线单测（控制流回归）

```bash
source .venv/bin/activate
python -m unittest tests.test_plan_execute_graph -v
```

**原理**：`ScriptedClient` 按节点 system prompt 识别当前节点（「规划器」→ planner /「步骤执行器」→ executor /「执行审查器」→ review / 其余 → generate），从对应预设响应队列弹出；工具侧用真实本地 `ToolManager`（spy 包裹 `execute_tool` 记录实际调用）。

**何时跑**：改图结构、节点逻辑、prompt 输出契约后先跑它回归（秒级、不需要 API key，可直接挂 CI）。

### 覆盖矩阵（8 用例）

| 用例 | 场景 | 回归点 |
|------|------|--------|
| A | planner → executor 工具成功 → generate | 快乐路径，工具参数与调用记录正确 |
| B | 缺信息 → clarify interrupt → 回答 → 重试同一步成功 | clarify 后 `current_step_idx` 不前进；无 options 时 payload **无 options 键**（向后兼容） |
| C | 最后一步 failed → review 判 replan → 新计划执行成功 | failed 也推进 idx，「全部完成」判断不得吃掉 failed 的 replan 机会 |
| D | 高风险工具审批被拒 | 结果 skipped，工具**绝不执行** |
| E | 审批通过 | interrupt resume 重放下工具只执行一次（动作解析幂等回归） |
| F | 闲聊 direct_reply | 不调任何工具，`tool=None` |
| G | planner 返回非法 JSON | 兜底单步直接回复计划，图仍走通 |
| H | needs_clarify **带 options** | interrupt payload 含 `question` + `options`，回答后重试成功（结构化追问链路） |

***

## Plan-and-Execute 端到端验证（真实 API）

### 准备

```bash
# .env 中启用（其余 key 配置同基线）
PLAN_EXECUTE_ENABLED=true
```

```bash
source .venv/bin/activate
python main.py
```

启动横幅显示「**Plan-and-Execute 规划执行模式**」即切换成功（main.py 通过 settings 开关自动选择，无需改代码）。

### 验证话术

| 场景 | 输入示例 | 观察点 |
|------|---------|--------|
| 多步规划 + 步骤依赖 | `帮我查一下我最近一个订单的物流进度` | `🧭 [planner v1] 生成 N 个步骤` → 逐步 `🔧 [执行]` → generate 汇总；订单号由 executor 从上游步骤结果中取（隐式依赖解析） |
| 结构化追问（options 点选） | `我要退款`（不给订单号和原因） | clarify 渲染 `1. xxx 2. xxx` 候选；分别试「输入序号」与「自由文字」两种回答 |
| 纯文本追问（向后兼容） | 缺参数且无明确候选的问题 | 只打印问题、无序号列表，直接输入回答 |
| replan 动态重规划 | `帮我查订单 ORD-XXX 的物流`（编一个查不到的单号） | `🔎 [review] failed → replan` → `🧭 [planner v2] 生成 N 个步骤（重规划）` |
| 跨轮上下文 | 第二轮说 `我刚才说的那个订单` | 见下方「预期边界」——当前不注入 |

### 运行时日志对照

| 日志 | 含义 |
|------|------|
| `🧭 [planner vN] 生成 N 个步骤` | 规划 / 重规划（v2+ 为 replan 产物） |
| `🔧 [执行] step_N → tool(args)` | 每步实际调用的工具与参数（观察依赖参数是否从上游正确取得） |
| `🔎 [review] failed → replan/done` | review 慢路径（LLM）的路由决策与理由 |
| `❓ 追问 + 序号列表` | clarify interrupt 触发（HITL） |

### 预期边界（当前版本特性，非 bug）

- **confirm 审批不会触发**：`HIGH_RISK_TOOLS` 白名单当前为空，高风险审批链路已实现但未启用
- **跨轮上下文不会注入**：`agent.py._build_input` 的 memory 注入为 TODO，跨轮指代（"我刚才那个订单"）planner 拿不到，每轮 thread_id 相互隔离
