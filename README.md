# E-commerce Customer Service Agent

面向电商客服场景的智能 Agent 系统，支持订单查询、退换货、物流跟踪、政策咨询等全链路工单处理。项目以「可切换的 Agent 执行范式 + 可复现的 RAG 检索优化」为主线，内置离线评测体系，所有架构决策均有量化数据支撑。

> 基于 [ecom-service-agent](https://github.com/HuaiNan54321/ecom-service-agent) 基线做架构升级：在 ReAct / Multi-Agent / Memory / Skills 之上，引入 LangGraph Plan-and-Execute 图编排与 dense+sparse 混合检索 + CrossEncoder 精排的 RAG 管线。

## 1. Project Overview

### 项目目标

- 用同一套工具/知识库/记忆底座，对比三种 Agent 执行范式在真实客服业务下的取舍：**ReAct**（即时问答）、**Multi-Agent**（意图分工）、**Plan-and-Execute**（长链路多步骤工单）。
- 把 RAG 检索当作可测量、可迭代的工程问题：从纯 dense 基线出发，按「失败案例驱动」逐步引入 BM25 混合检索与 CrossEncoder 精排，每步都有评测数据背书。

### 核心业务场景

| 场景 | 支撑能力 |
| --- | --- |
| 订单查询 / 物流跟踪 | `query_order` / `list_user_orders` / `query_logistics` 工具 |
| 退换货 | `apply_refund`（真实修改订单状态 + 幂等）+ `process-return` 技能 |
| 政策咨询（退货规则/会员权益/配送/FAQ） | RAG 知识检索（4 篇政策文档，dense/hybrid/hybrid_rerank 三模式） |
| 商品咨询 | `query_product` 模糊搜索 + `product-recommend` 技能 |
| 个性化服务 | STM/LTM 记忆（`recall_user_memory`）+ 对话压缩（pinned facts 锚定） |

## 2. Architecture

### LangGraph Workflow

Plan-and-Execute 模式（`PLAN_EXECUTE_ENABLED=true`）的状态图：

```
START → planner → executor ⇄ review ──► generate → END
                  ↕  ▲   └─┬──┴──┐
              confirm │      │   └─► planner（replan，≤ max_plan_iterations）
           (interrupt)│      └──────► clarify（interrupt，HITL 追问）
```

- `review` 是路由中枢：确定性信号（needs_clarify / 迭代上限 / 步骤全部完成）走规则快路径不调 LLM，仅「失败是否 replan」走 LLM 判断。
- 可插拔持久化：基于 LangGraph `BaseCheckpointSaver`，dev 用 MemorySaver，prod 可切 PostgresSaver，中断后可跨进程恢复。

### Planner / Executor / Review

- **Planner**：输出描述型步骤（只含语义描述，不含工具参数），`response_format=json_object` + 容错解析，垃圾输出兜底为单步直接回复。
- **Executor**：读历史 `step_results` 动态解析 `{tool, args}`（隐式依赖解析，无需显式 DAG）；输出四选一契约：`tool_call` / `needs_clarify` / `needs_replan` / `direct_reply`；高风险操作（如退款）执行前触发 confirm interrupt。
- **Review**：hybrid 路由（规则快路径 + LLM 重规划判断），快路径顺序固定为 needs_clarify → 迭代上限 → needs_replan → 失败重查 → 完成判定。
- **Clarify**：纯中继 HITL 节点（不调 LLM），带 options 的追问在 CLI 渲染为点选。

### Tool System

- `ToolManager` 统一调度 8 个工具：本地工具 + MCP 远程工具（MCP 连接失败自动降级本地）。
- 工具清单：`query_order` / `query_product` / `query_logistics` / `apply_refund` / `search_knowledge` / `list_user_orders` / `recall_user_memory` / `load_skill`。
- Skills 是 Markdown 形式的可复用流程模块（`process-return` / `track-order` / `product-recommend`）：SkillManager 扫描目录解析 YAML frontmatter 生成目录注入系统 prompt，`load_skill` 工具按需加载完整指令——流程知识与工具实现解耦。

### RAG Pipeline

```
knowledge/*.md ──chunker──► chunks ──embedder──► backend ──┐
（结构优先切分：## 二级标题   （bge-small-zh   （numpy /     │
  为边界，>1200字兜底拆分）    -v1.5 本地）      chroma）     ▼
query ──► dense top-10 ──► BM25 补位(±1) ──► CrossEncoder 精排 ──► top-5
              （dense 模式到此为止 / hybrid 到补位 / hybrid_rerank 全流程）
```

- **切分**：以 Markdown `##` 为核心边界，超长章节按三级标题/段落兜底，不用语义切分——chunk_id 稳定可复现。
- **双后端**：NumpyBackend（零依赖暴力扫描，评测基线可复现）/ ChromaBackend（HNSW，生产），接口一致无缝切换；启动时校验索引 embedding 模型一致性。
- **三模式**：`.env` 的 `RETRIEVAL_MODE` 枚举切换 `dense` / `hybrid` / `hybrid_rerank`，共用同一份 chunk 列表保证 chunk_id 对齐。
- **文档级聚合**：宽泛问题（top-k 被同一文档占据 ≥3 条且文档 ≤10 chunk）触发整篇展开，生产工具层使用。
- **可观测**：`search_knowledge` 输出结构化检索日志（排名/score/chunk_id/章节/展开标记/正文预览），终端 + `app/sessions/logs/rag.log` 双写。

## 3. Key Engineering Decisions

### Planner 与 Skill 的职责划分

Planner 负责「这一次工单要做什么」（动态规划的语义步骤），Skill 负责「这类流程应该怎么做」（静态可复用的 SOP 知识）。两者正交：planner 可以把「按退货流程处理」规划为一步，executor 执行时通过 `load_skill` 拿到完整 SOP 再逐步调工具——规划器不需要内置任何业务流程知识，新增流程只需加一个 Markdown 文件。

### Interrupt / Resume

- 两类人机中断分离：**confirm**（高风险操作审批，执行前本地可知，executor 内部 interrupt）与 **clarify**（缺信息追问，独立节点中继）。
- Resume 重放陷阱：LangGraph interrupt 恢复后会从头重放整个节点，导致重复 LLM 调用与非确定性；修复方式是在 NodeDeps 里加 `(thread_id, plan_iteration, idx, step_results 数, 澄清答案)` 键的缓存。
- 状态经 OpenAI SDK 调用前必须先 `convert_to_openai_messages`——LangGraph 的 `add_messages` reducer 会把 dict 转成 Message 对象，直接透传会 422；此契约由离线单测的报文断言防回归。

### Dense + Sparse Hybrid Retrieval

标准 RRF 融合在小库上实测回退（MRR@5 0.887→0.847）：RRF 加性融合抹平 dense 名次差，BM25 高噪声垃圾 chunk 凭双路加分反超 dense 头部。因此否决 RRF，采用「**dense 保底 + sparse 补位**」主从结构：

1. dense 前 top-5 为骨架，顺序不动；
2. 补位候选须同时满足「BM25 top-2 强命中」且「dense 排名在 (5, 10]」双重信号，至多替换骨架末位 1 个；
3. fetch_depth 之外的 chunk 不参与（垃圾防火墙）。

### Cross-Encoder Reranking

补位与精排分工互补：补位负责候选覆盖（recall@5 0.946→0.962），精排负责排序质量（MRR@5 0.887→0.968）。`Reranker` 封装 bge-reranker-base，进程内单例懒加载，仅 `hybrid_rerank` 模式加载 torch；精排只重排顺序不增删候选，返回分保留 dense 余弦分，同分按 chunk_id 排序保证确定性。

## 4. Evaluation

### RAG Retrieval Evaluation

32 条评测集（31 条可评分），top-5，`--no-judge` 纯检索口径，ChromaBackend + bge-small-zh-v1.5：

| 模式 | hit@5 | recall@5 | MRR@5 | 稳态延迟 |
| --- | --- | --- | --- | --- |
| dense（基线） | 1.0 | 0.946 | 0.887 | ~14ms/条 |
| hybrid（BM25 补位） | 1.0 | 0.962 | 0.887 | ~17ms/条 |
| hybrid_rerank（+精排） | 1.0 | 0.962 | **0.968** | ~700ms/条（CPU） |

复现：`python -m app.agent.rag.eval.run_eval --dataset app/agent/rag/eval/datasets/rag_eval.example.json --no-judge`（结果落盘 `app/agent/rag/eval/results/`）。

### Agent End-to-End Cases

- **RAG 端到端 A/B**（top-3，deepseek-v4-flash 生成 + 自写 LLM judge，31/31 成功）：answer_correctness 0.834→0.863（+3.0pt），MRR@3 0.887→0.984（+9.7pt）；judge 单条波动可达 ±0.6，结论以组均值为口径。
- **记忆压缩 harness**（deepseek-v4-flash，3 长脚本 × 4 轮）：final retention 96.3%、summary-reliant retention 95.8%、压缩比 2.64x；实证发现递归压缩漂移与 prompt 长度约束不可靠，驱动了 pinned facts 锚定 + 摘要硬截断（400 字）两个生产修复。
- **离线控制流单测**：100+ 用例（图路由 / interrupt-resume / replan / 补位契约 / 精排契约 / 退款状态机），`python -m unittest` 秒级回归，不触网。

### Known Limitations

- 会话存储为固定路径 `sessions/session.json`，单用户设计，无多用户隔离。
- 订单/退款数据为内存态 mock，进程重启后复位；评测与演示可用，非生产存储。
- 知识库仅 4 篇文档（<100 chunks）：此规模下 HNSW 与暴力扫描结果一致，chroma 的规模优势未体现；BM25 对跨文档同词干扰（如「破损」）仍可能引入噪声。
- CrossEncoder 精排在 Intel Mac CPU 上约 140ms/对，top-5 约 700ms；生产需 GPU 或 API 化 reranker 降延迟。
- LLM judge / 摘要均有跑次间方差，单次读数不可信，需多次取均值。

## 5. Quick Start & Demo

```bash
# 1. 环境（Python 3.11，项目使用仓库根目录 .venv）
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置（OpenAI 兼容接口；本地 embedding 模型首次运行自动下载）
cp .env.example .env
# 必填：OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME
# 推荐：PLAN_EXECUTE_ENABLED=true  RETRIEVAL_MODE=hybrid_rerank
#       EMBEDDING_BACKEND=local    RAG_BACKEND=chroma

# 3. 构建知识库索引（向量 + BM25）
python -m app.scripts.build_kb_index

# 4. 运行 CLI
python main.py
```

交互命令：`quit/exit` 退出 · `reset` 重置 · `memory` 查看记忆 · `skills` 查看技能。

试一句多步骤工单（Plan-and-Execute 模式）：

> 「帮我查一下订单 ORD-20240115-001 的物流，再看看退货政策里这种情况能不能退」

回归测试：`python -m unittest discover tests`（离线 mock，无需 API key）。
