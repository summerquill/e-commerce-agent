# RAG 升级设计文档

> 目标：建立可量化评测基线 → 对比"朴素检索" vs "增强检索"效果。
> 范围：Phase 1 = 评测集搭建 + 评测管线；Phase 2 = Query Rewrite + Rerank 增强。

***

## 一、当前现状

### 知识库规模

| 文档         | H2 chunks | 总字数          |
| ---------- | --------- | ------------ |
| 退换货政策.md   | 4         | 约 1700 字     |
| 常见问题FAQ.md | 4         | 约 1500 字     |
| 会员权益.md    | 5         | 约 1400 字     |
| 配送说明.md    | 6         | 约 1200 字     |
| **合计**     | **19**    | **约 5800 字** |

> ⚠️ 仅 19 个 chunk，按"每 chunk 2-3 问题"直问只能产出 \~40-60 条。
> 目标 100-200 条 → 需四类问题策略叠加。

### 现有 RAG 链路

```
用户 query → OpenAI Embeddings → 余弦相似度 → Top-K chunks → 塞 prompt → LLM 生成
```

无 Query Rewriting、无 Reranking、无 Hybrid。检索层 + 生成层都没有独立评测。

### 现有评测框架（第 9 期）

已有 `app/evaluation/evaluator.py` + `metrics.py` + `sandbox.py`，但面向 **Agent 整体**：

- 过程指标：tool\_accuracy / tool\_efficiency / process\_soundness
- 结果指标：intent\_match / keyword\_coverage / answer\_quality / faithfulness

**不覆盖 RAG 检索层**（召回率、精确率），也没有独立的 RAG 评测集。

***

## 二、评测集设计

### 数据 Schema

每条评测用例是一个 JSON 对象，同时满足 Ragas 输入和自研评测器需求：

```json
{
  "id": "q_001",
  "type": "direct",           // direct | multihop | negative | paraphrase
  "query": "七天无理由退货的运费谁出？",
  "expected_chunk_ids": ["退换货政策#01-00"],   // 检索层召回判断 ground truth
  "grading_notes": [          // 生成层完整性判断点
    "七天无理由退货由顾客承担退货运费",
    "建议使用上门取件服务，运费 12 元起"
  ],
  "expected_answer": "七天无理由退货的运费由顾客承担。",  // 可选，Ragas faithfulness 用
  "difficulty": "easy"        // easy | medium | hard
}
```

### 四类问题策略

| 类型                      | 说明                      | 示例                                    | 预估数量                |
| ----------------------- | ----------------------- | ------------------------------------- | ------------------- |
| **direct（直问）**          | 单 chunk 直接可答，最简单基础题     | "七天无理由退货的适用范围？" → `退换货政策#01-00`       | 19 chunk × 2 = \~38 |
| **paraphrase（口语改写）**    | 同一事实用不同口语表达提问，测召回鲁棒性    | "我穿了几次的鞋还能退吗"（改写自"不影响二次销售"）           | \~30-40             |
| **multihop（多跳跨 chunk）** | 需要跨 2+ chunk 回答，测语义检索覆盖 | "钻石会员的退货流程和普通会员有什么不同？"（跨会员权益 + 退换货政策） | \~20-30             |
| **negative（知识库盲区）**     | 知识库不含此信息，期望检索返回空/低分     | "你们支持货到付款吗"（FAQ 明确说不支持，属于 negative）   | \~10-15             |
| **合计**                  | <br />                  | <br />                                | **\~100-120**       |

> negative 用例分两种：
>
> - **明确回答不支持**（如货到付款）：知识库有明确否定答案 → 期望检索命中 + LLM 正确回答
> - **知识库确实没有**（如保修期限）：知识库没提 → 期望检索返回空 + LLM 说"不清楚"

### 生成流程

```
1. LLM 读每个 chunk → 生成 2 个 direct 问题
2. LLM 读整个知识库 → 生成跨 chunk multihop 问题
3. LLM 对每个 direct 问题 → 生成 1 个口语改写 paraphrase
4. 人工过滤（用户审核 → 删除垃圾题 → 补充缺的）
5. 输出 JSON 到 app/evaluation/rag_dataset.json
```

生成脚本支持 `--dry-run` 预览、`--limit N` 限数量、`--negative 10` 指定 negative 数量。

***

## 三、评测指标与 Ragas 集成

### 分层指标映射

```
┌────────────────────────────────────────────────────────────┐
│                    RAG 评测指标矩阵                          │
├──────────────┬───────────────────┬─────────────────────────┤
│    层级       │     指标           │    实现方式              │
├──────────────┼───────────────────┼─────────────────────────┤
│ 检索层        │ Context Recall    │ Ragas context_recall    │
│              │ (要的信息找到了吗) │ 输入: query, retrieved   │
│              │                   │ contexts, ground_truth   │
│              │ Context Precision │ Ragas context_precision │
│              │ (召回的里多少有用) │ 同上                    │
├──────────────┼───────────────────┼─────────────────────────┤
│ 生成层        │ Faithfulness      │ Ragas faithfulness      │
│              │ (有没有幻觉)       │ 输入: query, answer,    │
│              │                   │ contexts                │
│              │ Answer Relevancy   │ Ragas answer_relevancy │
│              │ (答的是所问的吗)   │ 同上                    │
│              │ 完整性            │ 自研 keyword_coverage   │
│              │ (grading_notes    │ 复用 metrics.py 现有    │
│              │  覆盖度)           │ 逻辑                    │
└──────────────┴───────────────────┴─────────────────────────┘
```

### Ragas vs 自研评测边界

| 维度            | Ragas                                                     | 自研 metrics.py                                                |
| ------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| **指标类型**      | 检索层 recall/precision + 生成层 faithfulness/relevancy（RAG 专用） | Agent 整体 tool\_accuracy / intent\_match / process\_soundness |
| **输入依赖**      | `ground_truth` 文本（参考答案片段） + `retrieved_contexts`          | 沙箱运行 trace（tool\_observations）                               |
| **LLM judge** | 内置（可配置模型）                                                 | 内置 `judge_faithfulness` / `judge_answer_quality`             |
| **运行环境**      | 需要 Ragas 包 + 评测集 JSON                                     | 不需要额外包，与现有 evaluator 兼容                                      |
| **定位**        | **RAG 链路独立评测**（本阶段重点）                                     | Agent 端到端评测（第 9 期已有）                                         |

**决策**：本阶段 Ragas 只做 RAG 链路评测，自研框架继续做 Agent 端到端。**不混用**，避免职责模糊。

### Ragas 版本与 API 注意事项

- Ragas v0.3+ API 大幅变化（DiscreteMetric、@experiment 装饰器、CLI）
- 需要先验证：
  1. `context_recall` / `context_precision` 的输入 schema：是需要 `ground_truth_contexts`（正确 chunk 文本数组）还是 `grading_notes`（评分要点）
  2. `faithfulness` 的输入：需要 `answer` + `retrieved_contexts`（不需要 ground truth 文本）
  3. OpenAI 兼容 client 接法：项目用 `aihubmix.com/v1`，Ragas 的 `llm_factory` 是否支持自定义 base\_url

***

## 四、Phase 2：RAG 增强功能设计

### 两个增强点

#### 1. Query Rewriting

```
原始 query: "我想退货"
    ↓
[LLM Rewriter] 读最近 3 轮对话 + 原始 query
    ↓
改写后: "退换货政策 七天无理由 质量问题 时效 运费"
    ↓
向量化 → 检索
```

- **开关**：`settings.rag_query_rewrite_enabled = False`（默认关，做 baseline）
- **位置**：`app/agent/rag/rewriter.py`，`KnowledgeRetriever.search()` 前调用
- **Prompt**：给 LLM 的改写 prompt 要求"扩关键词、消歧义、补隐含意图"
- **返回 trace**：`search_knowledge` tool 返回里加 `trace.rewrite = {"original": "...", "rewritten": "..."}`

#### 2. LLM Reranking

```
粗检索: top-8 (扩大召回)
    ↓
[LLM Reranker] 对每个 chunk 打分 "是否能回答用户问题"
    ↓
重排序 → 取 top-3
```

- **开关**：`settings.rag_rerank_enabled = False`
- **位置**：`app/agent/rag/reranker.py`，`KnowledgeRetriever.search()` 后调用
- **Prompt**：要求 LLM 对 chunk 打分 0-1，"完全能回答 / 部分相关 / 无关"
- **返回 trace**：加 `trace.rerank = [{"id": "...", "score": ..., "llm_score": ...}]`
- **不引入 cross-encoder 模型**：用同一个 chat 模型，当前知识库小足够用

### 两者组合效果

```
Baseline:  query → embed → top-3 cosine
V1:        query → rewrite → embed → top-8 cosine → rerank → top-3
```

预期提升（基于业内经验值）：

- Query Rewrite：召回率 +15-20%
- LLM Rerank：精确率 +10-15%
- 两者叠加：总体 F1 +20% 左右

### Trace 可观测性

每次 `search_knowledge` 返回里增加：

```json
{
  "success": true,
  "results": [...],
  "trace": {
    "original_query": "我想退货",
    "rewrite": {
      "enabled": true,
      "rewritten_query": "退换货条件 时效 运费规则",
      "latency_ms": 85
    },
    "retrieval": {
      "backend": "numpy",
      "top_k": 8,
      "latency_ms": 3
    },
    "rerank": {
      "enabled": true,
      "input_count": 8,
      "output_count": 3,
      "latency_ms": 120
    }
  }
}
```

面试讲点："我把检索链路完全可观测化了，每一步都能对比开关前后的效果差异，而不是黑盒调参"。

***

## 五、实施路线

```
Phase 1（当前）：评测基础设施
├── 1. 调研 Ragas API（context_recall/precision/faithfulness/relevancy 的签名）
├── 2. 写 scripts/gen_rag_evalset.py（生成 100+ 条评测题）
├── 3. 人工过滤 → app/evaluation/rag_dataset.json
├── 4. 写 scripts/eval_rag.py（Ragas 跑评测，输出 baseline 指标）
└── 5. 记录 baseline 指标（如 recall@3 = 0.65, precision@3 = 0.72）

Phase 2：RAG 增强
├── 6. 实现 rewriter.py + settings 开关
├── 7. 实现 reranker.py + settings 开关
├── 8. search_knowledge tool 加 trace
├── 9. 重跑 eval_rag.py → 对比 baseline
└── 10. 写对比报告（开关前后各指标变化）
```

***

## 六、面试可讲点

| 问题                                   | 回答要点                                                                                                     |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| "你怎么知道 RAG 好不好？"                     | "我先搭了评测管线——自建 100+ 条评测题（4 类问题），用 Ragas 测检索层 recall/precision 和生成层 faithfulness/relevancy，开关增强功能前后直接对比数字" |
| "Query Rewrite 为什么有用？"               | "用户口语化 query 和知识库书面语有语义鸿沟——rewrite 把'我想退货'展开成'退换货条件 时效 运费规则'，召回率实测从 0.65 提到 0.82"                        |
| "为什么用 LLM Rerank 而不是 cross-encoder？" | "我这个知识库只有 19 个 chunk，cross-encoder 杀鸡用牛刀——同一 chat 模型足够，而且工程上零额外依赖，换大知识库再考虑"                              |
| "negative 用例有什么用？"                   | "测 Agent 的'不知道'能力——真 RAG 系统不能瞎编，negative 用例检验的是'检索空 + LLM 诚实回复'这条链路"                                     |
| "Ragas 和你自研的评测怎么分工？"                 | "Ragas 专做 RAG 链路（检索 + 生成），我自研的继续做 Agent 端到端（工具调用、意图识别、推理过程）——两层职责不重叠"                                    |

