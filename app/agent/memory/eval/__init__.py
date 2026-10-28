"""上下文压缩评估 harness（与 app/agent/rag/eval/ 平行的评测模块）。

评估对象：EcomAgent 的滚动摘要压缩（split_history + summarize）。
- factcheck.py：关键事实保留检查（确定性字符串匹配，离线零 LLM）
- runner.py：压缩模拟 runner（复用生产 split_history，摘要器可插拔）
- baselines.py：离线参考摘要器（不调 API，作为度量管线自检锚点）
- run_eval.py：CLI 入口，结果落 results/
- datasets/compression_eval.json：带埋点事实的多轮客服对话脚本
"""
