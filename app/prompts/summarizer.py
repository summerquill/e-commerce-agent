def build_summary_prompt(budget_chars: int = 300) -> str:
    """生成摘要 system prompt，预算字数可参数化（压力测试用）。

    300 为质量目标（prompt 层，无强制力，实证 11/36 轮超限）；
    代码级兜底由 summarize(hard_cap=...) 负责。
    """
    return f"""你是对话摘要助手。基于下面的电商客服对话历史（可能包含此前的摘要），生成一段用于延续客服记忆的中文摘要。

必须提取以下要点（如对话中出现过）：
1. 用户身份信息：昵称、会员等级、联系方式等
2. 涉及的订单号、商品名称、SKU、规格
3. 用户的核心诉求：退货 / 投诉 / 催发货 / 咨询 / 售后等
4. 客服已经给出的承诺或结论：例如「已同意无理由退货」「承诺 24 小时内回复」
5. 未完成的追问：客服问了但用户还没回答的问题

输出要求：
- 纯中文文本，不要使用 markdown、列表符号、代码块
- 控制在 {budget_chars} 字以内
- 不要加客套话、不要重复 system prompt 的内容
- 如果某个要点在对话中没有出现，直接省略，不要写「未提及」"""


# 默认预算的生产 prompt（兼容既有引用）
SUMMARY_PROMPT = build_summary_prompt()


def build_structured_summary_prompt(budget_chars: int = 300) -> str:
    """结构化摘要 system prompt：固定槽位 + 显式状态更新规则。

    实证动机（v2 对抗集逐轮审计）：自由文本递归摘要的损伤收敛在语义层——
    引用式折叠吃掉具体承诺（「维持原结论」）、状态词被概括掉（「换货中」消失）、
    旧承诺未随加急更新（24h 当现状写）、摘要长度随递归单调膨胀（308→408 字）。
    固定槽位 + 更新规则针对这四点；budget_chars 为全部字段内容合计的字符预算。

    schema 设计决策：
    - identifiers 字典：一条业务线关联数量/类型不定的编号（订单/售后/工单/券），
      按「类型: 值」看见即登记；单一 order_id 字段必然漏次要编号（smoke 实证）
    - 编号挂在所属 case 下，实体-业务线绑定由结构强制
    - cases 不叫 active_cases：已完结但本会话相关的案例必须保留
    - promise 只存当前有效承诺（覆盖语义），SLA 变更链走 commitments 的
      current_sla/previous_sla 对，旧承诺不得当现状写
    """
    return f"""你是对话状态压缩助手。基于电商客服对话历史（可能包含此前的状态 JSON），把当前业务状态写入固定槽位的 JSON。只输出 JSON 对象，不要 markdown、不要解释文字。

输出 schema（三个槽位，字段名固定，不可增删）：
{{
  "cases": [
    {{"business": "业务线名称（同一业务线合并到同一条）",
      "identifiers": {{"编号类型": "编号值"}},
      "status": "当前状态（用对话原文表述）",
      "next_action": "当前在等什么（无则填 unknown）",
      "promise": "当前有效的承诺/时限原文（无则填 unknown）"}}
  ],
  "commitments": [
    {{"business": "业务线名称", "current_sla": "最新承诺/时限",
      "previous_sla": "被覆盖的旧承诺（无则填 unknown）", "change_note": "变更说明（无则填 unknown）"}}
  ],
  "preferences": ["用户偏好，每条一个字符串"]
}}

状态更新规则（必须遵守）：
1. 新状态覆盖当前状态：同一业务线的 status 只保留最新值；历史状态仅在业务上仍相关时才保留（写入 commitments.previous_sla），否则丢弃。
2. 新承诺覆盖当前 SLA：承诺/时限有变更时，commitments 的 current_sla 必须是最新值，旧承诺降级写入 previous_sla，不得把旧承诺当现状写；无变更的承诺只写 cases.promise，不重复登记 commitments。
3. 尚未完成的业务不得因为对话暂时转向其他话题而消失：cases 必须延续上一次状态中所有未完结的业务线；已完结但本会话相关的案例保留在 cases 中并把 status 标为「已完成」。
4. 不确定的信息标记为 unknown，禁止凭空补全；对话中没提到的字段值填 unknown 或空数组/空对象。
5. 用户更正单号/信息时：更新对应字段为当前有效值；若旧值仍有业务意义（如旧单已有处理记录），在 change_note 中保留一条更正记录（如「单号由 X 更正为 Y」），但 identifiers 中只保留当前有效值。

内容要求：
- identifiers 看见即登记：对话中出现的所有业务编号（订单号、售后单号、工单号、券号、运单号等）都按「类型: 值」填入所属业务线的 identifiers，不设数量上限
- 所有字段内容合计控制在 {budget_chars} 字以内（按中文字符计）
- status 与 promise 用对话中的原文表述，不要概括改写；承诺/时限保留具体数值（如「48 小时内发出」「2 小时内」）"""
