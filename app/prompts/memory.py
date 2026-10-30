"""记忆提取 Prompt：短期记忆 (STM) 和长期记忆 (LTM) 的事实抽取。

输出统一为结构化事实：
- content  : 事实正文
- category : identity / preference / behavior / issue / other
- fact_type: current（当前仍有效的状态）| historical（过去发生、可能已过时）
"""

STM_EXTRACTION_PROMPT = """你是对话分析助手。基于下面的电商客服对话片段，提取用户的关键信息和偏好。

已有的用户信息（如果有的话，避免重复，可以更新或补充）：
{existing_facts}

请从对话中提取以下类型的信息（如果对话中出现）：
1. 用户身份信息：姓名、会员等级、联系方式
2. 用户偏好：喜欢的颜色、风格、品牌、价格区间
3. 关注的商品或品类
4. 用户情绪/态度：满意、不满、焦虑、着急等
5. 特殊需求：送货时间、地址偏好、礼品包装等
6. 其他值得记住的信息

fact_type 判定规则：
- current：用户当前仍有效的状态或属性（姓名、会员等级、地址、长期偏好、
  正在进行且尚未结束的订单/物流/退款）
- historical：过去发生且已结束、或可能已经变化的事件
  （如「曾咨询过会员权益」「之前提交过一次退款申请」「上次买过某商品」）
- 不确定时一律填 historical，不要把过去事件写成当前状态

输出要求（只输出 JSON，不要 markdown 代码块、不要额外解释）：
{{"facts": [{{"content": "一句话描述", "category": "identity", "fact_type": "current"}}]}}

补充约束：
- 每条信息一条 fact，content 用简洁中文
- 只提取对话中明确提及或可合理推断的信息
- 不要把平台政策、会员权益规则等知识库内容当成用户记忆
- 如果对话中没有新的有用信息，输出 {{"facts": []}}"""


LTM_EXTRACTION_PROMPT = """你是用户画像分析助手。基于以下电商客服对话（可能包含对话摘要），提取值得长期记忆的用户信息。

这些信息将在未来的对话中帮助客服更好地服务该用户。

用户已有的长期记忆（避免重复）：
{existing_ltm}

请输出 JSON 格式，包含两个字段：
1. "facts": 值得长期记忆的用户信息数组，每条包含：
   - "content": 信息内容（简洁中文）
   - "category": 分类（identity/preference/behavior/issue/other）
   - "fact_type": "current"（用户当时仍有效的状态/属性）或
     "historical"（过去发生、可能已过时的行为或事件）；不确定时填 historical
2. "interaction_summary": 本次交互的一句话摘要

只提取用户个人信息与历史，不要把平台政策、会员权益规则等知识库内容写成用户记忆。

示例输出：
{{
  "facts": [
    {{"content": "用户是钻石会员", "category": "identity", "fact_type": "current"}},
    {{"content": "用户偏好红色系的运动鞋", "category": "preference", "fact_type": "current"}},
    {{"content": "用户此前咨询过会员权益，客服因系统故障未能答复", "category": "issue", "fact_type": "historical"}}
  ],
  "interaction_summary": "用户咨询了Nike运动鞋的退换货，最终申请了退款"
}}

如果对话中没有值得长期记忆的新信息，facts 返回空数组。
interaction_summary 必须填写。

只输出 JSON，不要加其他内容。"""
