# -*- coding: utf-8 -*-
"""
风险条款识别技能（LLM 驱动）

接收合同文本 + 法律上下文，用 LLM 分析风险条款。
规则引擎保留为辅助预处理（拆分条款、初步筛选）。
"""
import re
import json
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==================== 规则引擎（辅助预处理） ====================

TRIAL_PERIOD_RULES = [
    (0, 3, 0),
    (3, 12, 1),
    (12, 36, 2),
    (36, 9999, 6),
]

RISK_KEYWORDS = [
    "试用期", "违约金", "赔偿金", "竞业限制", "竞业禁止",
    "不承担", "免除", "自行承担", "无需赔偿",
    "自愿加班", "不支付加班费", "放弃加班费",
    "押金", "保证金", "培训费",
    "随时解除", "单方解除", "无条件解除", "随时辞退",
    "扣工资", "罚款", "扣除工资",
]


def _split_clauses(contract_text: str) -> list:
    """将合同文本拆分为条款"""
    parts = re.split(r"(?=第[一二三四五六七八九十百千\d]+条)", contract_text)
    clauses = [p.strip() for p in parts if p.strip()]
    if len(clauses) <= 1:
        clauses = [p.strip() for p in contract_text.split("\n") if p.strip()]
    return clauses


def _pre_screen(contract_text: str) -> str:
    """规则引擎预筛选：识别可能的风险关键词，作为 LLM 分析的参考"""
    matched = []
    for kw in RISK_KEYWORDS:
        if kw in contract_text:
            matched.append(kw)
    if matched:
        return f"规则预筛选发现以下风险关键词：{', '.join(matched)}"
    return "规则预筛选未发现明显风险关键词"


# ==================== LLM 分析链 ====================

def create_risk_clause_chain(llm):
    """
    创建 LLM 驱动的风险分析链

    接收合同文本 + 法律上下文（由 ReAct Agent 检索提供），
    用 LLM 进行专业分析，而非纯规则匹配。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业法务分析师。根据提供的合同条款和法律依据，识别风险条款。

分析维度：
1. 试用期是否符合劳动合同期限规定
2. 违约金条款是否合法（仅服务期和竞业限制可约定）
3. 竞业限制是否合理（期限≤2年，需有补偿金）
4. 是否存在免除用人单位法定责任的条款
5. 加班费约定是否符合法定标准
6. 是否存在违法收取押金/保证金
7. 解除合同条款是否合法
8. 工资克扣条款是否合法

对每个风险条款输出：
- risk_type: 风险类型
- risk_level: 高/中/低
- clause_text: 关键条款原文（截取关键部分）
- legal_basis: 法律依据（必须来自提供的法律上下文，如无则注明"需进一步检索"）
- analysis: 分析说明
- suggestion: 修改建议

如果法律上下文不足，可在 legal_basis 中标注"需进一步检索"，但仍基于劳动法常识分析。"""),
        ("user", """【法律依据】
{law_context}

【规则预筛选】
{pre_screen}

【合同文本】
{contract_text}

请逐条分析合同风险，输出 JSON 格式：
{{"risk_clauses": [{{"risk_type": "...", "risk_level": "高/中/低", "clause_text": "...", "legal_basis": "...", "analysis": "...", "suggestion": "..."}}], "summary": {{"high": N, "medium": N, "low": N, "overall": "..."}}}}""")
    ])
    return prompt | llm | StrOutputParser()


# ==================== @tool（LangChain 工具，保持兼容） ====================

# 全局 LLM 引用（由 init_skill 初始化）
_llm = None
_chain = None


def init_skill(llm):
    """初始化 skill 的 LLM 链（由 WorkflowService 调用）"""
    global _llm, _chain
    _llm = llm
    _chain = create_risk_clause_chain(llm)


@tool
def risk_clause_detector(contract_text: str, law_context: str = "") -> str:
    """
    风险条款识别工具（LLM 驱动）。输入合同文本和可选的法律依据，分析风险条款。

    Args:
        contract_text: 合同文本内容
        law_context: 法律依据（由检索工具提供，可选）

    Returns:
        JSON 格式的风险分析结果
    """
    if not contract_text or not contract_text.strip():
        return json.dumps({"error": "合同文本为空"}, ensure_ascii=False)

    if _chain is None:
        return json.dumps({"error": "skill 未初始化，请先调用 init_skill(llm)"}, ensure_ascii=False)

    # 规则引擎预筛选
    pre_screen = _pre_screen(contract_text)

    # LLM 分析
    try:
        result = _chain.invoke({
            "law_context": law_context or "暂无具体法律条文，请基于劳动法常识分析。",
            "pre_screen": pre_screen,
            "contract_text": contract_text,
        })
        return result
    except Exception as e:
        return json.dumps({"error": f"LLM 分析失败: {str(e)}"}, ensure_ascii=False)


# ==================== 统一接口 ====================

def risk_clause_skill(query: str, law_context: str = "") -> str:
    """统一接口：接受 query（合同文本）和可选的法律上下文"""
    return risk_clause_detector.invoke({"contract_text": query, "law_context": law_context})


# 导出
RISK_CLAUSE_TOOLS = [risk_clause_detector]
