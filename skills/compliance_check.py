# -*- coding: utf-8 -*-
"""
劳动法合规检查技能（LLM 驱动）

接收场景描述 + 法律上下文，用 LLM 判断合规性。
规则引擎保留为辅助预处理（场景识别、关键词匹配）。
"""
import re
import json
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==================== 规则引擎（辅助预处理） ====================

SCENARIO_KEYWORDS = {
    "试用期": ["试用期", "试用", "试工"],
    "加班": ["加班", "延长工作时间", "超时工作"],
    "解除合同": ["解除", "辞退", "开除", "终止", "裁员", "离职"],
    "社保": ["社保", "社会保险", "五险一金", "养老", "医疗", "失业保险", "工伤保险", "生育保险", "公积金"],
    "年假": ["年假", "年休假", "带薪休假", "休假"],
    "工资": ["工资", "薪资", "薪酬", "最低工资", "克扣", "拖欠"],
    "竞业限制": ["竞业限制", "竞业禁止", "竞业"],
    "工伤": ["工伤", "职业病", "工亡", "因工受伤", "因工死亡"],
}


def _match_scenarios(text: str) -> list:
    """快速识别涉及的场景"""
    matched = []
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matched.append(scenario)
                break
    return matched


def _pre_screen(scenario: str) -> str:
    """规则引擎预筛选：识别涉及的场景"""
    matched = _match_scenarios(scenario)
    if matched:
        return f"规则预筛选识别到以下场景：{', '.join(matched)}"
    return "规则预筛选未识别到具体场景，请基于劳动法常识分析。"


# ==================== LLM 分析链 ====================

def create_compliance_chain(llm):
    """
    创建 LLM 驱动的合规检查链

    接收场景描述 + 法律上下文（由 ReAct Agent 检索提供），
    用 LLM 进行专业合规判断，而非硬编码关键词匹配。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是劳动法合规审查专家。根据提供的法律依据，判断用工场景是否合法。

检查维度：
1. 试用期：时长、工资、是否重复约定
2. 加班：时长上限、加班费标准
3. 解除合同：程序、经济补偿、是否违法解除
4. 社保：是否依法缴纳
5. 年假：天数、补偿
6. 工资：最低工资、支付周期
7. 竞业限制：期限、补偿金
8. 工伤：认定时限、待遇

对每个检查项输出：
- check: 检查内容
- status: 合规/违规/需进一步确认
- legal_basis: 法律依据（必须来自提供的法律上下文，如无则注明"需进一步检索"）
- analysis: 分析说明
- suggestion: 整改建议（如违规）

如果法律上下文不足，可在 legal_basis 中标注"需进一步检索"，但仍基于劳动法常识判断。"""),
        ("user", """【法律依据】
{law_context}

【规则预筛选】
{pre_screen}

【用工场景】
{scenario}

请逐项检查合规性，输出 JSON 格式：
{{"details": [{{"check": "...", "status": "合规/违规/需进一步确认", "legal_basis": "...", "analysis": "...", "suggestion": "..."}}], "overall_status": "合规/存在违规/需进一步确认", "summary": "..."}}""")
    ])
    return prompt | llm | StrOutputParser()


# ==================== @tool（LangChain 工具，保持兼容） ====================

_llm = None
_chain = None


def init_skill(llm):
    """初始化 skill 的 LLM 链（由 WorkflowService 调用）"""
    global _llm, _chain
    _llm = llm
    _chain = create_compliance_chain(llm)


@tool
def compliance_check(scenario: str, law_context: str = "") -> str:
    """
    劳动法合规检查工具（LLM 驱动）。输入用工场景描述和可选的法律依据。

    Args:
        scenario: 用工场景描述，如"试用期6个月，工资80%"
        law_context: 法律依据（由检索工具提供，可选）

    Returns:
        JSON 格式的合规检查结果
    """
    if not scenario or not scenario.strip():
        return json.dumps({"error": "场景描述为空"}, ensure_ascii=False)

    if _chain is None:
        return json.dumps({"error": "skill 未初始化，请先调用 init_skill(llm)"}, ensure_ascii=False)

    # 规则引擎预筛选
    pre_screen = _pre_screen(scenario)

    # LLM 分析
    try:
        result = _chain.invoke({
            "law_context": law_context or "暂无具体法律条文，请基于劳动法常识判断。",
            "pre_screen": pre_screen,
            "scenario": scenario,
        })
        return result
    except Exception as e:
        return json.dumps({"error": f"LLM 分析失败: {str(e)}"}, ensure_ascii=False)


# ==================== 统一接口 ====================

def compliance_skill(query: str, law_context: str = "") -> str:
    """统一接口：接受 query（场景描述）和可选的法律上下文"""
    return compliance_check.invoke({"scenario": query, "law_context": law_context})


# 导出
COMPLIANCE_CHECK_TOOLS = [compliance_check]
