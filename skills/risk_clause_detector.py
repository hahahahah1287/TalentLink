# -*- coding: utf-8 -*-
"""
风险条款识别技能（结构化抽取 + 规则引擎）

技术范式：LLM 抽取式 NLP（口语合同 → 结构化字段） + 符号规则引擎（确定性判定）

为什么不靠 LLM 直接判：
    "试用期 6 个月 + 合同 1 年 = 违法" 这类判断有明确的法律阈值，是黑白分明的。
    交给 LLM 直接吐结论，结果不可复现、无法单测、可能幻觉。
    正确的分工：LLM 只做它擅长的——把"试用期半年""离职两年内不能同行"这种
    口语表达，抽取/归一化成结构化字段（probation_months=6, noncompete_years=2）；
    判定交给确定性规则引擎，每条结论都能追溯到触发的规则和法条。

流程：
    1. LLM 抽取：合同文本 → 结构化 ContractFacts（带 fallback 正则兜底）
    2. 规则引擎：对结构化字段逐条套用 RISK_RULES，输出可解释的风险项
    3. （可选）汇总：高/中/低风险计数
"""
import re
import json
from typing import Dict, Any, List, Optional

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from utils.skill_result import make_skill_result


# ==================== 规则引擎（确定性核心，可单测） ====================
#
# 每条规则是一个纯函数 facts -> Optional[risk_item]。
# 阈值来自《劳动法》，规则与判定逻辑分离，新增风险点只需加一条规则。

def _rule_probation(facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    试用期合法性：按合同期限分档判定上限（《劳动法》第21条 + 实务分档规则）。

    分档上限（合同期限 → 试用期上限月数）：
      [0, 3)   个月 : 不得约定试用期
      [3, 12)  个月 : ≤ 1 个月
      [12, 36) 个月 : ≤ 2 个月
      [36, ∞)        : ≤ 6 个月（法定绝对上限）
    """
    pm = facts.get("probation_months")
    cm = facts.get("contract_months")
    if pm is None:
        return None

    # 法定绝对上限：试用期最长 6 个月
    if pm > 6:
        return {
            "risk_type": "试用期超过法定上限",
            "risk_level": "高",
            "detail": f"约定试用期 {pm} 个月，超过法定绝对上限 6 个月。",
            "legal_basis": "《劳动法》第21条",
            "suggestion": "试用期最长不得超过 6 个月，超出部分无效，应按转正工资补足。",
        }

    # 按合同期限分档判定
    if cm is not None:
        if cm < 3:
            limit = 0
        elif cm < 12:
            limit = 1
        elif cm < 36:
            limit = 2
        else:
            limit = 6

        if limit == 0 and pm > 0:
            return {
                "risk_type": "试用期约定违法（短期合同不得约定试用期）",
                "risk_level": "高",
                "detail": f"合同期限 {cm} 个月不满 3 个月，不得约定试用期，却约定 {pm} 个月。",
                "legal_basis": "《劳动法》第21条（试用期应与合同期限相称）",
                "suggestion": "三个月以内的短期合同不得约定试用期。",
            }
        if pm > limit:
            return {
                "risk_type": "试用期超过该期限合同的法定上限",
                "risk_level": "高",
                "detail": f"合同期限 {cm} 个月，试用期最长应为 {limit} 个月，却约定 {pm} 个月。",
                "legal_basis": "《劳动法》第21条（试用期应与合同期限相称）",
                "suggestion": f"{cm} 个月期限的合同试用期不得超过 {limit} 个月，超出部分无效。",
            }
    return None


def _rule_penalty(facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """违约金条款：除服务期/竞业限制外，约定由劳动者承担违约金属高风险"""
    amount = facts.get("penalty_amount")
    reason = facts.get("penalty_reason", "") or ""
    if amount is None and not facts.get("has_penalty"):
        return None
    # 合法的两种情形
    legit = any(k in reason for k in ["服务期", "培训", "竞业"])
    if not legit:
        return {
            "risk_type": "违法约定违约金",
            "risk_level": "高",
            "detail": f"约定劳动者承担违约金{('（' + str(amount) + '元）') if amount else ''}，"
                      f"但事由（{reason or '未注明'}）不属于服务期或竞业限制。",
            "legal_basis": "《劳动法》第21、22条（违约金仅限服务期/竞业限制情形）",
            "suggestion": "除服务期和竞业限制外，不得约定由劳动者承担违约金，该条款应删除。",
        }
    return None


def _rule_noncompete(facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """竞业限制：期限上限 2 年，且必须有补偿金"""
    years = facts.get("noncompete_years")
    has_comp = facts.get("noncompete_compensation")
    if years is None and has_comp is None:
        return None
    items = []
    if years is not None and years > 2:
        items.append(f"竞业限制期限 {years} 年超过法定上限 2 年")
    if has_comp is False:
        items.append("未约定竞业限制补偿金")
    if items:
        return {
            "risk_type": "竞业限制条款不合规",
            "risk_level": "中",
            "detail": "；".join(items) + "。",
            "legal_basis": "《劳动法》第22条（竞业限制需合理且有补偿）",
            "suggestion": "竞业限制期限不超过 2 年，且应按月支付经济补偿，否则条款可能无效。",
        }
    return None


def _rule_overtime_unpaid(facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """免除加班费 / 自愿加班不付费条款"""
    if facts.get("overtime_unpaid"):
        return {
            "risk_type": "免除加班费",
            "risk_level": "高",
            "detail": "约定不支付加班费或以'自愿加班'为由免除加班费。",
            "legal_basis": "《劳动法》第44条",
            "suggestion": "延长工作时间应依法支付不低于 150%/200%/300% 的加班费，约定无效。",
        }
    return None


def _rule_waive_social_insurance(facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """放弃社保条款"""
    if facts.get("waive_social_insurance"):
        return {
            "risk_type": "放弃社会保险",
            "risk_level": "高",
            "detail": "约定劳动者自愿放弃社会保险。",
            "legal_basis": "《劳动法》第72条",
            "suggestion": "依法参加社保是强制义务，放弃约定无效，用人单位仍须缴纳。",
        }
    return None


def _rule_arbitrary_termination(facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """用人单位单方随意解除条款"""
    if facts.get("arbitrary_termination"):
        return {
            "risk_type": "用人单位随意解除权",
            "risk_level": "高",
            "detail": "约定用人单位可随时/无理由解除劳动合同且不补偿。",
            "legal_basis": "《劳动法》第25、26、28条",
            "suggestion": "解除须符合法定情形并履行程序，该条款排除劳动者法定权利，无效。",
        }
    return None


def _rule_deposit(facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """收取押金/保证金条款"""
    if facts.get("requires_deposit"):
        amount = facts.get("deposit_amount")
        return {
            "risk_type": "违法收取押金/保证金",
            "risk_level": "中",
            "detail": f"要求劳动者缴纳押金/保证金{('（' + str(amount) + '元）') if amount else ''}。",
            "legal_basis": "劳动法律法规禁止向劳动者收取财物作为入职条件",
            "suggestion": "用人单位不得以任何名义收取押金、保证金，应退还。",
        }
    return None


# 规则注册表：新增风险点只需在此追加一个纯函数
RISK_RULES = [
    _rule_probation,
    _rule_penalty,
    _rule_noncompete,
    _rule_overtime_unpaid,
    _rule_waive_social_insurance,
    _rule_arbitrary_termination,
    _rule_deposit,
]


def evaluate_rules(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    对结构化合同事实运行全部规则，汇总风险（确定性，纯函数，可单测）。

    Args:
        facts: 结构化合同字段

    Returns:
        {risk_clauses: [...], summary: {high, medium, low, overall}}
    """
    risks: List[Dict[str, Any]] = []
    for rule in RISK_RULES:
        try:
            item = rule(facts)
            if item:
                risks.append(item)
        except Exception:
            continue

    high = sum(1 for r in risks if r["risk_level"] == "高")
    medium = sum(1 for r in risks if r["risk_level"] == "中")
    low = sum(1 for r in risks if r["risk_level"] == "低")

    if high:
        overall = f"存在 {high} 项高风险条款，建议重点修改。"
    elif medium:
        overall = f"存在 {medium} 项中风险条款，建议关注。"
    elif risks:
        overall = "存在低风险条款。"
    else:
        overall = "未识别到明显风险条款（基于已抽取字段）。"

    return {
        "risk_clauses": risks,
        "summary": {"high": high, "medium": medium, "low": low, "overall": overall},
    }


# ==================== 正则兜底抽取（LLM 不可用时的 fallback） ====================

_CN_NUM = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
           '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}


def _parse_num(s: str) -> Optional[int]:
    """把'6'/'六'/'十二'解析为整数"""
    s = s.strip()
    if s.isdigit():
        return int(s)
    if not s:
        return None
    # 简单中文数字
    if s in _CN_NUM:
        return _CN_NUM[s]
    if s.startswith("十"):
        return 10 + (_CN_NUM.get(s[1:], 0) if len(s) > 1 else 0)
    if "十" in s:
        parts = s.split("十")
        tens = _CN_NUM.get(parts[0], 1)
        ones = _CN_NUM.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    return None


def regex_extract(text: str) -> Dict[str, Any]:
    """
    正则兜底抽取结构化字段（LLM 不可用或解析失败时使用）。

    覆盖最常见的几类风险表述，确保规则引擎在无 LLM 时仍可工作。
    """
    facts: Dict[str, Any] = {}

    m = re.search(r"试用期[为是]?\s*([\d一二三四五六七八九十两]+)\s*个月", text)
    if m:
        facts["probation_months"] = _parse_num(m.group(1))
    m = re.search(r"(?:劳动)?合同(?:期限)?[为是]?\s*([\d一二三四五六七八九十两]+)\s*年", text)
    if m:
        yrs = _parse_num(m.group(1))
        if yrs:
            facts["contract_months"] = yrs * 12
    m = re.search(r"合同(?:期限)?[为是]?\s*([\d一二三四五六七八九十两]+)\s*个月", text)
    if m:
        facts["contract_months"] = _parse_num(m.group(1))

    m = re.search(r"违约金\s*([\d.]+)\s*万", text)
    if m:
        facts["penalty_amount"] = int(float(m.group(1)) * 10000)
        facts["has_penalty"] = True
    elif re.search(r"违约金", text):
        facts["has_penalty"] = True
    if facts.get("has_penalty"):
        # 抽取违约金事由上下文
        ctx = text[max(0, text.find("违约金") - 20): text.find("违约金") + 20]
        facts["penalty_reason"] = ctx

    m = re.search(r"(?:离职后|竞业(?:限制)?)\s*([\d一二三四五六七八九十两]+)\s*年", text)
    if m and ("竞业" in text or "同行" in text or "同业" in text):
        facts["noncompete_years"] = _parse_num(m.group(1))
    if "竞业" in text:
        facts["noncompete_compensation"] = bool(re.search(r"补偿", text))

    if re.search(r"不支付加班费|自愿加班|放弃加班费|无加班费", text):
        facts["overtime_unpaid"] = True
    if re.search(r"放弃社[会保险]|自愿(?:放弃|不缴)社保|不缴纳社[会保险]", text):
        facts["waive_social_insurance"] = True
    if re.search(r"随时(?:解除|辞退)|任何(?:时候|理由)解除|单方(?:无条件)?解除|无需(?:支付)?(?:经济)?补偿", text):
        facts["arbitrary_termination"] = True
    m = re.search(r"押金\s*([\d.]+)\s*元|保证金\s*([\d.]+)\s*元", text)
    if m:
        facts["requires_deposit"] = True
        facts["deposit_amount"] = int(float(m.group(1) or m.group(2)))
    elif re.search(r"押金|保证金", text):
        facts["requires_deposit"] = True

    return facts


# ==================== LLM 抽取链（抽取式 NLP，不做判定） ====================

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是合同信息抽取器。你的唯一任务是从合同文本中**抽取结构化字段**，不要做合法性判断。

抽取以下字段（不存在则省略该字段，不要编造）：
- probation_months: 试用期月数（整数）
- contract_months: 劳动合同期限（换算成月，整数）
- has_penalty: 是否约定劳动者承担违约金（布尔）
- penalty_amount: 违约金金额（元，整数）
- penalty_reason: 违约金事由原文（字符串，如"提前离职""服务期"）
- noncompete_years: 竞业限制年限（整数）
- noncompete_compensation: 是否约定竞业补偿金（布尔）
- overtime_unpaid: 是否不支付加班费/自愿加班不付费（布尔）
- waive_social_insurance: 是否约定放弃社保（布尔）
- arbitrary_termination: 用人单位是否可随意/无理由解除（布尔）
- requires_deposit: 是否收取押金/保证金（布尔）
- deposit_amount: 押金金额（元，整数）

只输出 JSON，不要解释。"""),
    ("user", "合同文本：\n{contract_text}\n\n请抽取 JSON：")
])


def create_extract_chain(llm):
    """创建 LLM 抽取链（合同 → 结构化字段）"""
    return EXTRACT_PROMPT | llm | JsonOutputParser()


# ==================== @tool / 统一接口 ====================

_llm = None
_extract_chain = None


def init_skill(llm):
    """初始化抽取链（由 WorkflowService 调用）"""
    global _llm, _extract_chain
    _llm = llm
    _extract_chain = create_extract_chain(llm)


def _extract_facts_with_provenance(contract_text: str) -> tuple:
    """
    抽取结构化字段，并记录每个字段来自 LLM / regex / 冲突合并的 provenance。
    """
    llm_facts: Dict[str, Any] = {}
    regex_facts = regex_extract(contract_text)
    errors: List[str] = []
    if _extract_chain is not None:
        try:
            extracted = _extract_chain.invoke({"contract_text": contract_text})
            if isinstance(extracted, dict):
                llm_facts = {k: v for k, v in extracted.items() if v is not None}
        except Exception as e:
            errors.append(str(e))
            print(f"⚠️ [RiskClause] LLM 抽取失败，回退正则: {e}")

    facts: Dict[str, Any] = {}
    field_provenance: Dict[str, Any] = {}
    for key in sorted(set(llm_facts) | set(regex_facts)):
        llm_has = key in llm_facts
        regex_has = key in regex_facts
        conflict = llm_has and regex_has and llm_facts[key] != regex_facts[key]
        if regex_has:
            # 确定性规则抽取优先，避免 LLM 和明确文本冲突时吞掉硬证据。
            facts[key] = regex_facts[key]
            source = "llm+regex" if llm_has and not conflict else "regex"
            confidence = 0.95 if not conflict else 0.8
        else:
            facts[key] = llm_facts[key]
            source = "llm"
            confidence = 0.7
        field_provenance[key] = {
            "source": source,
            "confidence": confidence,
            "llm_value": llm_facts.get(key),
            "regex_value": regex_facts.get(key),
            "conflict": conflict,
        }

    provenance = {
        "extractor": "llm+regex",
        "field_provenance": field_provenance,
        "llm_available": _extract_chain is not None,
        "errors": errors,
    }
    return facts, provenance


def _extract_facts(contract_text: str) -> Dict[str, Any]:
    """Backward-compatible facts-only extractor."""
    facts, _ = _extract_facts_with_provenance(contract_text)
    return facts


@tool
def risk_clause_detector(contract_text: str, law_context: str = "") -> str:
    """风险条款识别工具。输入合同文本，识别其中的高风险条款并给出法律依据。

    采用"LLM 抽取结构化字段 + 规则引擎确定性判定"，结论可追溯到具体规则和法条。

    Args:
        contract_text: 合同文本内容
        law_context: 法律依据（可选，当前规则内置法条，主要用于补充说明）

    Returns:
        JSON 格式的风险分析结果（含每项风险的法律依据与修改建议）
    """
    if not contract_text or not contract_text.strip():
        return json.dumps({"error": "合同文本为空"}, ensure_ascii=False)

    facts, provenance = _extract_facts_with_provenance(contract_text)
    result = evaluate_rules(facts)
    result["extracted_facts"] = facts  # 暴露抽取结果，便于调试/可解释
    result["extraction_provenance"] = provenance
    findings = [
        {**item, "finding_type": "risk_clause"}
        for item in result.get("risk_clauses", [])
    ]
    unified = make_skill_result(
        skill_name="risk_clause_detector",
        facts=facts,
        findings=findings,
        evidence=[
            {"source_kind": "skill_basis", "legal_basis": item.get("legal_basis")}
            for item in result.get("risk_clauses", [])
            if item.get("legal_basis")
        ],
        provenance={"extraction": provenance},
        display_text=result.get("summary", {}).get("overall", ""),
        metrics=result.get("summary", {}),
        legacy=result,
    )
    return json.dumps(unified, ensure_ascii=False, indent=2)


def risk_clause_skill(query: str, law_context: str = "") -> str:
    """统一接口：接受 query（合同文本）"""
    return risk_clause_detector.invoke({"contract_text": query, "law_context": law_context})


# 导出
RISK_CLAUSE_TOOLS = [risk_clause_detector]
