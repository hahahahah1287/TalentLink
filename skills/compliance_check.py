# -*- coding: utf-8 -*-
"""
劳动法合规检查技能（决策表 / Rules-as-Data）

技术范式：规则即数据（外部化决策表） + 通用规则引擎

为什么不靠 LLM 直接判：
    加班 36 小时、试用期工资 80%、社保强制缴纳……这些是黑白分明的法律阈值。
    把它们写进 LLM 的 prompt 里，等于把"规则"耦合进"模型"——无法单测、
    无法度量覆盖、改一个阈值要改 prompt。
    正确做法：规则沉淀为外部 JSON 决策表（skills/data/compliance_rules.json），
    引擎按 (field, op, threshold) 做确定性匹配；LLM 只负责把用工场景描述
    抽取/归一化成结构化字段。规则可热更新、可单独维护、可计算命中覆盖率。

流程：
    1. LLM 抽取：场景描述 → 结构化 ScenarioFacts（正则兜底）
    2. 规则引擎：逐条套用决策表，输出命中的违规项（可追溯 rule_id + 法条）
    3. 汇总合规状态
"""
import os
import re
import json
from typing import Dict, Any, List, Optional

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from utils.skill_result import make_skill_result


# ==================== 决策表加载 ====================

_RULES_PATH = os.path.join(os.path.dirname(__file__), "data", "compliance_rules.json")


def load_rules() -> List[Dict[str, Any]]:
    """加载外部化决策表"""
    try:
        with open(_RULES_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("rules", [])
    except Exception as e:
        print(f"⚠️ [Compliance] 决策表加载失败: {e}")
        return []


# ==================== 通用规则引擎（确定性核心，可单测） ====================

def _apply_op(value: Any, op: str, threshold: Any) -> bool:
    """
    通用比较算子。命中返回 True（即"触发违规判定"）。

    支持：gt/gte/lt/lte/eq/ne/in/true/false
    """
    if op == "true":
        return value is True
    if op == "false":
        return value is False
    if value is None:
        return False
    try:
        if op == "gt":
            return value > threshold
        if op == "gte":
            return value >= threshold
        if op == "lt":
            return value < threshold
        if op == "lte":
            return value <= threshold
        if op == "eq":
            return value == threshold
        if op == "ne":
            return value != threshold
        if op == "in":
            return value in threshold
    except TypeError:
        return False
    return False


def evaluate_compliance(facts: Dict[str, Any], rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    用决策表评估场景合规性（纯函数，可单测）。

    Args:
        facts: 结构化场景字段
        rules: 决策表规则列表

    Returns:
        {
          violations: [...命中的违规项],
          checked: [...实际评估了的规则维度],
          coverage: 评估覆盖率,
          overall_status: 总体结论,
        }
    """
    violations: List[Dict[str, Any]] = []
    checked_dims = set()
    evaluated_rules = 0

    for rule in rules:
        field = rule.get("field")
        # 前置条件：仅当条件字段为真时才评估
        cond = rule.get("condition")
        if cond and not facts.get(cond):
            continue
        # 字段缺失则跳过（无法评估），不计入已评估
        if field not in facts or facts.get(field) is None:
            continue

        evaluated_rules += 1
        checked_dims.add(rule.get("dimension"))

        if _apply_op(facts[field], rule["op"], rule.get("threshold")):
            violations.append({
                "rule_id": rule.get("id"),
                "dimension": rule.get("dimension"),
                "verdict": rule.get("verdict_violate"),
                "actual_value": facts[field],
                "threshold": rule.get("threshold"),
                "legal_basis": rule.get("legal_basis"),
                "suggestion": rule.get("suggestion"),
            })

    total_rules = len(rules)
    coverage = round(evaluated_rules / total_rules, 4) if total_rules else 0.0

    if violations:
        overall = f"存在违规：命中 {len(violations)} 条规则。"
    elif evaluated_rules:
        overall = "已评估项均合规。"
    else:
        overall = "未能从场景中抽取到可评估字段，建议补充用工细节。"

    return {
        "violations": violations,
        "checked_dimensions": sorted(d for d in checked_dims if d),
        "coverage": coverage,
        "evaluated_rules": evaluated_rules,
        "total_rules": total_rules,
        "overall_status": overall,
    }


# ==================== 正则兜底抽取 ====================

_CN_NUM = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
           '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}


def _parse_num(s: str) -> Optional[int]:
    s = (s or "").strip()
    if s.isdigit():
        return int(s)
    if s in _CN_NUM:
        return _CN_NUM[s]
    if s.startswith("十"):
        return 10 + (_CN_NUM.get(s[1:], 0) if len(s) > 1 else 0)
    if "十" in s:
        parts = s.split("十")
        return _CN_NUM.get(parts[0], 1) * 10 + (_CN_NUM.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0)
    return None


def regex_extract(text: str) -> Dict[str, Any]:
    """正则兜底抽取结构化场景字段"""
    facts: Dict[str, Any] = {}

    m = re.search(r"试用期[为是]?\s*([\d一二三四五六七八九十两]+)\s*个月", text)
    if m:
        facts["probation_months"] = _parse_num(m.group(1))
    m = re.search(r"试用期工资[为是]?(?:正式工资的)?\s*(\d+)\s*%", text)
    if m:
        facts["probation_wage_ratio"] = int(m.group(1)) / 100.0
    m = re.search(r"(?:每[天日])\s*加班\s*([\d一二三四五六七八九十两]+)\s*(?:个)?小时", text)
    if m:
        facts["daily_overtime_hours"] = _parse_num(m.group(1))
    m = re.search(r"每月\s*加班\s*([\d一二三四五六七八九十两]+)\s*(?:个)?小时", text)
    if m:
        facts["monthly_overtime_hours"] = _parse_num(m.group(1))

    if re.search(r"没有加班费|不[支给]付加班费|只(?:给)?调休|无加班费", text):
        facts["overtime_pay_provided"] = False
    if re.search(r"不[缴交]社保|不缴纳社[会保险]|放弃社[会保险]|没(?:有)?(?:上|交)社保", text):
        facts["social_insurance_paid"] = False
    if re.search(r"低于最低工资|不足最低工资", text):
        facts["below_minimum_wage"] = True

    return facts


# ==================== LLM 抽取链（抽取式，不做判定） ====================

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是用工场景信息抽取器。任务是从场景描述中**抽取结构化字段**，不要做合规判断。

抽取以下字段（不存在则省略，不要编造）：
- probation_months: 试用期月数（整数）
- probation_wage_ratio: 试用期工资占正式工资比例（小数，如 0.7 表示 70%）
- daily_overtime_hours: 每日加班小时数（整数）
- monthly_overtime_hours: 每月加班小时数（整数）
- overtime_pay_provided: 是否支付加班费（布尔；只给调休不付费=false）
- social_insurance_paid: 是否依法缴纳社保（布尔；放弃/不缴=false）
- below_minimum_wage: 工资是否低于最低工资标准（布尔）
- termination_type_no_fault: 是否属于无过失性解除（布尔）
- termination_notice_days: 解除前提前通知天数（整数）

只输出 JSON，不要解释。"""),
    ("user", "用工场景：\n{scenario}\n\n请抽取 JSON：")
])


def create_extract_chain(llm):
    """创建 LLM 抽取链（场景 → 结构化字段）"""
    return EXTRACT_PROMPT | llm | JsonOutputParser()


# ==================== @tool / 统一接口 ====================

_llm = None
_extract_chain = None
_rules: List[Dict[str, Any]] = []


def init_skill(llm):
    """初始化抽取链 + 加载决策表（由 WorkflowService 调用）"""
    global _llm, _extract_chain, _rules
    _llm = llm
    _extract_chain = create_extract_chain(llm)
    _rules = load_rules()


def _extract_facts_with_provenance(scenario: str) -> tuple:
    """抽取结构化字段，并记录 LLM/regex 来源、置信度与冲突。"""
    llm_facts: Dict[str, Any] = {}
    regex_facts = regex_extract(scenario)
    errors: List[str] = []
    if _extract_chain is not None:
        try:
            extracted = _extract_chain.invoke({"scenario": scenario})
            if isinstance(extracted, dict):
                llm_facts = {k: v for k, v in extracted.items() if v is not None}
        except Exception as e:
            errors.append(str(e))
            print(f"⚠️ [Compliance] LLM 抽取失败，回退正则: {e}")

    facts: Dict[str, Any] = {}
    field_provenance: Dict[str, Any] = {}
    for key in sorted(set(llm_facts) | set(regex_facts)):
        llm_has = key in llm_facts
        regex_has = key in regex_facts
        conflict = llm_has and regex_has and llm_facts[key] != regex_facts[key]
        if regex_has:
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


def _extract_facts(scenario: str) -> Dict[str, Any]:
    """Backward-compatible facts-only extractor."""
    facts, _ = _extract_facts_with_provenance(scenario)
    return facts


@tool
def compliance_check(scenario: str, law_context: str = "") -> str:
    """劳动法合规检查工具。输入用工场景描述，按决策表判定是否合规。

    采用"LLM 抽取字段 + 外部决策表规则引擎"，每条结论可追溯到 rule_id 与法条。

    Args:
        scenario: 用工场景描述，如"试用期6个月，工资80%，每天加班4小时无加班费"
        law_context: 法律依据（可选，规则内置法条）

    Returns:
        JSON 格式的合规检查结果（含命中规则、阈值、法律依据、整改建议、覆盖率）
    """
    if not scenario or not scenario.strip():
        return json.dumps({"error": "场景描述为空"}, ensure_ascii=False)

    rules = _rules or load_rules()
    facts, provenance = _extract_facts_with_provenance(scenario)
    result = evaluate_compliance(facts, rules)
    result["extracted_facts"] = facts
    result["extraction_provenance"] = provenance
    findings = [
        {**item, "finding_type": "compliance_violation"}
        for item in result.get("violations", [])
    ]
    unified = make_skill_result(
        skill_name="compliance_check",
        facts=facts,
        findings=findings,
        evidence=[
            {
                "source_kind": "skill_basis",
                "rule_id": item.get("rule_id"),
                "legal_basis": item.get("legal_basis"),
            }
            for item in result.get("violations", [])
            if item.get("legal_basis")
        ],
        provenance={"extraction": provenance},
        display_text=result.get("overall_status", ""),
        metrics={
            "coverage": result.get("coverage"),
            "evaluated_rules": result.get("evaluated_rules"),
            "total_rules": result.get("total_rules"),
            "violations": len(result.get("violations", [])),
        },
        legacy=result,
    )
    return json.dumps(unified, ensure_ascii=False, indent=2)


def compliance_skill(query: str, law_context: str = "") -> str:
    """统一接口：接受 query（场景描述）"""
    return compliance_check.invoke({"scenario": query, "law_context": law_context})


# 导出
COMPLIANCE_CHECK_TOOLS = [compliance_check]
