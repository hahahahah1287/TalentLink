# -*- coding: utf-8 -*-
"""
确定性意图路由 v2（加权多信号评分 + 同义词扩展 + 可观测性）

从 v1 的布尔关键词匹配升级为：
1. 加权多信号评分：每个关键词有独立权重，skill 得分 = Σ(命中词权重)
2. 同义词/泛化词典：覆盖口语化表达（"还能不能告"→statute_checker）
3. 路由可观测性：结构化日志（得分、命中词、决策路径）便于调试与迭代

设计不变项：
- 纯函数、零模型、可单测、可复现
- 检索固化为首节点（不在路由决策内），路由只决定额外 skill
- 合同识别基于后端内容特征打分，不依赖前端字段
- RouteResult 结构向后兼容（已有 25+ 测试断言依赖）
"""
import re
import logging
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)

ROUTE_VERSION = "deterministic-router-v3"


# ==================== 加权关键词表（v2 核心升级） ====================
#
# 每个 skill 一组 (关键词, 权重) 对。query 中每命中一个词累加其权重，
# 超过 skill 触发阈值即进入执行列表。权重体系：
#   3.0 = 强信号（该词几乎只对应此 skill）
#   2.0 = 中信号（常见但非独占）
#   1.0 = 弱信号（泛化/口语化表达，需叠加才触发）

# (关键词, 权重)
SKILL_WEIGHTED_KEYWORDS: Dict[str, List[tuple]] = {
    "statute_checker": [
        # 强信号
        ("仲裁时效", 3.0), ("诉讼时效", 3.0), ("时效", 2.5),
        ("期限届满", 2.5), ("还能仲裁", 3.0), ("还能起诉", 3.0),
        # 中信号
        ("过期", 2.0), ("过没过", 2.0), ("多久内", 2.0),
        ("几年内", 2.0), ("超过时间", 2.0), ("多长时间内", 2.0),
        ("仲裁", 1.5), ("申请仲裁", 2.0),
        # 弱信号（同义词/口语扩展）
        ("还能不能", 1.5), ("来得及", 1.5), ("还能申请", 1.5),
        ("来不来得及", 1.5), ("错过了吗", 1.5), ("过了时间", 1.5),
        ("还有效吗", 1.5), ("还能告吗", 1.5), ("赶不赶趟", 1.0),
    ],
    "legal_term_explainer": [
        # 强信号（问"是什么"类）
        ("什么是", 3.0), ("是什么意思", 3.0), ("怎么理解", 2.5),
        ("含义", 2.5), ("解释", 2.0),
        # 中信号（对比/辨析类）
        ("区别", 2.5), ("有什么不同", 2.5), ("怎么算", 2.0),
        # 术语本身（弱信号，需叠加"是什么"类才触发）
        ("经济补偿", 1.5), ("赔偿金", 1.5), ("代通知金", 1.5),
        ("竞业限制", 1.5), ("服务期", 1.5),
        ("n+1", 2.0), ("2n", 2.0), ("双倍工资", 1.5), ("无固定期限", 1.5),
        # 同义词扩展
        ("啥意思", 2.5), ("啥区别", 2.5), ("怎么回事", 1.5),
        ("通俗讲", 2.0), ("用大白话", 2.0),
    ],
    "case_retriever": [
        # 强信号
        ("类似案例", 3.0), ("怎么判", 3.0), ("判例", 3.0),
        ("先例", 3.0), ("案例", 2.5),
        # 中信号
        ("类似", 2.0), ("案子", 2.0), ("胜诉", 2.0), ("败诉", 2.0),
        ("判过", 2.0), ("怎么处理的", 2.0),
        # 弱信号/同义词扩展
        ("别人", 1.0), ("这种情况", 1.5), ("类似情况", 2.0),
        ("以前", 1.0), ("有没有参考", 2.0), ("前车之鉴", 2.0),
        ("之前的案子", 2.5), ("别人碰到", 1.5), ("同样遭遇", 1.5),
    ],
    "compliance_check": [
        # 强信号（明确问合法性）
        ("合法吗", 3.0), ("违法吗", 3.0), ("合规", 2.5), ("违规", 2.5),
        ("合不合法", 3.0), ("违不违法", 3.0),
        # 中信号
        ("可以吗", 2.0), ("允许吗", 2.0), ("有没有问题", 2.0),
        ("符合规定", 2.0), ("加班费", 2.0), ("社保", 2.0),
        ("试用期工资", 2.0), ("最低工资", 2.0), ("克扣", 2.0),
        # 弱信号/同义词扩展
        ("这样行吗", 1.5), ("有问题吗", 1.5), ("算违法吗", 2.5),
        ("受法律保护", 1.5), ("合理吗", 1.5), ("正常吗", 1.0),
    ],
    "web_search": [
        # 只给最新/地方/政策类问题触发，不让主链路默认联网
        ("最新", 3.0), ("现行", 2.5), ("今年", 2.0), ("202", 1.5),
        ("地方政策", 3.0), ("地方标准", 3.0), ("当地", 2.0),
        ("最低工资标准", 3.0), ("社保基数", 3.0), ("公积金基数", 3.0),
        ("最新政策", 3.0), ("政策更新", 3.0), ("新规", 2.5),
    ],
}

# 各 skill 触发阈值（得分 >= 此值即触发）
SKILL_THRESHOLDS: Dict[str, float] = {
    "statute_checker": 2.0,
    "legal_term_explainer": 2.5,
    "case_retriever": 2.0,
    "compliance_check": 2.0,
    "web_search": 4.0,
}

# 向后兼容：旧 SKILL_KEYWORDS 格式（供已有导入不报错）
SKILL_KEYWORDS: Dict[str, List[str]] = {
    name: [kw for kw, _ in pairs]
    for name, pairs in SKILL_WEIGHTED_KEYWORDS.items()
}

# 法务兜底触发词（命中任意时，若无更具体 skill 则走 compliance_check）
LEGAL_FALLBACK_KEYWORDS: List[str] = [
    "劳动法", "劳动合同", "工资", "加班", "解除", "辞退", "开除", "裁员",
    "社保", "公积金", "工伤", "年假", "产假", "病假", "试用期",
    "经济补偿", "赔偿", "仲裁", "维权",
]


# ==================== 合同内容特征判定（不变，后端特征打分） ====================

# (正则, 权重, 说明)
_CONTRACT_FEATURES = [
    (re.compile(r"甲\s*方"), 2.0, "甲方"),
    (re.compile(r"乙\s*方"), 2.0, "乙方"),
    (re.compile(r"第[一二三四五六七八九十百\d]+条"), 2.0, "条款编号"),
    (re.compile(r"劳动合同期限|合同期限[为是]"), 1.5, "合同期限条款"),
    (re.compile(r"试用期[为是]"), 1.5, "试用期条款"),
    (re.compile(r"本合同|本协议"), 1.0, "合同指代"),
    (re.compile(r"双方(?:经)?(?:协商|约定|签订)"), 1.0, "签署语"),
    (re.compile(r"经济补偿|违约金|竞业限制"), 0.5, "合同常见条款词"),
    (re.compile(r"年\s*月\s*日|签订(?:日期|于)"), 0.5, "签署日期"),
]

_CONTRACT_SCORE_THRESHOLD = 4.0
_CONTRACT_MIN_LEN = 40


def contract_feature_score(text: str) -> float:
    """
    计算文本的"合同文书特征"得分（确定性，纯函数，可单测）。
    每类特征命中一次即计入其权重（不重复累加同一特征），返回总分。
    """
    if not text:
        return 0.0
    score = 0.0
    for pattern, weight, _label in _CONTRACT_FEATURES:
        if pattern.search(text):
            score += weight
    return score


def looks_like_contract(text: Optional[str]) -> bool:
    """基于内容特征判断一段文本是否是合同文书。"""
    if not text or len(text.strip()) < _CONTRACT_MIN_LEN:
        return False
    return contract_feature_score(text) >= _CONTRACT_SCORE_THRESHOLD


# ==================== 路由结果结构（向后兼容） ====================

class RouteResult(TypedDict):
    has_contract: bool
    skills: List[str]
    contract_source: str        # "content" | "field" | "both" | "none"
    matched: Dict[str, List[str]]  # 每个 skill 命中的关键词（调试）
    scores: Dict[str, float]    # v2 新增：每个 skill 的加权得分
    route_version: str
    confidence: float
    confidence_by_skill: Dict[str, float]
    trace: List[Dict[str, Any]]


# 合同场景固定挂载的 skill
_CONTRACT_SKILLS = ["risk_clause_detector", "compliance_check"]


# ==================== 加权评分核心（v2 升级） ====================

def _score_skills(query: str) -> Dict[str, tuple]:
    """
    对 query 做加权多信号评分。

    Returns:
        {skill_name: (score, [命中词列表])}
    """
    results: Dict[str, tuple] = {}
    for skill_name, weighted_kws in SKILL_WEIGHTED_KEYWORDS.items():
        score = 0.0
        hits: List[str] = []
        for kw, weight in weighted_kws:
            if kw in query:
                score += weight
                hits.append(kw)
        results[skill_name] = (score, hits)
    return results


def _match_keywords(query: str, keywords: List[str]) -> List[str]:
    """返回 query 中命中的关键词列表（去重保序，兼容旧逻辑）"""
    hits = []
    for kw in keywords:
        if kw in query:
            hits.append(kw)
    return hits


# ==================== 路由主函数 ====================

def route_intent(query: str, contract_text: Optional[str] = None) -> RouteResult:
    """
    确定性意图路由 v2：加权多信号评分 + 合同内容特征判定。

    升级点（相比 v1 布尔匹配）：
    1. 每个关键词有独立权重，累加得分超阈值才触发 skill
    2. 同义词/口语化表达覆盖更广（"来得及吗""算违法吗"等）
    3. 结构化 scores 字段供下游观测与调试

    判定逻辑不变：
      1. 合同识别：query 或 contract_text 内容特征超阈值 → has_contract
      2. skill 路由：合同场景固定挂 risk+compliance + 额外命中；
         非合同按加权得分选 skill；都没命中但有法务词 → 兜底 compliance
    """
    q = query or ""

    # --- 1. 合同识别（内容特征，不信前端声明） ---
    contract_in_field = looks_like_contract(contract_text)
    contract_in_query = looks_like_contract(q)
    has_contract = contract_in_field or contract_in_query
    if contract_in_field and contract_in_query:
        contract_source = "both"
    elif contract_in_field:
        contract_source = "field"
    elif contract_in_query:
        contract_source = "content"
    else:
        contract_source = "none"

    # --- 2. 加权多信号评分 ---
    skill_scores = _score_skills(q)

    matched: Dict[str, List[str]] = {}
    scores: Dict[str, float] = {}
    confidence_by_skill: Dict[str, float] = {}
    trace: List[Dict[str, Any]] = []
    skills: List[str] = []

    def _add(name: str):
        if name not in skills:
            skills.append(name)

    # 超过阈值的 skill
    triggered_skills: List[str] = []
    for name, (score, hits) in skill_scores.items():
        scores[name] = round(score, 2)
        if hits:
            matched[name] = hits
        threshold = SKILL_THRESHOLDS.get(name, 2.0)
        confidence_by_skill[name] = round(min(score / threshold, 1.0), 4) if threshold else 0.0
        trace.append({
            "stage": "skill_score",
            "skill": name,
            "score": round(score, 2),
            "threshold": threshold,
            "hits": hits,
            "triggered": score >= threshold,
        })
        if score >= threshold:
            triggered_skills.append(name)

    if has_contract:
        # 合同场景：固定挂风险条款 + 合规，再加 query 额外触发的 skill
        trace.append({
            "stage": "contract_detect",
            "source": contract_source,
            "query_score": contract_feature_score(q),
            "contract_text_score": contract_feature_score(contract_text or ""),
            "triggered": True,
        })
        for s in _CONTRACT_SKILLS:
            _add(s)
            confidence_by_skill[s] = max(confidence_by_skill.get(s, 0.0), 1.0)
        for s in triggered_skills:
            _add(s)
    else:
        # 非合同场景：按加权得分选 skill
        for s in triggered_skills:
            _add(s)
        # 兜底：没触发具体 skill 但确实是法务问题 → compliance_check
        fallback_hits = _match_keywords(q, LEGAL_FALLBACK_KEYWORDS)
        if not skills and fallback_hits:
            _add("compliance_check")
            confidence_by_skill["compliance_check"] = max(confidence_by_skill.get("compliance_check", 0.0), 0.55)
            trace.append({
                "stage": "legal_fallback",
                "skill": "compliance_check",
                "hits": fallback_hits,
                "triggered": True,
            })

    confidence = round(max([confidence_by_skill.get(s, 0.0) for s in skills] or [0.0]), 4)

    # --- 3. 可观测性日志 ---
    logger.debug(
        "route_intent decision",
        extra={
            "query_len": len(q),
            "has_contract": has_contract,
            "contract_source": contract_source,
            "scores": scores,
            "triggered": triggered_skills,
            "final_skills": skills,
        },
    )

    return RouteResult(
        has_contract=has_contract,
        skills=skills,
        contract_source=contract_source,
        matched=matched,
        scores=scores,
        route_version=ROUTE_VERSION,
        confidence=confidence,
        confidence_by_skill=confidence_by_skill,
        trace=trace,
    )
