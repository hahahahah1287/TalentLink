# -*- coding: utf-8 -*-
"""
Skills 技能包

导出所有可用的 Agent 工具/技能。
"""
from .web_search import web_search, web_search_skill, WEB_SEARCH_TOOLS
from .risk_clause_detector import risk_clause_detector, risk_clause_skill, RISK_CLAUSE_TOOLS
from .compliance_check import compliance_check, compliance_skill, COMPLIANCE_CHECK_TOOLS
from .legal_term_explainer import legal_term_explainer, legal_term_skill, LEGAL_TERM_TOOLS
from .statute_checker import statute_checker, statute_skill, STATUTE_CHECKER_TOOLS
from .case_retriever import case_retriever, case_retriever_skill, CASE_RETRIEVER_TOOLS
from .registry import SkillRegistry

# 所有 LangChain @tool 列表
ALL_TOOLS = (
    WEB_SEARCH_TOOLS + RISK_CLAUSE_TOOLS +
    COMPLIANCE_CHECK_TOOLS + LEGAL_TERM_TOOLS +
    STATUTE_CHECKER_TOOLS + CASE_RETRIEVER_TOOLS
)

__all__ = [
    # 搜索（可选补充能力）
    "web_search",
    # 法律技能（LangChain @tool 版本）
    "risk_clause_detector", "compliance_check",
    "legal_term_explainer", "statute_checker", "case_retriever",
    # 统一接口版本
    "web_search_skill",
    "risk_clause_skill", "compliance_skill",
    "legal_term_skill", "statute_skill", "case_retriever_skill",
    # 工具列表
    "WEB_SEARCH_TOOLS", "RISK_CLAUSE_TOOLS",
    "COMPLIANCE_CHECK_TOOLS", "LEGAL_TERM_TOOLS",
    "STATUTE_CHECKER_TOOLS", "CASE_RETRIEVER_TOOLS", "ALL_TOOLS",
    # 注册表
    "SkillRegistry",
]
