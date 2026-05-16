# -*- coding: utf-8 -*-
"""
工具层

提供各业务场景的原子工具函数，供 LangGraph workflow 节点调用。
"""
from .contract_tools import create_contract_chain, create_synthesis_chain
from .job_tools import web_search, job_search
from .common_tools import policy_check, format_citations

__all__ = [
    # 合同分析
    "create_contract_chain",
    "create_synthesis_chain",
    # 求职搜索
    "web_search",
    "job_search",
    # 通用
    "policy_check",
    "format_citations",
]
