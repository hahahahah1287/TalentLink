# -*- coding: utf-8 -*-
"""
Skills 技能包

导出所有可用的 Agent 工具/技能。
"""

from .web_search import web_search, job_search, WEB_SEARCH_TOOLS
from .local_retriever import LocalRetrieverSkill, create_local_retriever_tool

__all__ = [
    # 联网搜索
    "web_search",
    "job_search",
    "WEB_SEARCH_TOOLS",
    # 本地检索
    "LocalRetrieverSkill",
    "create_local_retriever_tool",
]
