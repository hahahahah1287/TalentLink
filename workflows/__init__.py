# -*- coding: utf-8 -*-
"""
LangGraph Workflows

各业务场景的状态图定义。
"""
from .contract_review import create_contract_review_graph
from .research_agent import create_research_agent_graph

__all__ = [
    "create_contract_review_graph",
    "create_research_agent_graph",
]
