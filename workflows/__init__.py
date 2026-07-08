# -*- coding: utf-8 -*-
"""
LangGraph Workflows

统一法务图（确定性 DAG）。
"""
from .legal_graph import build_legal_graph, SkillSpec

__all__ = [
    "build_legal_graph",
    "SkillSpec",
]
