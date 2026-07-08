# -*- coding: utf-8 -*-
"""
工具层

合成链等供法务图节点调用的原子工具。
"""
from .contract_tools import (
    create_contract_chain,
    create_legal_qa_chain,
    create_synthesis_chain,
)

__all__ = [
    "create_contract_chain",
    "create_legal_qa_chain",
    "create_synthesis_chain",
]
