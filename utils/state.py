# -*- coding: utf-8 -*-
"""
LangGraph 状态定义

定义所有 workflow 共享的 AppState 结构。
LangGraph 的核心是状态——每一步的输入输出都显式写入 state，
便于调试、审计、checkpoint 和失败重试。
"""
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langchain_core.documents import Document


class AppState(TypedDict, total=False):
    """
    统一应用状态

    确定性法务图的共享 state。
    total=False 意味着所有字段都是可选的，各节点按需读写。

    字段分组：
    - 请求信息：用户输入、场景、合同文本
    - 对话历史：从 ChatHistoryManager 注入
    - 意图路由：确定性关键词路由的输出（has_contract / route_skills）
    - 检索结果：本地法条检索结果（检索固化为首节点）
    - Skill 输出：确定性 skill 的结构化结果
    - 工具记录：每一步的输入输出（审计）
    - 输出：草稿、最终答案
    - Guard：确定性输出防护的反馈（引用问题触发一次 revise）
    """

    # === 请求信息 ===
    conversation_id: str
    user_id: str
    scene: Literal["legal", "chat"]  # "legal" | "chat"（确定性法务图统一为 "legal"）
    query: str
    contract_text: Optional[str]  # 合同文本（可选；合同判定以内容特征为准，不依赖此字段）

    # === 对话历史（从 ChatHistoryManager 注入） ===
    history: str

    # === 意图路由（确定性关键词路由的输出） ===
    has_contract: bool      # 是否判定为合同审查场景（内容特征）
    route_skills: List[str] # 检索后、合成前要执行的确定性 skill 列表

    # === 检索结果 ===
    retrieved_docs: List[Document]    # 本地知识库检索结果
    evidence_items: List[Dict[str, Any]]  # article-level 证据对象（可追溯引用）
    law_context: str                  # 证据渲染后的法律条文文本

    # === 工具调用记录 ===
    tool_history: List[Dict[str, Any]]  # [{step, tool, input, output, duration}]

    # === 输出 ===
    draft_answer: str
    final_answer: str

    # === Skill 结构化输出 ===
    skill_outputs: Dict[str, str]  # skill_name -> result（保留结构化数据）

    # === 流程控制 ===
    status: str             # "running" | "done" | "error"

    # === Guard（确定性输出防护） ===
    guard_issues: List[str]  # 引用验证发现的问题（触发一次 revise 重生成）
    guard_retry: int         # 已重生成次数（最多 1 次）
