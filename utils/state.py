# -*- coding: utf-8 -*-
"""
LangGraph 状态定义

定义所有 workflow 共享的 AppState 结构。
LangGraph 的核心是状态——每一步的输入输出都显式写入 state，
便于调试、审计、checkpoint 和失败重试。
"""
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document


class AppState(TypedDict, total=False):
    """
    统一应用状态

    所有 LangGraph workflow 共享同一个 state 结构。
    total=False 意味着所有字段都是可选的，各 workflow 按需读写。

    字段分组：
    - 请求信息：用户输入、场景、合同文本
    - 对话历史：从 ChatHistoryManager 注入
    - 流程控制：Planner 生成的计划、当前步骤
    - 检索结果：本地检索文档、联网搜索结果
    - 工具记录：每一步工具调用的输入输出
    - 输出：草稿、最终答案、引用
    - 合规：策略检查标记
    """

    # === 请求信息 ===
    conversation_id: str
    user_id: str
    scene: str              # "contract" | "job" | "chat"
    query: str
    contract_text: Optional[str]  # 合同文本（仅合同审查场景）

    # === 对话历史（从 ChatHistoryManager 注入） ===
    history: str

    # === 流程控制 ===
    plan: List[str]         # Planner 生成的步骤列表
    current_step: int       # 当前执行到第几步

    # === 检索结果 ===
    retrieved_docs: List[Document]    # 本地知识库检索结果
    law_context: str                  # 拼接后的法律条文文本
    search_results: List[str]         # 联网搜索结果

    # === 工具调用记录 ===
    tool_history: List[Dict[str, Any]]  # [{step, tool, input, output, duration}]

    # === 输出 ===
    draft_answer: str
    final_answer: str
    citations: List[str]

    # === 合规 ===
    policy_flags: List[str]
    status: str             # "running" | "done" | "error"
