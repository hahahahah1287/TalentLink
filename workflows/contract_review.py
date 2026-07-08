# -*- coding: utf-8 -*-
"""
合同审查 LangGraph Workflow

固定流程，不需要 Planner 动态规划：
  retrieve → analyze → review (Review Agent)

为什么不用 Planner：
  合同审查的步骤是固定的（检索法律 → 对比分析 → 输出），
  让 9B 模型去规划"先检索再分析"是浪费 token。
"""
import time
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from utils.state import AppState
from utils.retrieval_service import RetrievalService
from utils.tools.contract_tools import create_contract_chain
from workflows.review_agent import create_review_agent_node, create_remove_citations_node, check_review_result
from workflows.shared_nodes import RETRY_PROMPT_SUFFIX


def make_retrieve_node(retrieval_service: RetrievalService):
    """创建检索节点"""

    def retrieve(state: AppState) -> Dict[str, Any]:
        query = state["query"]
        print(f"🔍 [ContractReview:retrieve] 检索法律条文: {query[:50]}...")

        law_context = retrieval_service.retrieve_as_string(query)

        return {
            "law_context": law_context,
            "tool_history": state.get("tool_history", []) + [{
                "step": "retrieve",
                "tool": "legal_retrieval",
                "input": query,
                "output_len": len(law_context),
                "timestamp": time.time(),
            }],
        }

    return retrieve


def make_analyze_node(contract_chain):
    """创建分析节点

    首次运行时直接分析；Review Agent 发现问题重跑时，
    会把 review_issues 拼入 question，让 LLM 知道上次哪里有问题并修正。
    """

    async def analyze(state: AppState) -> Dict[str, Any]:
        review_issues = state.get("review_issues", [])
        is_retry = bool(review_issues)

        if is_retry:
            print(f"🔄 [ContractReview:analyze] Review Agent 反馈问题，重新分析...")
            print(f"   具体问题: {review_issues}")
        else:
            print(f"📝 [ContractReview:analyze] 合同分析中...")

        # 拼接 question：首次直接用 query，重试时附带错误输出 + 问题反馈
        question = state["query"]
        if is_retry:
            draft = state.get("draft_answer", "")
            issues_text = "\n".join(f"- {issue}" for issue in review_issues)
            question += (
                f"\n\n{RETRY_PROMPT_SUFFIX}"
                f"\n\n【Review Agent 发现的具体问题】\n{issues_text}"
                f"\n\n【上次错误输出（供参考，避免重复错误）】\n{draft[:500]}"
                f"\n\n请针对上述问题修正你的回答。"
            )

        result = await contract_chain.ainvoke({
            "history": state.get("history", ""),
            "law": state.get("law_context", "未检索到相关法律条文"),
            "contract": state.get("contract_text", ""),
            "question": question,
        })

        step_name = "analyze_retry" if is_retry else "analyze"
        return {
            "draft_answer": result,
            "final_answer": result,
            "review_issues": [],  # 消费完清空，避免循环触发
            "review_retry": state.get("review_retry", 0) + (1 if is_retry else 0),
            "tool_history": state.get("tool_history", []) + [{
                "step": step_name,
                "tool": "contract_chain",
                "input": question[:200],
                "output_len": len(result),
                "timestamp": time.time(),
            }],
        }

    return analyze



def create_contract_review_graph(
    llm,
    retrieval_service: RetrievalService,
    checkpointer=None,
):
    """
    创建合同审查 workflow 图

    流程：retrieve → analyze → review → END
    Review Agent 发现引用问题时：review → analyze（带上问题反馈）→ review → END

    Args:
        llm: LangChain LLM 实例
        retrieval_service: RetrievalService 实例
        checkpointer: LangGraph Checkpointer（用于断点恢复）

    Returns:
        编译好的 LangGraph graph
    """
    contract_chain = create_contract_chain(llm)

    retrieve_node = make_retrieve_node(retrieval_service)
    analyze_node = make_analyze_node(contract_chain)
    review_node = create_review_agent_node(llm)
    remove_node = create_remove_citations_node(llm)

    workflow = StateGraph(AppState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("review", review_node)
    workflow.add_node("remove_citations", remove_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "review")
    workflow.add_conditional_edges(
        "review",
        check_review_result,
        {
            "revise": "analyze",
            "remove": "remove_citations",
            "approve": END,
        },
    )
    workflow.add_edge("remove_citations", END)

    return workflow.compile(checkpointer=checkpointer)
