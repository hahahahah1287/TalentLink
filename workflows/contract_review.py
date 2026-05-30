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
from workflows.review_agent import create_review_agent_node, check_review_result


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
    """创建分析节点"""

    async def analyze(state: AppState) -> Dict[str, Any]:
        print(f"📝 [ContractReview:analyze] 合同分析中...")

        result = await contract_chain.ainvoke({
            "history": state.get("history", ""),
            "law": state.get("law_context", "未检索到相关法律条文"),
            "contract": state.get("contract_text", ""),
            "question": state["query"],
        })

        return {
            "draft_answer": result,
            "final_answer": result,
            "tool_history": state.get("tool_history", []) + [{
                "step": "analyze",
                "tool": "contract_chain",
                "input": state["query"],
                "output_len": len(result),
                "timestamp": time.time(),
            }],
        }

    return analyze


def make_re_synthesize_node(contract_chain):
    """创建重试合成节点（带增强提示）"""

    RETRY_PROMPT_SUFFIX = (
        "\n\n【重要】请严格基于提供的法律法规内容回答，"
        "不要引用未在参考资料中出现的法律名称和条文号。"
        "如果参考资料中没有相关信息，请用概括性表述。"
    )

    async def re_synthesize(state: AppState) -> Dict[str, Any]:
        print(f"🔄 [ContractReview:re_synthesize] Review Agent 发现问题，重新生成...")

        result = await contract_chain.ainvoke({
            "history": state.get("history", ""),
            "law": state.get("law_context", "未检索到相关法律条文"),
            "contract": state.get("contract_text", ""),
            "question": state["query"] + RETRY_PROMPT_SUFFIX,
        })

        return {
            "draft_answer": result,
            "final_answer": result,
            "review_retry": state.get("review_retry", 0) + 1,
            "tool_history": state.get("tool_history", []) + [{
                "step": "re_synthesize",
                "tool": "contract_chain_retry",
                "input": state["query"],
                "output_len": len(result),
                "timestamp": time.time(),
            }],
        }

    return re_synthesize


def create_contract_review_graph(
    llm,
    retrieval_service: RetrievalService,
):
    """
    创建合同审查 workflow 图

    流程：retrieve → analyze → review → END
    Review Agent 发现引用问题时：review → re_synthesize → review → END

    Args:
        llm: LangChain LLM 实例
        retrieval_service: RetrievalService 实例

    Returns:
        编译好的 LangGraph graph
    """
    contract_chain = create_contract_chain(llm)

    retrieve_node = make_retrieve_node(retrieval_service)
    analyze_node = make_analyze_node(contract_chain)
    review_node = create_review_agent_node(llm)
    re_synthesize_node = make_re_synthesize_node(contract_chain)

    workflow = StateGraph(AppState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("review", review_node)
    workflow.add_node("re_synthesize", re_synthesize_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "review")
    workflow.add_conditional_edges(
        "review",
        check_review_result,
        {
            "revise": "re_synthesize",
            "approve": END,
        },
    )
    workflow.add_edge("re_synthesize", "review")

    return workflow.compile()
