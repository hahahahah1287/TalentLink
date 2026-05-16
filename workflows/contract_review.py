# -*- coding: utf-8 -*-
"""
合同审查 LangGraph Workflow

固定流程，不需要 Planner 动态规划：
  retrieve → analyze → policy_check

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
from utils.tools.common_tools import policy_check


def make_retrieve_node(retrieval_service: RetrievalService):
    """创建检索节点"""

    def retrieve(state: AppState) -> Dict[str, Any]:
        query = state["query"]
        print(f"🔍 [ContractReview:retrieve] 检索法律条文: {query[:50]}...")

        law_context = retrieval_service.retrieve_as_string(query, use_hyde=True)

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


def make_policy_check_node(guardrails):
    """创建策略检查节点"""

    def check(state: AppState) -> Dict[str, Any]:
        print(f"🛡️ [ContractReview:policy_check] 合规检查中...")

        result = policy_check(
            response=state.get("final_answer", ""),
            guardrails=guardrails,
            context={"intent": "contract_critique"},
        )

        return {
            "final_answer": result["output"],
            "policy_flags": result["policy_flags"],
            "status": "done",
        }

    return check


def create_contract_review_graph(
    llm,
    retrieval_service: RetrievalService,
    guardrails,
):
    """
    创建合同审查 workflow 图

    固定流程：retrieve → analyze → policy_check → END

    Args:
        llm: LangChain LLM 实例
        retrieval_service: RetrievalService 实例
        guardrails: GuardrailsPipeline 实例

    Returns:
        编译好的 LangGraph graph
    """
    contract_chain = create_contract_chain(llm)

    retrieve_node = make_retrieve_node(retrieval_service)
    analyze_node = make_analyze_node(contract_chain)
    policy_node = make_policy_check_node(guardrails)

    workflow = StateGraph(AppState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("policy_check", policy_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "policy_check")
    workflow.add_edge("policy_check", END)

    return workflow.compile()
