# -*- coding: utf-8 -*-
"""
研究型任务 LangGraph Workflow

通用的研究型任务图，不限于求职。Planner 根据用户问题自动决定使用哪些工具：
  plan → execute (循环) → synthesize → policy_check

适用场景：
- "对比旧法律和新法律的区别" → legal_search + web_search
- "附近有什么工作" → job_search
- "劳动法第38条说了啥，顺便看看市场薪资" → legal_search + web_search + job_search

Planner 使用硬编码 JSON 解析 + fallback，适配 9B 小模型。
"""
import json
import time
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.state import AppState
from utils.retrieval_service import RetrievalService
from utils.tools.job_tools import web_search, job_search
from utils.tools.contract_tools import create_synthesis_chain
from utils.tools.common_tools import policy_check


# ==================== 硬编码 fallback 计划 ====================

FALLBACK_PLAN = ["legal_search", "web_search"]


# ==================== 节点工厂 ====================

def make_plan_node(llm):
    """创建 Planner 节点"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个任务规划助手。根据用户的问题，决定需要执行哪些步骤。

可选步骤：
- legal_search: 搜索本地法律法规知识库（适合查法律条文、合同模板、劳动法等）
- web_search: 联网搜索最新信息（适合查最新政策、新闻、行业动态等）
- job_search: 搜索招聘信息（适合查职位、薪资、公司招聘等）

规则：
- 只输出需要的步骤，不要多余
- 按执行顺序排列
- 如果只需要一步，也用列表格式"""),
        ("user", """用户问题：{query}

请输出 JSON 格式的计划：
{{"steps": ["step1", "step2", ...]}}""")
    ])

    chain = prompt | llm | StrOutputParser()

    async def plan(state: AppState) -> Dict[str, Any]:
        query = state["query"]
        print(f"📋 [ResearchAgent:plan] 生成执行计划...")

        try:
            raw_output = await chain.ainvoke({"query": query})
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(cleaned)
            steps = parsed.get("steps", [])

            valid_steps = {"legal_search", "web_search", "job_search"}
            steps = [s for s in steps if s in valid_steps]

            if not steps:
                print(f"⚠️ [ResearchAgent:plan] 计划无有效步骤，使用 fallback")
                steps = FALLBACK_PLAN

        except Exception as e:
            print(f"⚠️ [ResearchAgent:plan] JSON 解析失败: {e}，使用 fallback")
            steps = FALLBACK_PLAN

        print(f"📋 [ResearchAgent:plan] 执行计划: {steps}")

        return {
            "plan": steps,
            "current_step": 0,
            "tool_history": state.get("tool_history", []) + [{
                "step": "plan",
                "tool": "planner",
                "input": query,
                "output": steps,
                "timestamp": time.time(),
            }],
        }

    return plan


def make_execute_node(retrieval_service: RetrievalService):
    """创建执行节点（循环执行计划中的每一步）"""

    def execute(state: AppState) -> Dict[str, Any]:
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        query = state["query"]

        if current_step >= len(plan):
            return {"status": "executed"}

        step_name = plan[current_step]
        print(f"⚡ [ResearchAgent:execute] 步骤 {current_step + 1}/{len(plan)}: {step_name}")

        result = ""
        if step_name == "legal_search":
            result = retrieval_service.retrieve_as_string(query, use_hyde=False)
            updates = {
                "law_context": result,
                "current_step": current_step + 1,
            }
        elif step_name == "web_search":
            result = web_search(query)
            updates = {
                "search_results": state.get("search_results", []) + [result],
                "current_step": current_step + 1,
            }
        elif step_name == "job_search":
            result = job_search(query)
            updates = {
                "search_results": state.get("search_results", []) + [result],
                "current_step": current_step + 1,
            }
        else:
            updates = {"current_step": current_step + 1}

        updates["tool_history"] = state.get("tool_history", []) + [{
            "step": f"execute_{step_name}",
            "tool": step_name,
            "input": query,
            "output_len": len(result) if result else 0,
            "timestamp": time.time(),
        }]

        return updates

    return execute


def check_more_tasks(state: AppState) -> str:
    """条件边：检查是否还有剩余任务"""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    if current_step < len(plan):
        return "continue"
    return "done"


def make_synthesize_node(llm):
    """创建合成节点"""
    synthesis_chain = create_synthesis_chain(llm)

    async def synthesize(state: AppState) -> Dict[str, Any]:
        print(f"📝 [ResearchAgent:synthesize] 生成最终答案...")

        context_parts = []
        if state.get("law_context"):
            context_parts.append(f"【法律法规】\n{state['law_context']}")
        for i, sr in enumerate(state.get("search_results", [])):
            context_parts.append(f"【搜索结果 {i+1}】\n{sr}")
        context = "\n\n".join(context_parts) if context_parts else "无参考资料"

        result = await synthesis_chain.ainvoke({
            "history": state.get("history", ""),
            "context": context,
            "question": state["query"],
        })

        return {
            "draft_answer": result,
            "final_answer": result,
            "tool_history": state.get("tool_history", []) + [{
                "step": "synthesize",
                "tool": "synthesis_chain",
                "input": state["query"],
                "output_len": len(result),
                "timestamp": time.time(),
            }],
        }

    return synthesize


def make_policy_check_node(guardrails):
    """创建策略检查节点"""

    def check(state: AppState) -> Dict[str, Any]:
        print(f"🛡️ [ResearchAgent:policy_check] 合规检查中...")

        result = policy_check(
            response=state.get("final_answer", ""),
            guardrails=guardrails,
            context={"intent": "research"},
        )

        return {
            "final_answer": result["output"],
            "policy_flags": result["policy_flags"],
            "status": "done",
        }

    return check


# ==================== 图构建 ====================

def create_research_agent_graph(
    llm,
    retrieval_service: RetrievalService,
    guardrails,
):
    """
    创建研究型任务 workflow 图

    动态流程：plan → execute (循环) → synthesize → policy_check → END

    Planner 根据用户问题自动决定使用哪些工具。
    适用场景：法律对比、求职搜索、行业研究等需要多步推理的任务。

    Args:
        llm: LangChain LLM 实例
        retrieval_service: RetrievalService 实例
        guardrails: GuardrailsPipeline 实例

    Returns:
        编译好的 LangGraph graph
    """
    plan_node = make_plan_node(llm)
    execute_node = make_execute_node(retrieval_service)
    synthesize_node = make_synthesize_node(llm)
    policy_node = make_policy_check_node(guardrails)

    workflow = StateGraph(AppState)

    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("policy_check", policy_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "execute")
    workflow.add_conditional_edges(
        "execute",
        check_more_tasks,
        {
            "continue": "execute",
            "done": "synthesize",
        },
    )
    workflow.add_edge("synthesize", "policy_check")
    workflow.add_edge("policy_check", END)

    return workflow.compile()
