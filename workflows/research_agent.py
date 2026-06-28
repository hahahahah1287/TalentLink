# -*- coding: utf-8 -*-
"""
研究型任务 LangGraph Workflow

混合策略：Plan-and-Execute（全局规划） + ReAct（复杂 step 内灵活执行）。

架构：
  plan → execute (循环) → synthesize → review (Review Agent)

执行策略：
- 简单 step（web_search、job_search 等）→ 直接调用
- 复杂 step（risk_clause_detector、compliance_check）→ ReAct Agent 自主推理
  ReAct Agent 可灵活调用检索工具 + skill 工具，自主决定执行顺序

Planner 使用硬编码 JSON 解析 + fallback，适配 9B 小模型。
"""
import json
import time
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from utils.state import AppState
from utils.retrieval_service import RetrievalService
from utils.tools.contract_tools import create_synthesis_chain
from utils.tool_call_parser import _strip_tool_call_xml
from workflows.review_agent import create_review_agent_node, create_remove_citations_node, check_review_result
from workflows.shared_nodes import RETRY_PROMPT_SUFFIX
from skills.registry import SkillRegistry
from skills.base import create_skill_tool


# ==================== 硬编码 fallback 计划 ====================

FALLBACK_PLAN = ["legal_search", "web_search"]


# ==================== ReAct Step Executor ====================

def make_react_step_executor(
    llm,
    retrieval_service: RetrievalService,
    registry: SkillRegistry,
    extra_tools: list = None,
):
    """
    为复杂 step 创建 ReAct Agent 执行器

    ReAct Agent 拥有：
    - 检索工具（local_knowledge_search）
    - 搜索工具（web_search、job_search 等，通过 extra_tools 传入）
    - step 对应的 skill 工具

    Agent 自主决定调用哪些工具、以什么顺序调用。

    Args:
        llm: LangChain LLM 实例
        retrieval_service: RetrievalService 实例
        registry: SkillRegistry 实例
        extra_tools: 额外工具列表（如 WEB_SEARCH_TOOLS）

    Returns:
        execute_complex_step(step_name, query) 函数
    """
    retrieval_tool = retrieval_service.as_tool()  # local_knowledge_search
    extra_tools = extra_tools or []

    # 缓存已创建的 ReAct agent（按 step_name）
    _agent_cache: dict[str, Any] = {}

    def _build_prompt(step_name: str, all_tools: list) -> str:
        """根据可用工具动态生成系统 prompt"""
        tool_lines = []
        for t in all_tools:
            name = getattr(t, "name", t.__class__.__name__)
            desc = getattr(t, "description", "")
            tool_lines.append(f"- {name}: {desc}")
        tools_text = "\n".join(tool_lines)

        return (
            f"你是法律分析助手。请根据用户问题自主完成任务。\n\n"
            f"可用工具：\n"
            f"{tools_text}\n\n"
            f"提示：\n"
            f"- 需要法律依据时，用 local_knowledge_search 检索\n"
            f"- 需要联网补充信息时，用 web_search 或 job_search\n"
            f"- 调用 {step_name} 时，把检索到的内容通过 law_context 参数传入\n\n"
            f"请自主决定工具调用顺序和策略。"
        )

    def _get_or_create_agent(step_name: str):
        """懒加载 ReAct Agent（缓存，避免重复创建）"""
        if step_name not in _agent_cache:
            skill_fn = registry.get_skill_fn(step_name)
            description = registry.get_description(step_name) or step_name
            skill_tool = create_skill_tool(skill_fn, step_name, description)

            all_tools = [retrieval_tool] + extra_tools + [skill_tool]

            _agent_cache[step_name] = create_react_agent(
                llm,
                tools=all_tools,
                prompt=_build_prompt(step_name, all_tools),
            )
        return _agent_cache[step_name]

    def execute_complex_step(step_name: str, query: str) -> str:
        """
        用 ReAct Agent 执行复杂 step

        Args:
            step_name: skill 名称
            query: 用户问题

        Returns:
            ReAct Agent 的最终回复文本
        """
        print(f"🤖 [ReAct] 执行复杂 step: {step_name}")
        agent = _get_or_create_agent(step_name)

        # 限制迭代次数，防止 9B 模型无限循环
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": 6},
        )

        # 提取最后一条 AIMessage 的内容
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                cleaned = _strip_tool_call_xml(msg.content)
                if cleaned:
                    return cleaned

        # fallback：返回最后一条消息（清理 tool call XML）
        if messages:
            last = messages[-1]
            if hasattr(last, "content") and last.content:
                return _strip_tool_call_xml(str(last.content))
        return ""

    return execute_complex_step


# ==================== 节点工厂 ====================

def make_plan_node(llm, registry: SkillRegistry):
    """创建 Planner 节点（从 registry 动态获取选项列表）"""

    planner_options = registry.get_planner_options()

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是一个任务规划助手。根据用户的问题，决定需要执行哪些步骤。

可选步骤：
{planner_options}

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

            # 用 registry 验证步骤有效性
            valid_steps = set(registry.get_all_skill_names())
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


def make_execute_node(
    retrieval_service: RetrievalService,
    registry: SkillRegistry,
    react_executor,
):
    """
    创建执行节点 — 混合分发策略

    分发逻辑：
    1. legal_search → 特殊分支（依赖 retrieval_service）
    2. 复杂 skill → ReAct Agent 自主推理执行
    3. 简单 skill → 直接调用

    Args:
        retrieval_service: RetrievalService 实例
        registry: SkillRegistry 实例
        react_executor: make_react_step_executor 返回的函数
    """

    def execute(state: AppState) -> Dict[str, Any]:
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        query = state["query"]
        review_issues = state.get("review_issues", [])

        # Review Agent 反馈了问题 → 重置步骤，构建反馈上下文
        if review_issues and current_step >= len(plan):
            draft = state.get("draft_answer", "")
            issues_text = "\n".join(f"- {issue}" for issue in review_issues)
            feedback_context = (
                f"\n\n【Review Agent 发现的问题】\n{issues_text}\n\n"
                f"【上次错误输出（供参考，避免重复错误）】\n{draft[:500]}\n\n"
                f"请针对上述问题修正，不要重复同样的错误。"
            )
            print(f"🔄 [ResearchAgent:execute] Review Agent 反馈问题，重新执行全部步骤...")
            print(f"   具体问题: {review_issues}")
            # 重置步骤 + 存储反馈上下文 + 清空 review_issues
            current_step = 0
            return {
                "current_step": 0,
                "review_issues": [],
                "review_feedback": feedback_context,
                "tool_history": state.get("tool_history", []) + [{
                    "step": "execute_retry_start",
                    "tool": "feedback",
                    "input": issues_text,
                    "timestamp": time.time(),
                }],
            }

        # 追加反馈上下文到 query（每个子任务都能看到）
        feedback = state.get("review_feedback", "")
        if feedback:
            query = query + feedback

        if current_step >= len(plan):
            return {"status": "executed"}

        step_name = plan[current_step]
        print(f"⚡ [ResearchAgent:execute] 步骤 {current_step + 1}/{len(plan)}: {step_name}")

        result = ""
        try:
            if step_name == "legal_search":
                # 特殊分支：检索服务（不是 skill，直接调用 retrieval_service）
                result = retrieval_service.retrieve_as_string(query)
                return {
                    "law_context": result,
                    "current_step": current_step + 1,
                    "tool_history": state.get("tool_history", []) + [{
                        "step": f"execute_{step_name}",
                        "tool": step_name,
                        "input": query,
                        "output_len": len(result),
                        "timestamp": time.time(),
                    }],
                }

            elif registry.is_complex(step_name):
                # 复杂 step：ReAct Agent 自主推理（可调用检索 + skill 工具）
                result = react_executor(step_name, query)

            else:
                # 简单 step：直接调用 skill 函数
                skill_fn = registry.get_skill_fn(step_name)
                if skill_fn:
                    result = skill_fn(query)
                else:
                    result = f"未知步骤: {step_name}，已跳过。"
                    print(f"⚠️ [ResearchAgent:execute] 未注册的 skill: {step_name}")

            # 统一返回（简单 + 复杂 step 共用）
            return {
                "skill_outputs": {**state.get("skill_outputs", {}), step_name: result},
                "search_results": state.get("search_results", []) + [result],
                "current_step": current_step + 1,
                "tool_history": state.get("tool_history", []) + [{
                    "step": f"execute_{step_name}",
                    "tool": step_name,
                    "input": query,
                    "output_len": len(result),
                    "timestamp": time.time(),
                }],
            }

        except Exception as e:
            # 错误隔离：单个 step 失败不崩溃
            error_msg = f"步骤 {step_name} 执行失败: {str(e)}"
            print(f"❌ [ResearchAgent:execute] {error_msg}")
            return {
                "skill_outputs": {**state.get("skill_outputs", {}), step_name: error_msg},
                "current_step": current_step + 1,
                "tool_history": state.get("tool_history", []) + [{
                    "step": f"execute_{step_name}",
                    "tool": step_name,
                    "input": query,
                    "error": str(e),
                    "timestamp": time.time(),
                }],
            }

    return execute


def check_more_tasks(state: AppState) -> str:
    """条件边：检查是否还有剩余任务"""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    if current_step < len(plan):
        return "continue"
    return "done"


def make_synthesize_node(llm):
    """创建合成节点

    首次运行时直接合成；Review Agent 发现问题重跑时，
    会把 review_issues 拼入 question，让 LLM 知道上次哪里有问题并修正。
    """
    synthesis_chain = create_synthesis_chain(llm)

    async def synthesize(state: AppState) -> Dict[str, Any]:
        review_issues = state.get("review_issues", [])
        is_retry = bool(review_issues)

        if is_retry:
            print(f"🔄 [ResearchAgent:synthesize] Review Agent 反馈问题，重新合成...")
            print(f"   具体问题: {review_issues}")
        else:
            print(f"📝 [ResearchAgent:synthesize] 生成最终答案...")

        context_parts = []
        if state.get("law_context"):
            context_parts.append(f"【法律法规】\n{state['law_context']}")
        for i, sr in enumerate(state.get("search_results", [])):
            context_parts.append(f"【搜索结果 {i+1}】\n{sr}")
        context = "\n\n".join(context_parts) if context_parts else "无参考资料"

        # 拼接 question：首次直接用 query，重试时附带 Review Agent 的问题反馈
        question = state["query"]
        if is_retry:
            issues_text = "\n".join(f"- {issue}" for issue in review_issues)
            question += (
                f"\n\n{RETRY_PROMPT_SUFFIX}"
                f"\n\n【Review Agent 发现的具体问题】\n{issues_text}"
                f"\n\n请针对上述问题修正你的回答。"
            )

        result = await synthesis_chain.ainvoke({
            "history": state.get("history", ""),
            "context": context,
            "question": question,
        })

        step_name = "synthesize_retry" if is_retry else "synthesize"
        return {
            "draft_answer": result,
            "final_answer": result,
            "review_retry": state.get("review_retry", 0) + (1 if is_retry else 0),
            "tool_history": state.get("tool_history", []) + [{
                "step": step_name,
                "tool": "synthesis_chain",
                "input": question[:200],
                "output_len": len(result),
                "timestamp": time.time(),
            }],
        }

    return synthesize


# ==================== 图构建 ====================

def create_research_agent_graph(
    llm,
    retrieval_service: RetrievalService,
    registry: SkillRegistry,
    react_executor=None,
):
    """
    创建研究型任务 workflow 图

    混合策略：Plan-and-Execute（全局规划） + ReAct（复杂 step 内灵活执行）。

    流程：plan → execute (循环) → synthesize → review → END
    Review Agent 发现引用问题时：review → execute（带上问题反馈）→ synthesize → review → END

    Args:
        llm: LangChain LLM 实例
        retrieval_service: RetrievalService 实例
        registry: SkillRegistry 实例
        react_executor: ReAct 执行器（可选，不传则所有 step 都直接调用）

    Returns:
        编译好的 LangGraph graph
    """
    plan_node = make_plan_node(llm, registry)
    execute_node = make_execute_node(retrieval_service, registry, react_executor)
    synthesize_node = make_synthesize_node(llm)
    review_node = create_review_agent_node(llm)
    remove_node = create_remove_citations_node(llm)

    workflow = StateGraph(AppState)

    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("review", review_node)
    workflow.add_node("remove_citations", remove_node)

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
    workflow.add_edge("synthesize", "review")
    workflow.add_conditional_edges(
        "review",
        check_review_result,
        {
            "revise": "execute",
            "remove": "remove_citations",
            "approve": END,
        },
    )
    workflow.add_edge("remove_citations", END)

    return workflow.compile()
