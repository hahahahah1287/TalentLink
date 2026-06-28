# -*- coding: utf-8 -*-
"""
共享的 LangGraph 节点

提取 contract_review 和 research_agent 共用的节点逻辑，
消除重复代码。
"""
import time
from typing import Dict, Any

from utils.state import AppState
from utils.tools.contract_tools import create_synthesis_chain


# 共享的重试提示后缀
RETRY_PROMPT_SUFFIX = (
    "\n\n【重要】请严格基于提供的参考资料回答，"
    "不要引用未在参考资料中出现的法律名称和条文号。"
    "如果参考资料中没有相关信息，请用概括性表述。"
)


def _build_context(state: AppState) -> str:
    """从 state 中构建上下文文本（共享逻辑）"""
    context_parts = []
    if state.get("law_context"):
        context_parts.append(f"【法律法规】\n{state['law_context']}")
    for i, sr in enumerate(state.get("search_results", [])):
        context_parts.append(f"【搜索结果 {i+1}】\n{sr}")
    return "\n\n".join(context_parts) if context_parts else "无参考资料"


def make_re_synthesize_node(llm):
    """
    创建共享的重试合成节点

    Review Agent 发现问题后，用增强提示重新生成答案。
    contract_review 和 research_agent 共用此节点。
    """
    synthesis_chain = create_synthesis_chain(llm)

    async def re_synthesize(state: AppState) -> Dict[str, Any]:
        print(f"🔄 [ReSynthesize] Review Agent 发现问题，重新生成...")

        context = _build_context(state)

        result = await synthesis_chain.ainvoke({
            "history": state.get("history", ""),
            "context": context,
            "question": state["query"] + RETRY_PROMPT_SUFFIX,
        })

        return {
            "draft_answer": result,
            "final_answer": result,
            "review_retry": state.get("review_retry", 0) + 1,
            "tool_history": state.get("tool_history", []) + [{
                "step": "re_synthesize",
                "tool": "synthesis_chain_retry",
                "input": state["query"],
                "output_len": len(result),
                "timestamp": time.time(),
            }],
        }

    return re_synthesize
