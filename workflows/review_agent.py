# -*- coding: utf-8 -*-
"""
Review Agent — 输出审查 Agent

独立的多 Agent 角色，负责审查主 Agent 的输出。
把 Guardrails 包装为工具，自主决定审查策略。

与旧的 policy_check 固定管线的区别：
- 旧：按固定顺序跑所有 guard，不管内容是否需要
- 新：Review Agent 自主决定调用哪些工具、按什么顺序、是否需要重试
"""
import json
import time
from typing import Dict, Any, List

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from utils.state import AppState
from utils.tools.guard_tools import get_guard_tools


REVIEW_SYSTEM_PROMPT = """你是输出质量审查专家。你的任务是审查 AI 助手生成的回复，确保安全、准确、合规。

## 审查步骤

1. **PII 检查**：用 pii_check 检查是否泄露个人信息（手机号、身份证、银行卡、邮箱）
2. **引用验证**：如果回复涉及法律内容（法条、合同、劳动法等），用 citation_check 验证引用的法律条文是否准确
3. **质量检查**：用 quality_check 检查回复是否过短或有重复
4. **免责声明**：如果涉及法律内容，用 add_disclaimer 添加免责声明

## 规则

- 只在必要时调用工具，不要重复调用同一个工具
- 如果回复不涉及法律内容，跳过 citation_check 和 add_disclaimer
- 如果 citation_check 发现问题，在你的最终回复中明确标注
- 最终输出必须是审查通过的完整回复文本，不要输出分析过程

## 输入格式

你会收到一段需要审查的 AI 回复文本，以及可选的法律检索上下文。"""


def create_review_agent_node(llm):
    """
    创建 Review Agent 节点

    用 create_react_agent 构建一个自主审查 Agent，
    它会根据回复内容自主决定调用哪些 Guard 工具。

    Args:
        llm: LangChain LLM 实例

    Returns:
        review 节点函数
    """
    guard_tools = get_guard_tools()
    agent = create_react_agent(
        llm,
        tools=guard_tools,
        prompt=REVIEW_SYSTEM_PROMPT,
    )

    async def review(state: AppState) -> Dict[str, Any]:
        print(f"🔍 [ReviewAgent] 开始审查输出...")

        draft = state.get("draft_answer", "")
        law_context = state.get("law_context", "")

        if not draft:
            print(f"⚠️ [ReviewAgent] 无草稿可审，跳过")
            return {
                "final_answer": "",
                "review_status": "approve",
                "review_issues": [],
            }

        # 构造审查输入
        review_input = f"请审查以下 AI 回复：\n\n{draft}"
        if law_context and len(law_context.strip()) >= 10:
            review_input += f"\n\n【法律检索上下文】\n{law_context}"

        try:
            # 调用 Review Agent
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=review_input)],
            })

            # 提取最终回复
            messages = result.get("messages", [])
            final_text = _extract_final_text(messages)

            # 检查是否发现引用问题
            citation_issues = _extract_citation_issues(messages)
            review_status = "revise" if citation_issues else "approve"

            if review_status == "revise":
                print(f"⚠️ [ReviewAgent] 发现引用问题: {citation_issues}")
            else:
                print(f"✅ [ReviewAgent] 审查通过")

            return {
                "final_answer": final_text,
                "review_status": review_status,
                "review_issues": citation_issues,
                "tool_history": state.get("tool_history", []) + [{
                    "step": "review",
                    "tool": "review_agent",
                    "input": draft[:200],
                    "output_len": len(final_text),
                    "review_status": review_status,
                    "timestamp": time.time(),
                }],
            }

        except Exception as e:
            print(f"⚠️ [ReviewAgent] 审查异常: {e}，使用原始草稿")
            return {
                "final_answer": draft,
                "review_status": "approve",
                "review_issues": [],
                "tool_history": state.get("tool_history", []) + [{
                    "step": "review",
                    "tool": "review_agent",
                    "input": draft[:200],
                    "error": str(e),
                    "timestamp": time.time(),
                }],
            }

    return review


def _extract_final_text(messages: list) -> str:
    """从 Agent 消息历史中提取最终文本"""
    # 从后往前找最后一条非工具调用的 AIMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
    # 如果没有找到，返回最后一条消息的内容
    if messages:
        last = messages[-1]
        if hasattr(last, 'content'):
            return last.content
    return ""


def _extract_citation_issues(messages: list) -> List[str]:
    """从 Agent 消息历史中提取引用验证问题"""
    issues = []
    for msg in messages:
        # 检查 ToolMessage 中的 citation_check 结果
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if "[引用验证] 发现问题" in content:
                # 提取问题描述
                lines = content.split("\n")
                for line in lines:
                    if "[引用验证]" in line:
                        issues.append(line.strip())
                        break
    return issues


def check_review_result(state: AppState) -> str:
    """条件边：检查 Review Agent 的审查结果"""
    review_retry = state.get("review_retry", 0)
    if state.get("review_status") == "revise" and review_retry < 1:
        return "revise"
    return "approve"
