# -*- coding: utf-8 -*-
"""
Review Agent — 输出审查 Agent

独立的多 Agent 角色，负责审查主 Agent 的输出。
把 Guardrails 包装为工具，自主决定审查策略。

与旧的 policy_check 固定管线的区别：
- 旧：按固定顺序跑所有 guard，不管内容是否需要
- 新：Review Agent 自主决定调用哪些工具、按什么顺序、是否需要重试
"""
import re
import json
import time
from typing import Dict, Any, List

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from utils.state import AppState
from utils.tools.guard_tools import get_guard_tools


REVIEW_SYSTEM_PROMPT = """你是输出质量审查专家。审查 AI 回复的安全性和准确性。

## 可用工具

- pii_check: 检测并脱敏个人信息（手机号、身份证、银行卡、邮箱）
- citation_check: 验证法律引用是否准确（需要 law_context）
- quality_check: 检查回复质量（长度、重复）
- add_disclaimer: 添加法律免责声明

## 审查策略

根据回复内容自主决定需要调用哪些工具。不涉及法律内容时可跳过 citation_check 和 add_disclaimer。

## 约束

- 每个工具最多调用1次
- 工具调用完毕后，直接输出审查通过的完整回复文本，不要输出分析过程
- 如果 citation_check 发现问题，在回复中用 [引用问题] 标注具体哪些引用有误，系统会自动触发重新生成"""


def _build_tool_call_pattern():
    """
    构建 tool call XML 清理模式。

    策略：匹配 XML 标签，但只移除"看起来像 tool call"的标签。
    判断依据（三选一即匹配）：
    1. 标签名包含 tool/call/function（覆盖 tool_call、function 等常见格式）
    2. 标签内包含 JSON 内容 {...}（tool call 通常输出 JSON 参数）
    3. 未闭合的 tool/call/function 标签到字符串末尾

    这样既不会遗漏常见的 tool call 格式，也不会误删法律文本中的 <条> 等内容。
    """
    o, c, s = chr(60), chr(62), chr(47)  # <, >, /

    # 模式1：标签名包含 tool/call/function，完整开闭标签对
    named = (
        o + r'(\w*(?:tool|call|function)\w*)[^' + c + r']*' + c
        + r'[\s\S]*?'
        + o + s + r'\1' + c
    )

    # 模式2：任何标签，但内容包含 JSON（花括号）→ 大概率是 tool call
    with_json = (
        o + r'(\w+)[^' + c + r']*' + c
        + r'[^' + c + r']*\{[\s\S]*?\}[^' + c + r']*'
        + o + s + r'\1' + c
    )

    # 模式3：未闭合的 tool/call/function 标签（到字符串末尾）
    unclosed = (
        o + r'(\w*(?:tool|call|function)\w*)[^' + c + r']*' + c
        + r'[\s\S]*'
    )

    return re.compile(named + r'|' + with_json + r'|' + unclosed, re.IGNORECASE)


_TOOL_CALL_RE = _build_tool_call_pattern()


def _strip_tool_call_xml(text: str) -> str:
    """
    清理 LLM 输出中的 tool call XML 标签。

    某些小模型（如 Qwen 9B）会将 tool call 以纯文本 XML 格式输出在 content 字段中，
    而非使用 LangChain 的 tool_calls 属性。此函数移除这些 XML 片段。
    """
    if not text:
        return ""
    cleaned = _TOOL_CALL_RE.sub('', text)
    # 清理移除标签后残留的多余空行
    cleaned = re.sub(r'\n{2,}', '\n', cleaned).strip()
    return cleaned


# 动态获取 Guard 工具名集合（用于 _extract_final_text 跳过中间结果）
_GUARD_TOOL_NAMES = {t.name for t in get_guard_tools()}


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
            # 调用 Review Agent（限制迭代次数，防止 9B 模型无限循环）
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=review_input)]},
                config={"recursion_limit": 6},
            )

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
    """
    从 Agent 消息历史中提取最终文本。

    策略：
    1. 优先找最后一条有内容、无 tool_calls 的 AIMessage（正常结束）
    2. 若找不到，fallback 到非 Guard 工具的 ToolMessage（Agent 以工具调用结束的异常情况）
    3. 最终兜底：最后一条消息的内容
    """
    # 路径1：找最后一条"干净"的 AIMessage（有内容、无 tool_calls）
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            cleaned = _strip_tool_call_xml(msg.content)
            if cleaned:
                return cleaned

    # 路径2：fallback 到 ToolMessage（跳过 Guard 工具的中间结果）
    # 用 ToolMessage.name 判断是否为 Guard 工具，不依赖内容字符串匹配
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # 跳过 Guard 工具（pii_check、citation_check 等）
            if msg.name in _GUARD_TOOL_NAMES:
                continue
            cleaned = _strip_tool_call_xml(
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            if cleaned:
                return cleaned

    # 路径3：最终兜底
    if messages:
        last = messages[-1]
        if hasattr(last, 'content') and last.content:
            return _strip_tool_call_xml(str(last.content))
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


REMOVE_CITATIONS_PROMPT = """你是一个文本编辑器。你的任务是从给定文本中**精准删除**指定的有问题的法律引用，其他内容保持不变。

## 严格规则

1. **只删除**下面列出的有问题的引用（包括引用所在的句子或从句）
2. **不要**删除其他任何内容
3. **不要**改写、重述或补充其他内容
4. **不要**添加新的内容
5. 如果删除引用后段落变得不通顺，只做最小必要的调整（如连接词）
6. 删除后保留原文的结构和格式

## 有问题的引用

{issues}

## 原文

{text}

## 输出要求

直接输出修改后的完整文本，不要输出解释、分析或其他内容。"""


def create_remove_citations_node(llm):
    """创建精准删除有问题引用的节点

    当 Review Agent 重试后仍有引用问题时，
    用 LLM 精准删除有问题的引用，其他内容不动。
    """

    async def remove_citations(state: AppState) -> Dict[str, Any]:
        draft = state.get("draft_answer", "")
        review_issues = state.get("review_issues", [])

        if not draft or not review_issues:
            return {"final_answer": draft, "review_issues": []}

        issues_text = "\n".join(f"- {issue}" for issue in review_issues)
        print(f"✂️ [RemoveCitations] 精准删除有问题的引用...")
        print(f"   待删除: {review_issues}")

        prompt = REMOVE_CITATIONS_PROMPT.format(
            issues=issues_text,
            text=draft,
        )

        try:
            result = await llm.ainvoke(prompt)
            cleaned = result.content if hasattr(result, "content") else str(result)
            cleaned = _strip_tool_call_xml(cleaned)
        except Exception as e:
            print(f"⚠️ [RemoveCitations] LLM 调用异常: {e}，使用原文")
            cleaned = draft

        return {
            "final_answer": cleaned,
            "review_issues": [],  # 清空，不再重试
            "tool_history": state.get("tool_history", []) + [{
                "step": "remove_citations",
                "tool": "llm_cleanup",
                "input": issues_text,
                "output_len": len(cleaned),
                "timestamp": time.time(),
            }],
        }

    return remove_citations


def check_review_result(state: AppState) -> str:
    """条件边：检查 Review Agent 的审查结果

    两阶段策略：
    - 第一次发现问题 → revise（重跑源头，修正引用）
    - 第二次还有问题 → remove（LLM 精准删除有问题的引用）
    """
    review_retry = state.get("review_retry", 0)
    if state.get("review_status") != "revise":
        return "approve"
    if review_retry < 1:
        return "revise"
    return "remove"
