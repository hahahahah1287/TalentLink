# -*- coding: utf-8 -*-
"""
通用工具

策略检查、结果合成等跨场景复用的工具函数。
"""
from typing import Dict, Any, List


def policy_check(
    response: str,
    guardrails,
    context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    策略检查（调用 GuardrailsPipeline）

    Args:
        response: 待检查的回答
        guardrails: GuardrailsPipeline 实例
        context: 额外上下文（如 intent）

    Returns:
        检查结果，包含 output, modified, guards_triggered, policy_flags
    """
    result = guardrails.run(response, context=context or {})

    policy_flags = []
    citation_issues = []
    if result["modified"]:
        for guard_info in result.get("guards_triggered", []):
            guard_name = guard_info.get("guard", "unknown")
            policy_flags.append(guard_name)
            if guard_name == "引用验证":
                citation_issues.append(guard_info.get("details", ""))

    return {
        "output": result["output"],
        "modified": result["modified"],
        "guards_triggered": result.get("guards_triggered", []),
        "policy_flags": policy_flags,
        "citation_issues": citation_issues,
    }


def format_citations(docs: list) -> List[str]:
    """
    从检索文档中提取引用信息

    Args:
        docs: Document 列表

    Returns:
        引用信息列表
    """
    citations = []
    for doc in docs:
        source = doc.metadata.get("source", "未知来源")
        article = doc.metadata.get("article", "")
        summary = doc.metadata.get("summary", "")
        if article:
            citations.append(f"{source} - {article}: {summary}")
        else:
            citations.append(f"{source}: {doc.page_content[:100]}...")
    return citations
