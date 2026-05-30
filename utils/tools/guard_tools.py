# -*- coding: utf-8 -*-
"""
Guard 工具层

把 Guardrails 中的 Guard 包装为 LangChain @tool，
供 Review Agent 自主调用。
"""
from langchain_core.tools import tool

from utils.guardrails import PIIGuard, CitationGuard, QualityGuard, DisclaimerGuard


# 单例 Guard 实例（无状态，可复用）
_pii_guard = PIIGuard()
_citation_guard = CitationGuard()
_quality_guard = QualityGuard()
_disclaimer_guard = DisclaimerGuard()


@tool
def pii_check(text: str) -> str:
    """检测并脱敏文本中的个人信息（手机号、身份证号、银行卡号、邮箱）。

    输入：需要检查的 AI 回复文本
    输出：脱敏后的文本，或未发现问题时返回原文
    """
    result = _pii_guard.process(text)
    if result.modified:
        return f"[PII脱敏] {result.details}\n\n{result.output}"
    return text


@tool
def citation_check(text: str, law_context: str) -> str:
    """验证回复中引用的法律名称和条文号是否存在于检索结果中。

    用于检测 AI 是否编造了不存在的法律条文。

    输入：
        text: 需要检查的 AI 回复文本
        law_context: 检索到的法律条文原文（从知识库检索的结果）
    输出：验证结果描述，包含发现的问题（如有）
    """
    if not law_context or len(law_context.strip()) < 10:
        return "跳过引用验证：未提供法律检索上下文"

    result = _citation_guard.process(text, context={"law_context": law_context})
    if result.modified:
        return f"[引用验证] 发现问题：{result.details}\n\n{result.output}"
    return "引用验证通过，所有引用的法律条文均在检索结果中找到。"


@tool
def quality_check(text: str) -> str:
    """检查回复质量，包括长度是否过短、是否有重复内容。

    输入：需要检查的 AI 回复文本
    输出：质量检查报告
    """
    result = _quality_guard.process(text)
    if result.modified:
        return f"[质量检查] {result.details}"
    return "质量检查通过。"


@tool
def add_disclaimer(text: str) -> str:
    """当回复涉及法律内容时，自动追加免责声明。

    只在回复包含法律关键词（如劳动法、赔偿、仲裁等）时才追加。
    如果回复不涉及法律内容，原样返回。

    输入：需要检查的 AI 回复文本
    输出：带免责声明的文本（如果涉及法律），或原文
    """
    result = _disclaimer_guard.process(text)
    if result.modified:
        return result.output
    return text


def get_guard_tools():
    """获取所有 Guard 工具列表"""
    return [pii_check, citation_check, quality_check, add_disclaimer]
