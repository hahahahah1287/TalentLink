# -*- coding: utf-8 -*-
"""
合同分析工具

封装合同审查的 LLM 调用逻辑。
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_contract_chain(llm):
    """
    创建合同审查 LLM 链

    Args:
        llm: LangChain LLM 实例

    Returns:
        可调用的 chain（接受 dict，返回 str）
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名专业法务助手。
请基于以下法律法规库和用户上传的合同内容，专业地回答用户的问题。
如果法律库中没有相关条款，请明确说明并给出一般性建议。"""),
        ("user", """
【历史对话】
{history}

【法律法规参考】
{law}

【合同内容】
{contract}

【用户问题】
{question}

请给出专业分析：""")
    ])
    return prompt | llm | StrOutputParser()


def create_synthesis_chain(llm):
    """
    创建通用结果合成链（用于求职等场景的最终答案生成）

    Args:
        llm: LangChain LLM 实例

    Returns:
        可调用的 chain
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个有用、友好的 AI 助手。
请根据以下信息，简洁、准确地回答用户问题。
如果有检索到的参考资料，请优先基于资料回答；如果资料不足，请说明并给出一般性建议。"""),
        ("user", """
【历史对话】
{history}

【参考资料】
{context}

【用户问题】
{question}

请回答：""")
    ])
    return prompt | llm | StrOutputParser()
