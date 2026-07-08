# -*- coding: utf-8 -*-
"""
RAG 查询改写模块 (Query Rewriting)

Advanced RAG 的关键环节：用户的原始查询往往短小、模糊，
直接用于检索会导致召回质量下降。本模块实现 HyDE 改写：

  HyDE (Hypothetical Document Embedding):
    让 LLM 先生成一段"假答案"，用这段文本代替原始 query 做 Embedding 检索。
    原理：假答案与真实文档在语义空间中更接近，从而提升向量检索召回率。
    论文：Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QueryRewriter:
    """
    查询改写器（HyDE）

    使用示例:
        rewriter = QueryRewriter(llm)
        hyde_query = rewriter.hyde_rewrite("试用期工资怎么算")
        docs = retriever.invoke(hyde_query)
    """

    def __init__(self, llm, enabled: bool = True):
        """
        Args:
            llm: LLM 实例（用于生成改写查询）
            enabled: 是否启用改写（关闭时直接返回原查询）
        """
        self.llm = llm
        self.enabled = enabled

        # HyDE Prompt：让 LLM 扮演法律顾问生成假设性回答
        self._hyde_chain = (
            ChatPromptTemplate.from_messages([
                ("system", (
                    "你是一名资深法律顾问。请根据用户的问题，直接给出一段专业、详细的回答。"
                    "不要说'我不确定'之类的话，直接回答即可。\n"
                    "要求：回答长度 100-200 字，包含具体的法律条文引用。"
                )),
                ("user", "{query}")
            ])
            | self.llm
            | StrOutputParser()
        )

    def hyde_rewrite(self, query: str) -> str:
        """
        HyDE 查询改写

        生成假设性文档作为检索查询，提升向量语义匹配精度。

        Args:
            query: 原始用户查询

        Returns:
            假设性文档文本（用于替代原始 query 做 Embedding 检索）
        """
        if not self.enabled:
            return query

        try:
            hypothetical_doc = self._hyde_chain.invoke({"query": query})
            # 将原始查询和假设性文档拼接，兼顾关键词和语义
            return f"{query}\n{hypothetical_doc.strip()}"
        except Exception as e:
            print(f"⚠️ [QueryRewriter] HyDE 改写失败: {e}，使用原始查询")
            return query

