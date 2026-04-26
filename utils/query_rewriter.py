# -*- coding: utf-8 -*-
"""
RAG 查询改写模块 (Query Rewriting)

Advanced RAG 的关键环节：用户的原始查询往往短小、模糊，
直接用于检索会导致召回质量下降。
本模块实现两种改写策略：

1. HyDE (Hypothetical Document Embedding):
   让 LLM 先生成一段"假答案"，用这段文本代替原始 query 做 Embedding 检索。
   原理：假答案与真实文档在语义空间中更接近，从而提升向量检索召回率。
   论文：Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"

2. Multi-Query:
   对用户查询生成 N 个语义等价但措辞不同的变体查询，
   分别检索后合并去重，扩大召回覆盖面。
"""
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


class QueryRewriter:
    """
    查询改写器

    使用示例:
        rewriter = QueryRewriter(llm)

        # HyDE 模式
        hyde_query = rewriter.hyde_rewrite("试用期工资怎么算")
        docs = retriever.get_relevant_documents(hyde_query)

        # Multi-Query 模式
        queries = rewriter.multi_query_rewrite("试用期工资怎么算")
        all_docs = []
        for q in queries:
            all_docs.extend(retriever.get_relevant_documents(q))
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

        # Multi-Query Prompt：生成多个查询变体
        self._multi_query_chain = (
            ChatPromptTemplate.from_messages([
                ("system", (
                    "你是一个搜索查询优化专家。用户会给你一个查询，"
                    "请生成 3 个含义相同但措辞不同的查询变体，用于提升检索召回率。\n"
                    "要求：\n"
                    "- 每行一个查询\n"
                    "- 不要编号\n"
                    "- 不要添加任何其他文本\n"
                    "- 涵盖不同的表达角度（如口语化、书面化、关键词提取）"
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

    def multi_query_rewrite(self, query: str, include_original: bool = True) -> List[str]:
        """
        Multi-Query 查询改写

        生成多个语义等价的查询变体，分别检索后合并。

        Args:
            query: 原始用户查询
            include_original: 是否在返回结果中包含原始查询

        Returns:
            查询列表（含原始查询和变体）
        """
        if not self.enabled:
            return [query]

        queries = [query] if include_original else []

        try:
            raw_output = self._multi_query_chain.invoke({"query": query})
            # 解析生成的变体（每行一个）
            variants = [
                line.strip()
                for line in raw_output.strip().split("\n")
                if line.strip() and len(line.strip()) > 2
            ]
            queries.extend(variants[:3])  # 最多取 3 个变体
        except Exception as e:
            print(f"⚠️ [QueryRewriter] Multi-Query 改写失败: {e}")

        return queries if queries else [query]

    def rewrite_and_retrieve(
        self,
        query: str,
        retriever,
        strategy: str = "hyde",
        top_k: int = 5
    ) -> List[Document]:
        """
        端到端改写+检索

        Args:
            query: 原始用户查询
            retriever: 检索器实例
            strategy: 改写策略 - "hyde" | "multi_query" | "none"
            top_k: 返回文档数量

        Returns:
            去重后的文档列表
        """
        if strategy == "hyde":
            rewritten = self.hyde_rewrite(query)
            docs = retriever.get_relevant_documents(rewritten)
            return docs[:top_k]

        elif strategy == "multi_query":
            queries = self.multi_query_rewrite(query)
            # 并行检索并去重
            seen_contents = set()
            unique_docs = []
            for q in queries:
                for doc in retriever.get_relevant_documents(q):
                    content_key = doc.page_content.strip()[:100]
                    if content_key not in seen_contents:
                        seen_contents.add(content_key)
                        unique_docs.append(doc)
            return unique_docs[:top_k]

        else:
            return retriever.get_relevant_documents(query)[:top_k]
