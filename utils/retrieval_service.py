# -*- coding: utf-8 -*-
"""
统一检索服务

封装 HyDE 查询改写 + BM25/FAISS 混合检索 + Cross-Encoder Rerank。

HyDE 开关由 config.retrieval.hyde_enabled 统一控制，
调用方可传 use_hyde=True/False 覆盖，传 None 则使用配置默认值。
"""
from typing import List, Optional
from langchain.tools import tool
from langchain_core.documents import Document


class RetrievalService:
    """
    统一检索服务

    将 HyDE 改写、混合检索、Rerank 封装为单一入口，
    同时支持管线模式（直接调用）和工具模式（as_tool 给 agent 用）。
    """

    def __init__(
        self,
        retriever,           # EnsembleRetriever 实例
        reranker=None,       # RerankService 实例，可选
        query_rewriter=None, # QueryRewriter 实例，可选
        top_k: int = 5,
        rerank_enabled: bool = True,
        score_threshold: float = 0.3,
        hyde_enabled: bool = True,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.query_rewriter = query_rewriter
        self.top_k = top_k
        self.rerank_enabled = rerank_enabled
        self.score_threshold = score_threshold
        self.hyde_enabled = hyde_enabled

    def retrieve(self, query: str, use_hyde: Optional[bool] = None) -> List[Document]:
        """
        统一检索入口

        Args:
            query: 用户查询
            use_hyde: 是否启用 HyDE 查询改写。None 时使用 self.hyde_enabled 默认值

        Returns:
            检索并重排序后的文档列表
        """
        if use_hyde is None:
            use_hyde = self.hyde_enabled

        # 1. 可选 HyDE 改写
        effective_query = query
        if use_hyde and self.query_rewriter:
            try:
                effective_query = self.query_rewriter.hyde_rewrite(query)
                print(f"📝 [RetrievalService] HyDE 改写完成，增强查询长度: {len(effective_query)} 字符")
            except Exception as e:
                print(f"⚠️ [RetrievalService] HyDE 改写失败，使用原始查询: {e}")
                effective_query = query

        # 2. 混合检索 (BM25 + FAISS)
        raw_docs = self.retriever.invoke(effective_query)

        if not raw_docs:
            return []

        # 3. Cross-Encoder Rerank
        if self.rerank_enabled and self.reranker is not None:
            reranked_docs = self.reranker.rerank(
                query, raw_docs,
                top_k=self.top_k,
                score_threshold=self.score_threshold,
            )
            return reranked_docs

        return raw_docs[: self.top_k]

    def retrieve_as_string(
        self, query: str, use_hyde: Optional[bool] = None, separator: str = "\n\n"
    ) -> str:
        """
        检索并返回拼接后的文本

        Args:
            query: 用户查询
            use_hyde: 是否启用 HyDE
            separator: 文档之间的分隔符

        Returns:
            拼接的文档内容字符串
        """
        docs = self.retrieve(query, use_hyde=use_hyde)
        if not docs:
            return "未找到相关内容。"
        return separator.join(doc.page_content for doc in docs)

    def as_tool(self):
        """
        将检索能力封装为 LangChain Tool（供 ReAct agent 使用）

        agent 的 Thought 步骤替代 HyDE 做查询改写，因此 use_hyde=False。

        Returns:
            LangChain Tool 对象
        """
        service = self

        @tool
        def local_knowledge_search(query: str) -> str:
            """
            本地知识库搜索工具。用于查询已存储的法律法规、合同模板等历史数据。
            适合查询：劳动法、合同法、公司已有的法律文档等。

            Args:
                query: 查询内容

            Returns:
                相关文档内容
            """
            return service.retrieve_as_string(query)

        return local_knowledge_search
