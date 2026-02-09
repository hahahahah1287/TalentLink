# -*- coding: utf-8 -*-
"""
本地知识库检索技能

封装 FAISS + BM25 混合检索 + Rerank 为 LangChain Tool。
"""
from typing import List, Optional, Callable
from langchain.tools import tool
from langchain_core.documents import Document


class LocalRetrieverSkill:
    """
    本地知识库检索技能
    
    使用混合检索 (BM25 + FAISS) + Rerank 策略。
    设计为可注入的技能类，便于在不同 Agent 中复用。
    """
    
    def __init__(
        self,
        retriever,  # EnsembleRetriever 实例
        reranker = None,  # RerankService 实例，可选
        top_k: int = 3
    ):
        """
        Args:
            retriever: 混合检索器 (EnsembleRetriever)
            reranker: 重排序服务 (RerankService)，可选
            top_k: 最终返回的文档数量
        """
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[Document]:
        """
        检索并重排序
        
        Args:
            query: 用户查询
        
        Returns:
            相关文档列表
        """
        # 1. 粗排：混合检索 (BM25 + FAISS)
        raw_docs = self.retriever.get_relevant_documents(query)
        
        if not raw_docs:
            return []
        
        # 2. 精排：Cross-Encoder Rerank (如果可用)
        if self.reranker is not None:
            reranked_docs = self.reranker.rerank(query, raw_docs, top_k=self.top_k)
            return reranked_docs
        
        # 无 Reranker 时直接截断
        return raw_docs[:self.top_k]
    
    def retrieve_as_string(self, query: str, separator: str = "\n\n---\n\n") -> str:
        """
        检索并返回拼接后的文本
        
        Args:
            query: 用户查询
            separator: 文档之间的分隔符
        
        Returns:
            拼接后的文档内容
        """
        docs = self.retrieve(query)
        if not docs:
            return "未找到相关内容。"
        
        return separator.join([doc.page_content for doc in docs])
    
    def as_tool(self):
        """
        将检索能力封装为 LangChain Tool
        
        Returns:
            LangChain Tool 对象
        """
        skill = self  # 闭包捕获
        
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
            return skill.retrieve_as_string(query)
        
        return local_knowledge_search


def create_local_retriever_tool(
    retriever,
    reranker = None,
    top_k: int = 3
):
    """
    工厂函数：快速创建本地检索工具
    
    Args:
        retriever: 混合检索器
        reranker: 重排序服务（可选）
        top_k: 返回数量
    
    Returns:
        LangChain Tool 对象
    """
    skill = LocalRetrieverSkill(retriever, reranker, top_k)
    return skill.as_tool()
