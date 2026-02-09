# -*- coding: utf-8 -*-
"""
Cross-Encoder 重排序服务

功能：
- 对粗排结果进行精排
- 自动设备检测 (GPU/CPU)
- 内容去重
- 批量推理避免 OOM
"""
from typing import List, Optional
from langchain_core.documents import Document

try:
    from sentence_transformers import CrossEncoder
    import torch
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    print("⚠️ [Reranker] sentence-transformers 未安装，重排序功能不可用")


class RerankService:
    """
    基于 Cross-Encoder 的重排序服务
    
    使用示例:
        reranker = RerankService()
        reranked_docs = reranker.rerank("什么是试用期", docs)
    """
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        初始化重排序模型
        
        Args:
            model_name: HuggingFace 模型名称
            device: 运行设备 (cuda/cpu)，None 则自动检测
            batch_size: 批量推理大小
        """
        self.batch_size = batch_size
        self.model = None
        
        if not RERANKER_AVAILABLE:
            print("⚠️ [Reranker] 模块不可用，将跳过重排序")
            return
        
        # 自动设备检测
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"📦 [Reranker] 加载模型 {model_name} 到 {self.device}...")
        try:
            self.model = CrossEncoder(model_name, device=self.device)
            print(f"✅ [Reranker] 加载完成")
        except Exception as e:
            print(f"❌ [Reranker] 加载失败: {e}")
            self.model = None

    def rerank(
        self, 
        query: str, 
        docs: List[Document], 
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        对文档进行重排序
        
        Args:
            query: 用户查询
            docs: 待排序的文档列表
            top_k: 返回前 k 个结果
            score_threshold: 可选的分数阈值，低于此分数的文档会被过滤
        
        Returns:
            重排序后的文档列表
        """
        if not docs:
            return []
        
        # 模型不可用时，直接返回原结果
        if self.model is None:
            print("⚠️ [Reranker] 模型未加载，返回原始排序")
            return docs[:top_k]
        
        # 1. 内容去重 (基于 page_content)
        unique_docs_map = {}
        for doc in docs:
            content = doc.page_content.strip()
            if content and content not in unique_docs_map:
                unique_docs_map[content] = doc
        
        doc_list = list(unique_docs_map.values())
        
        if not doc_list:
            return []
        
        # 2. 构建模型输入对 [query, doc_content]
        pairs = [[query, doc.page_content] for doc in doc_list]
        
        # 3. 批量推理
        try:
            scores = self.model.predict(
                pairs, 
                batch_size=self.batch_size, 
                show_progress_bar=False
            )
        except Exception as e:
            print(f"❌ [Reranker] 推理失败: {e}")
            return docs[:top_k]
        
        # 4. 排序
        scored_docs = sorted(
            zip(scores, doc_list), 
            key=lambda x: x[0], 
            reverse=True
        )
        
        # 5. 可选：分数过滤
        if score_threshold is not None:
            scored_docs = [
                (score, doc) for score, doc in scored_docs 
                if score >= score_threshold
            ]
        
        # 6. 返回 top_k
        result = [doc for _, doc in scored_docs[:top_k]]
        
        return result
    
    def rerank_with_scores(
        self, 
        query: str, 
        docs: List[Document], 
        top_k: int = 3
    ) -> List[tuple]:
        """
        重排序并返回分数
        
        Returns:
            [(score, Document), ...]
        """
        if not docs or self.model is None:
            return [(0.0, doc) for doc in docs[:top_k]]
        
        unique_docs_map = {doc.page_content.strip(): doc for doc in docs if doc.page_content.strip()}
        doc_list = list(unique_docs_map.values())
        
        if not doc_list:
            return []
        
        pairs = [[query, doc.page_content] for doc in doc_list]
        
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception as e:
            print(f"❌ [Reranker] 推理失败: {e}")
            return [(0.0, doc) for doc in docs[:top_k]]
        
        scored_docs = sorted(zip(scores, doc_list), key=lambda x: x[0], reverse=True)
        
        return [(float(score), doc) for score, doc in scored_docs[:top_k]]
