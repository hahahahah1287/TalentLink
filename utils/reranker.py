# -*- coding: utf-8 -*-
"""
Cross-Encoder 重排序服务

功能：
- 对粗排结果进行精排
- 自动设备检测 (GPU/CPU)
- 内容去重
- 批量推理避免 OOM

依赖：仅 transformers + torch，不依赖 sentence-transformers。
"""
from typing import List, Optional
from langchain_core.documents import Document

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


class RerankService:
    """
    基于 Cross-Encoder 的重排序服务

    使用 transformers 直接加载模型，不依赖 sentence-transformers。

    使用示例:
        reranker = RerankService()
        reranked_docs = reranker.rerank("什么是试用期", docs)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

        if not RERANKER_AVAILABLE:
            return

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"📦 [Reranker] 加载模型 {model_name} 到 {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ [Reranker] 加载完成")
        except Exception as e:
            print(f"❌ [Reranker] 加载失败: {e}")
            self.model = None
            self.tokenizer = None

    def _predict_scores(self, pairs: List[List[str]]) -> List[float]:
        """
        对 query-doc 对进行打分

        Args:
            pairs: [[query, doc_content], ...]

        Returns:
            每个 pair 的相关性分数
        """
        all_scores = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits.squeeze(-1)
                scores = logits.float().cpu().tolist()

            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        return all_scores

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """
        对文档进行重排序

        Args:
            query: 用户查询
            docs: 待排序的文档列表
            top_k: 返回前 k 个结果
            score_threshold: 可选的分数阈值

        Returns:
            重排序后的文档列表
        """
        if not docs:
            return []

        if self.model is None:
            return docs[:top_k]

        # 1. 内容去重
        unique_docs_map = {}
        for doc in docs:
            content = doc.page_content.strip()
            if content and content not in unique_docs_map:
                unique_docs_map[content] = doc

        doc_list = list(unique_docs_map.values())
        if not doc_list:
            return []

        # 2. 构建 [query, doc] 对
        pairs = [[query, doc.page_content] for doc in doc_list]

        # 3. 批量打分
        try:
            scores = self._predict_scores(pairs)
        except Exception as e:
            print(f"❌ [Reranker] 推理失败: {e}")
            return docs[:top_k]

        # 4. 排序
        scored_docs = sorted(
            zip(scores, doc_list), key=lambda x: x[0], reverse=True
        )

        # 5. 可选：分数过滤；若阈值过高导致全空，回退到原始 top_k
        if score_threshold is not None:
            filtered = [(s, d) for s, d in scored_docs if s >= score_threshold]
            if filtered:
                scored_docs = filtered
            else:
                print("⚠️ [Reranker] 阈值过滤结果为空，回退到未过滤 top_k")

        return [doc for _, doc in scored_docs[:top_k]]

    def rerank_with_scores(
        self,
        query: str,
        docs: List[Document],
        top_k: int = 3,
    ) -> List[tuple]:
        """
        重排序并返回分数

        Returns:
            [(score, Document), ...]
        """
        if not docs or self.model is None:
            return [(0.0, doc) for doc in docs[:top_k]]

        unique_docs_map = {
            doc.page_content.strip(): doc
            for doc in docs
            if doc.page_content.strip()
        }
        doc_list = list(unique_docs_map.values())
        if not doc_list:
            return []

        pairs = [[query, doc.page_content] for doc in doc_list]

        try:
            scores = self._predict_scores(pairs)
        except Exception as e:
            print(f"❌ [Reranker] 推理失败: {e}")
            return [(0.0, doc) for doc in docs[:top_k]]

        scored_docs = sorted(
            zip(scores, doc_list), key=lambda x: x[0], reverse=True
        )

        return [(float(s), doc) for s, doc in scored_docs[:top_k]]
