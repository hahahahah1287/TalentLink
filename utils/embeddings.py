# -*- coding: utf-8 -*-
"""
Lightweight embedding wrapper based on transformers + torch.

LangChain's HuggingFaceBgeEmbeddings currently goes through sentence-transformers,
which is a large dependency.  The project already uses transformers + torch for
reranking, so embeddings share the same dependency family and expose the minimal
LangChain-compatible embed_query / embed_documents interface required by FAISS and
case retrieval.
"""
from typing import List, Optional

try:
    import torch
    import torch.nn.functional as F
    from langchain_core.embeddings import Embeddings
    from transformers import AutoModel, AutoTokenizer
    EMBEDDINGS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without ML deps
    torch = None
    F = None
    AutoModel = None
    AutoTokenizer = None
    Embeddings = object
    EMBEDDINGS_AVAILABLE = False


class TransformerEmbeddings(Embeddings):
    """LangChain-compatible embeddings without sentence-transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = 16,
        max_length: int = 512,
        local_files_only: bool = True,
    ):
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "TransformerEmbeddings requires transformers and torch. "
                "Install project requirements, but sentence-transformers is not required."
            )
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        self.local_files_only = local_files_only

        print(f"📦 [Embeddings] 加载模型 {model_name} 到 {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.to(device)
        self.model.eval()
        print("✅ [Embeddings] 加载完成")

    def _pool(self, last_hidden_state, attention_mask):
        # BGE 系列常用 CLS pooling；保持与常见 BGE 向量构建方式一致。
        return last_hidden_state[:, 0]

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
            vectors = self._pool(outputs.last_hidden_state, encoded["attention_mask"])
            if self.normalize_embeddings:
                vectors = F.normalize(vectors, p=2, dim=1)
        return vectors.detach().cpu().float().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        normalized = [text or "" for text in texts]
        vectors: List[List[float]] = []
        for i in range(0, len(normalized), self.batch_size):
            vectors.extend(self._encode_batch(normalized[i:i + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text or ""])[0]
