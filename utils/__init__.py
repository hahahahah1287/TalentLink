# -*- coding: utf-8 -*-
"""
Utils 工具包

导出：
- RerankService: 重排序服务
- create_text_splitter: 分词器工厂函数
- CircuitBreaker: 熔断器
- QueryRewriter: 查询改写器
- SemanticCache: 语义缓存
- GuardrailsPipeline: 输出防护管线
- parse_legal_document: 法律文档结构化切分
- annotate_documents: 元数据标注
"""

from .reranker import RerankService
from .text_splitter import create_text_splitter
from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from .query_rewriter import QueryRewriter
from .semantic_cache import SemanticCache
from .guardrails import GuardrailsPipeline
from .legal_parser import parse_legal_document
from .metadata_annotator import annotate_documents

__all__ = [
    "RerankService",
    "create_text_splitter",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "QueryRewriter",
    "SemanticCache",
    "GuardrailsPipeline",
    "parse_legal_document",
    "annotate_documents",
]
