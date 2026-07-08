# -*- coding: utf-8 -*-
"""
Utils 工具包

导出：
- RerankService: 重排序服务
- QueryRewriter: 查询改写器（HyDE）
- GuardrailsPipeline: 输出防护管线
- parse_legal_document: 法律文档结构化切分
- annotate_documents: 元数据标注
- RetrievalService: 统一检索服务
- WorkflowCheckpoint: Redis 精确结果缓存
- PromptInjectionDetector: Prompt 注入检测器
"""

from .reranker import RerankService
from .query_rewriter import QueryRewriter
from .guardrails import GuardrailsPipeline
from .legal_parser import parse_legal_document
from .metadata_annotator import annotate_documents
from .retrieval_service import RetrievalService
from .checkpoint import WorkflowCheckpoint
from .security import PromptInjectionDetector
from .evidence import (
    document_to_evidence,
    documents_to_evidence,
    render_evidence_context,
)

__all__ = [
    "RerankService",
    "QueryRewriter",
    "GuardrailsPipeline",
    "parse_legal_document",
    "annotate_documents",
    "RetrievalService",
    "WorkflowCheckpoint",
    "PromptInjectionDetector",
    "document_to_evidence",
    "documents_to_evidence",
    "render_evidence_context",
]
