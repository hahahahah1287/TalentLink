# -*- coding: utf-8 -*-
"""
Article-level legal retrieval evaluation.

This is the deterministic legal supplement to the real RAGAS chain in
`tests/eval_ragas.py`: it evaluates retrieval against expected law/article labels
and keeps token-overlap metrics only as auxiliary diagnostics.

Usage:
    python tests/eval_legal_retrieval.py --limit 20 --no-hyde
    python tests/eval_legal_retrieval.py --top-k 5 --output tests/reports/legal_retrieval.json
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever

from config import AppConfig
from tests.ragas_dataset import RAGAS_DATASET
from utils import RerankService, RetrievalService
from utils.embeddings import TransformerEmbeddings
from utils.evidence import normalize_article_number, normalize_law_title


ArticleRef = Tuple[str, int]


# ==================== legacy token-overlap diagnostics ====================

def _tokenize(text: str) -> set:
    article_refs = set(re.findall(r"第[一二三四五六七八九十百千零〇\d]+[条章节款项目]", text or ""))
    stop_words = {
        "的", "了", "在", "是", "和", "与", "或", "及", "等", "对", "为", "不", "有",
        "这", "那", "被", "从", "到", "中", "上", "下", "内", "外", "其", "该", "将",
        "已", "也", "都", "而", "但", "如", "则", "可", "应", "要", "会", "能", "以",
        "于", "由", "因", "此", "之", "所",
    }
    words = set(w for w in jieba.lcut(text or "") if len(w) >= 2 and w not in stop_words)
    en_words = set(re.findall(r"[a-zA-Z]+", (text or "").lower()))
    numbers = set(re.findall(r"\d+", text or ""))
    return words | article_refs | en_words | numbers


def compute_context_precision(question: str, contexts: Sequence[str]) -> float:
    if not contexts:
        return 0.0
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    c_tokens = _tokenize(" ".join(contexts))
    return len(q_tokens & c_tokens) / len(q_tokens)


def compute_context_recall(ground_truth: str, contexts: Sequence[str]) -> float:
    if not contexts:
        return 0.0
    gt_tokens = _tokenize(ground_truth)
    if not gt_tokens:
        return 0.0
    c_tokens = _tokenize(" ".join(contexts))
    return len(gt_tokens & c_tokens) / len(gt_tokens)


# ==================== article-label extraction ====================

_LAW_ARTICLE_PATTERN = re.compile(
    r"《([^》]+)》[^。；;，,、\n]{0,20}?第\s*([一二三四五六七八九十百千零〇\d]+)\s*条"
)
_ARTICLE_PATTERN = re.compile(r"第\s*([一二三四五六七八九十百千零〇\d]+)\s*条")


def _canonical_law(title: str) -> str:
    return normalize_law_title(title or "中华人民共和国劳动法")["canonical_law_title"]


def _normalize_ref(law_title: str, article: Any) -> Optional[ArticleRef]:
    article_num = normalize_article_number(article)
    if article_num is None:
        return None
    return (_canonical_law(law_title), article_num)


def expected_refs_from_item(item: Dict[str, Any]) -> List[ArticleRef]:
    """Support explicit labels first, then fallback to ground_truth extraction."""
    refs: List[ArticleRef] = []

    for ref in item.get("expected_refs", []) or []:
        if isinstance(ref, dict):
            normalized = _normalize_ref(ref.get("law") or ref.get("law_title"), ref.get("article"))
            if normalized:
                refs.append(normalized)

    expected_law = item.get("expected_law") or "中华人民共和国劳动法"
    for article in item.get("expected_articles", []) or []:
        if isinstance(article, dict):
            normalized = _normalize_ref(article.get("law") or article.get("law_title") or expected_law, article.get("article"))
        else:
            normalized = _normalize_ref(expected_law, article)
        if normalized:
            refs.append(normalized)

    if not refs:
        ground_truth = item.get("ground_truth", "")
        for law_title, article_text in _LAW_ARTICLE_PATTERN.findall(ground_truth):
            normalized = _normalize_ref(law_title, article_text)
            if normalized:
                refs.append(normalized)
        # If the ground truth uses bare 第X条, default to expected_law / 劳动法.
        if not refs:
            for article_text in _ARTICLE_PATTERN.findall(ground_truth):
                normalized = _normalize_ref(expected_law, article_text)
                if normalized:
                    refs.append(normalized)

    # Stable de-duplication preserving order.
    deduped: List[ArticleRef] = []
    seen = set()
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped


def retrieved_refs_from_evidence(evidence_items: Sequence[Dict[str, Any]]) -> List[ArticleRef]:
    refs: List[ArticleRef] = []
    for item in evidence_items or []:
        source = item.get("source") or {}
        article_num = normalize_article_number(source.get("article_number"))
        law_title = source.get("canonical_law_title") or source.get("law_title")
        if article_num is None or not law_title:
            continue
        refs.append((_canonical_law(law_title), article_num))
    return refs


def _ref_to_dict(ref: ArticleRef) -> Dict[str, Any]:
    return {"law": ref[0], "article": ref[1]}


# ==================== article-level metrics ====================

def hit_at_k(expected: Sequence[ArticleRef], retrieved: Sequence[ArticleRef], k: int) -> float:
    if not expected:
        return 0.0
    return 1.0 if set(expected) & set(retrieved[:k]) else 0.0


def recall_at_k(expected: Sequence[ArticleRef], retrieved: Sequence[ArticleRef], k: int) -> float:
    if not expected:
        return 0.0
    return len(set(expected) & set(retrieved[:k])) / len(set(expected))


def precision_at_k(expected: Sequence[ArticleRef], retrieved: Sequence[ArticleRef], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(expected) & set(retrieved[:k])) / k


def mrr(expected: Sequence[ArticleRef], retrieved: Sequence[ArticleRef]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    for idx, ref in enumerate(retrieved, 1):
        if ref in expected_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(expected: Sequence[ArticleRef], retrieved: Sequence[ArticleRef], k: int) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    dcg = 0.0
    for idx, ref in enumerate(retrieved[:k], 1):
        rel = 1.0 if ref in expected_set else 0.0
        dcg += rel / math.log2(idx + 1)
    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _avg(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


# ==================== retrieval service ====================

def build_retrieval_service(config: AppConfig, enable_hyde: bool) -> RetrievalService:
    """Build the production-like retrieval chain without hardcoding HyDE."""
    embeddings = TransformerEmbeddings(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        normalize_embeddings=config.embedding.normalize_embeddings,
        local_files_only=True,
    )
    reranker = RerankService(
        model_name=config.reranker.model_name,
        device=config.reranker.device,
        batch_size=config.reranker.batch_size,
    )
    vector_store = FAISS.load_local(
        config.retrieval.faiss_index_path + "_with_metadata",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    docs = list(vector_store.docstore._dict.values())

    faiss_ret = vector_store.as_retriever(search_kwargs={"k": config.retrieval.retrieval_k})
    bm25_ret = BM25Retriever.from_documents(docs)
    bm25_ret.k = config.retrieval.retrieval_k

    ensemble = EnsembleRetriever(
        retrievers=[bm25_ret, faiss_ret],
        weights=[config.retrieval.bm25_weight, config.retrieval.faiss_weight],
    )

    query_rewriter = None
    if enable_hyde:
        print(f"📦 加载 LLM（用于 HyDE 改写）: {config.llm.model_path}...")
        try:
            from langchain_experimental.chat_models import ChatLlamaCpp
        except ImportError:
            from langchain_community.chat_models import ChatLlamaCpp
        from utils import QueryRewriter
        llm = ChatLlamaCpp(
            model_path=config.llm.model_path,
            n_gpu_layers=config.llm.n_gpu_layers,
            n_ctx=config.llm.n_ctx,
            temperature=config.llm.temperature,
            verbose=False,
            streaming=False,
        )
        query_rewriter = QueryRewriter(llm=llm, enabled=True)
        print("✅ HyDE 查询改写已启用")

    return RetrievalService(
        retriever=ensemble,
        reranker=reranker,
        query_rewriter=query_rewriter,
        top_k=config.reranker.top_k,
        rerank_enabled=config.retrieval.rerank_enabled,
        score_threshold=config.reranker.score_threshold,
        hyde_enabled=enable_hyde,
    )


# ==================== main ====================

def main():
    parser = argparse.ArgumentParser(description="Article-level legal retrieval eval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default="eval_ragas_report.json")
    parser.add_argument("--hyde", dest="hyde", action="store_true", help="启用 HyDE 查询改写")
    parser.add_argument("--no-hyde", dest="hyde", action="store_false", help="关闭 HyDE 查询改写")
    parser.set_defaults(hyde=None)
    args = parser.parse_args()

    config = AppConfig()
    enable_hyde = config.retrieval.hyde_enabled if args.hyde is None else args.hyde
    dataset = RAGAS_DATASET[:args.limit] if args.limit > 0 else RAGAS_DATASET

    print(f"\n{'=' * 60}")
    print(f"法条级检索评估 — {len(dataset)} 条 | top_k={args.top_k} | hyde={enable_hyde}")
    print(f"{'=' * 60}\n")

    retrieval_service = build_retrieval_service(config, enable_hyde=enable_hyde)

    results: List[Dict[str, Any]] = []
    for i, item in enumerate(dataset):
        print(f"  [{i + 1}/{len(dataset)}] {item['question'][:60]}...")
        expected = expected_refs_from_item(item)
        retrieval = retrieval_service.retrieve_with_evidence(item["question"], use_hyde=enable_hyde)
        evidence_items = retrieval["evidence_items"]
        retrieved = retrieved_refs_from_evidence(evidence_items)
        contexts = [d.page_content for d in retrieval["docs"]]

        row = {
            "id": item.get("id", f"sample_{i + 1:03d}"),
            "question": item["question"],
            "scene": item.get("scene", "legal"),
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "expected_refs": [_ref_to_dict(r) for r in expected],
            "retrieved_refs": [_ref_to_dict(r) for r in retrieved],
            "retrieved_citations": [
                (e.get("source") or {}).get("canonical_citation")
                for e in evidence_items
            ],
            "hit@1": round(hit_at_k(expected, retrieved, 1), 4),
            f"hit@{args.top_k}": round(hit_at_k(expected, retrieved, args.top_k), 4),
            f"recall@{args.top_k}": round(recall_at_k(expected, retrieved, args.top_k), 4),
            f"precision@{args.top_k}": round(precision_at_k(expected, retrieved, args.top_k), 4),
            "mrr": round(mrr(expected, retrieved), 4),
            f"ndcg@{args.top_k}": round(ndcg_at_k(expected, retrieved, args.top_k), 4),
            "context_token_precision": round(compute_context_precision(item["question"], contexts), 4),
            "context_token_recall": round(compute_context_recall(item.get("ground_truth", ""), contexts), 4),
            "missing_refs": [_ref_to_dict(r) for r in set(expected) - set(retrieved[:args.top_k])],
            "unexpected_refs": [_ref_to_dict(r) for r in set(retrieved[:args.top_k]) - set(expected)],
            "num_contexts": len(contexts),
        }
        results.append(row)

    metric_keys = [
        "hit@1", f"hit@{args.top_k}", f"recall@{args.top_k}",
        f"precision@{args.top_k}", "mrr", f"ndcg@{args.top_k}",
        "context_token_precision", "context_token_recall",
    ]
    summary = {key: round(_avg(row[key] for row in results), 4) for key in metric_keys}

    def grouped(field: str) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in results:
            value = row.get(field) or "unknown"
            groups.setdefault(value, []).append(row)
        return {
            value: {
                "count": len(rows),
                **{key: round(_avg(row[key] for row in rows), 4) for key in metric_keys},
            }
            for value, rows in groups.items()
        }

    report = {
        "config": {
            "top_k": args.top_k,
            "hyde": enable_hyde,
            "rerank_enabled": config.retrieval.rerank_enabled,
            "score_threshold": config.reranker.score_threshold,
            "faiss_index_path": config.retrieval.faiss_index_path,
        },
        "summary": summary,
        "by_category": grouped("category"),
        "by_difficulty": grouped("difficulty"),
        "details": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("评估结果")
    print(f"{'=' * 60}")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    print(f"\n📄 报告已保存: {output_path}")


if __name__ == "__main__":
    main()
