# -*- coding: utf-8 -*-
"""
RAG 评估脚本（不依赖 LLM Judge）

评估 RAG 系统的 2 个核心指标：
- Context Precision（上下文精确度）
- Context Recall（上下文召回率）

用法：
    python tests/eval_ragas.py
    python tests/eval_ragas.py --limit 10
    python tests/eval_ragas.py --hyde  # 启用 HyDE
"""
import os
import sys
import json
import re
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jieba
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever

from config import AppConfig
from utils import RerankService, RetrievalService
from tests.ragas_dataset import RAGAS_DATASET


# ==================== 评估指标 ====================

def _tokenize(text: str) -> set:
    article_refs = set(re.findall(r'第[一二三四五六七八九十百千零\d]+[条章节款项目]', text))
    stop_words = {'的', '了', '在', '是', '和', '与', '或', '及', '等',
                  '对', '为', '不', '有', '这', '那', '被', '从', '到',
                  '中', '上', '下', '内', '外', '其', '该', '将', '已',
                  '也', '都', '而', '但', '如', '则', '可', '应', '要',
                  '会', '能', '以', '于', '由', '因', '此', '之', '所'}
    words = set(w for w in jieba.lcut(text) if len(w) >= 2 and w not in stop_words)
    en_words = set(re.findall(r'[a-zA-Z]+', text.lower()))
    numbers = set(re.findall(r'\d+', text))
    return words | article_refs | en_words | numbers


def compute_context_precision(question: str, contexts: list) -> float:
    if not contexts:
        return 0.0
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    c_tokens = _tokenize(" ".join(contexts))
    return len(q_tokens & c_tokens) / len(q_tokens)


def compute_context_recall(ground_truth: str, contexts: list) -> float:
    if not contexts:
        return 0.0
    gt_tokens = _tokenize(ground_truth)
    if not gt_tokens:
        return 0.0
    c_tokens = _tokenize(" ".join(contexts))
    return len(gt_tokens & c_tokens) / len(gt_tokens)


# ==================== 构建检索服务 ====================

def build_retrieval_service(config: AppConfig, enable_hyde: bool = False) -> RetrievalService:
    """构建与生产一致的检索链路"""
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=config.embedding.model_name,
        model_kwargs={'device': config.embedding.device},
        encode_kwargs={'normalize_embeddings': config.embedding.normalize_embeddings}
    )
    reranker = RerankService(
        model_name=config.reranker.model_name,
        device=config.reranker.device,
        batch_size=config.reranker.batch_size
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
        print("✅ LLM 加载完成")
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


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="RAG 评估")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=str, default="eval_ragas_report.json")
    parser.add_argument("--hyde", action="store_true")
    args = parser.parse_args()

    config = AppConfig()
    dataset = RAGAS_DATASET[:args.limit] if args.limit > 0 else RAGAS_DATASET

    print(f"\n{'='*60}")
    print(f"RAG 评估 — {len(dataset)} 条")
    print(f"{'='*60}\n")

    retrieval_service = build_retrieval_service(config,True)

    print(f"\n🔄 运行评估...\n")
    results = []
    for i, item in enumerate(dataset):
        print(f"  [{i+1}/{len(dataset)}] {item['question'][:60]}...")
        docs = retrieval_service.retrieve(item["question"], use_hyde=True)
        contexts = [d.page_content for d in docs]
        results.append({
            "question": item["question"],
            "scene": item["scene"],
            "context_precision": round(compute_context_precision(item["question"], contexts), 4),
            "context_recall": round(compute_context_recall(item["ground_truth"], contexts), 4),
            "num_contexts": len(contexts),
        })

    # 汇总
    avg_p = sum(r["context_precision"] for r in results) / len(results)
    avg_r = sum(r["context_recall"] for r in results) / len(results)

    scene_stats = {}
    for r in results:
        s = r["scene"]
        scene_stats.setdefault(s, {"p": [], "r": []})
        scene_stats[s]["p"].append(r["context_precision"])
        scene_stats[s]["r"].append(r["context_recall"])

    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"  Overall Precision: {avg_p:.4f}  Recall: {avg_r:.4f}")
    for s, v in scene_stats.items():
        sp = sum(v["p"]) / len(v["p"])
        sr = sum(v["r"]) / len(v["r"])
        print(f"  [{s}] Precision: {sp:.4f}  Recall: {sr:.4f}  (n={len(v['p'])})")

    report = {
        "summary": {"context_precision": round(avg_p, 4), "context_recall": round(avg_r, 4)},
        "scene_summary": {
            s: {"context_precision": round(sum(v["p"])/len(v["p"]), 4),
                "context_recall": round(sum(v["r"])/len(v["r"]), 4),
                "count": len(v["p"])}
            for s, v in scene_stats.items()
        },
        "details": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 报告已保存: {args.output}")
    print(f"✅ 评估完成！")


if __name__ == "__main__":
    main()
