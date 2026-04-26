# -*- coding: utf-8 -*-
"""
RAG 评估管线 (Retrieval Evaluation Pipeline)

用于量化验证不同检索策略的效果，输出 Recall@K, MRR, Hit Rate 等指标。
对比四种策略：
  1. 纯 BM25
  2. 纯 FAISS (Dense Retrieval)
  3. BM25 + FAISS (Hybrid/Ensemble)
  4. BM25 + FAISS + Rerank (完整管线)

运行方式:
  python eval_rag.py

注意:
  - 本脚本只需要 Embedding 模型和 Reranker 模型，不需要 LLM (GGUF)
  - 评估数据集使用手工标注的 question→relevant_doc 对
"""
import os
import sys
import time
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

# --- 确保能导入项目模块 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from utils import RerankService, create_text_splitter


# ==================== 评估数据集 ====================

# 手工标注的评估集：(查询, 期望召回的关键内容片段列表)
# 这些片段只需是文档中的部分文本，用于判断是否命中
EVAL_DATASET: List[Dict] = [
    {
        "query": "试用期工资标准是多少",
        "expected_keywords": ["试用期", "工资", "不得低于", "百分之八十"],
    },
    {
        "query": "劳动合同应该包含哪些内容",
        "expected_keywords": ["劳动合同", "应当具备", "条款"],
    },
    {
        "query": "加班费怎么计算",
        "expected_keywords": ["加班", "工资报酬", "百分之"],
    },
    {
        "query": "什么情况下可以解除劳动合同",
        "expected_keywords": ["解除", "劳动合同"],
    },
    {
        "query": "工伤认定标准",
        "expected_keywords": ["工伤", "认定"],
    },
    {
        "query": "年假天数规定",
        "expected_keywords": ["年休假", "天"],
    },
    {
        "query": "经济补偿金怎么算",
        "expected_keywords": ["经济补偿", "月工资"],
    },
    {
        "query": "劳动仲裁时效是多久",
        "expected_keywords": ["仲裁", "时效"],
    },
    {
        "query": "未签劳动合同的法律后果",
        "expected_keywords": ["未", "书面劳动合同", "二倍"],
    },
    {
        "query": "社会保险包括哪些",
        "expected_keywords": ["社会保险", "养老", "医疗"],
    },
    {
        "query": "女职工产假规定",
        "expected_keywords": ["女职工", "产假"],
    },
    {
        "query": "最低工资标准规定",
        "expected_keywords": ["最低工资"],
    },
    {
        "query": "劳动者休息休假权利",
        "expected_keywords": ["休息", "休假"],
    },
    {
        "query": "劳务派遣的限制条件",
        "expected_keywords": ["劳务派遣"],
    },
    {
        "query": "集体合同的订立程序",
        "expected_keywords": ["集体合同"],
    },
]


# ==================== 评估指标 ====================

@dataclass
class EvalMetrics:
    """评估结果"""
    strategy_name: str
    recall_at_k: float = 0.0    # 召回率 @ K
    mrr: float = 0.0            # Mean Reciprocal Rank
    hit_rate: float = 0.0       # 命中率（至少一个文档命中）
    avg_latency_ms: float = 0.0 # 平均检索延迟
    details: List[dict] = field(default_factory=list)


def _is_relevant(doc_content: str, expected_keywords: List[str]) -> bool:
    """判断文档是否与期望关键词匹配（至少命中 50% 的关键词）"""
    hit_count = sum(1 for kw in expected_keywords if kw in doc_content)
    return hit_count >= max(1, len(expected_keywords) // 2)


def evaluate_retriever(
    retriever,
    dataset: List[Dict],
    top_k: int = 5,
    strategy_name: str = "unknown",
    reranker=None,
    rerank_top_k: int = 3,
) -> EvalMetrics:
    """
    评估检索器性能

    Args:
        retriever: 检索器实例
        dataset: 评估数据集
        top_k: 检索返回数量
        strategy_name: 策略名称
        reranker: 可选的重排序服务
        rerank_top_k: 重排序后保留数量
    """
    metrics = EvalMetrics(strategy_name=strategy_name)
    total_recall = 0.0
    total_mrr = 0.0
    total_hits = 0

    for item in dataset:
        query = item["query"]
        expected_keywords = item["expected_keywords"]

        start_time = time.time()

        # 检索
        try:
            docs = retriever.get_relevant_documents(query)
            if hasattr(retriever, 'k'):
                docs = docs[:top_k]
            else:
                docs = docs[:top_k]
        except Exception as e:
            print(f"  ⚠️ 检索失败: {query} -> {e}")
            metrics.details.append({"query": query, "error": str(e)})
            continue

        # 重排序（如果提供）
        if reranker and docs:
            try:
                docs = reranker.rerank(query, docs, top_k=rerank_top_k)
            except Exception as e:
                print(f"  ⚠️ 重排序失败: {query} -> {e}")

        elapsed_ms = (time.time() - start_time) * 1000

        # 计算指标
        is_hit = False
        first_relevant_rank = 0
        relevant_count = 0

        for i, doc in enumerate(docs):
            if _is_relevant(doc.page_content, expected_keywords):
                relevant_count += 1
                if not is_hit:
                    is_hit = True
                    first_relevant_rank = i + 1

        # Recall: 是否召回了相关文档
        recall = 1.0 if relevant_count > 0 else 0.0
        total_recall += recall

        # MRR: 第一个相关文档的排名倒数
        mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0
        total_mrr += mrr

        # Hit Rate
        if is_hit:
            total_hits += 1

        metrics.details.append({
            "query": query,
            "recall": recall,
            "mrr": mrr,
            "hit": is_hit,
            "relevant_docs": relevant_count,
            "total_docs": len(docs),
            "latency_ms": round(elapsed_ms, 2),
        })

        metrics.avg_latency_ms += elapsed_ms

    n = len(dataset)
    if n > 0:
        metrics.recall_at_k = total_recall / n
        metrics.mrr = total_mrr / n
        metrics.hit_rate = total_hits / n
        metrics.avg_latency_ms /= n

    return metrics


def print_results_table(results: List[EvalMetrics]):
    """打印评估结果对比表"""
    print("\n" + "=" * 80)
    print("📊 RAG 检索策略评估报告")
    print("=" * 80)
    print(
        f"{'策略':<30} {'Recall@K':>10} {'MRR':>10} {'Hit Rate':>10} {'Latency':>12}"
    )
    print("-" * 80)

    for m in results:
        print(
            f"{m.strategy_name:<30} "
            f"{m.recall_at_k:>9.1%} "
            f"{m.mrr:>9.3f} "
            f"{m.hit_rate:>9.1%} "
            f"{m.avg_latency_ms:>10.1f}ms"
        )

    print("-" * 80)

    # 最佳策略
    best = max(results, key=lambda x: x.recall_at_k)
    print(f"\n🏆 最佳策略: {best.strategy_name} (Recall@K = {best.recall_at_k:.1%})")

    # 各查询详情
    print("\n📋 逐查询详情 (最佳策略):")
    print(f"{'查询':<30} {'Recall':>8} {'MRR':>8} {'命中':>6} {'延迟':>10}")
    print("-" * 70)
    for d in best.details:
        hit_mark = "✅" if d.get("hit") else "❌"
        print(
            f"{d['query']:<28} "
            f"{d.get('recall', 0):>7.0%} "
            f"{d.get('mrr', 0):>7.3f} "
            f"{hit_mark:>6} "
            f"{d.get('latency_ms', 0):>8.1f}ms"
        )


def main():
    """主评估流程"""
    knowledge_path = "labor_law.txt"

    if not os.path.exists(knowledge_path):
        print(f"❌ 知识库文件不存在: {knowledge_path}")
        print("请确保 labor_law.txt 在当前目录下")
        sys.exit(1)

    print("📦 加载 Embedding 模型 (BAAI/bge-m3)...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✅ Embedding 加载完成")

    # --- 加载并切分文档 ---
    print("📄 加载知识库...")
    text_splitter = create_text_splitter(splitter_type="legal")
    loader = TextLoader(knowledge_path, encoding="utf-8")
    raw_docs = loader.load()
    docs = text_splitter.split_documents(raw_docs)
    print(f"✅ 已切分为 {len(docs)} 个文档片段")

    # --- 构建检索器 ---
    print("🔨 构建 FAISS 索引...")
    faiss_index_path = "faiss_eval_index"
    if os.path.exists(faiss_index_path):
        vector_store = FAISS.load_local(
            faiss_index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(faiss_index_path)
    print("✅ FAISS 索引就绪")

    retrieval_k = 5

    # 1. 纯 BM25
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = retrieval_k

    # 2. 纯 FAISS
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_k})

    # 3. BM25 + FAISS (Ensemble)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )

    # 4. 加载 Reranker
    print("📦 加载 Reranker (BAAI/bge-reranker-v2-m3)...")
    try:
        reranker = RerankService(
            model_name="BAAI/bge-reranker-v2-m3", device="cpu", batch_size=32
        )
        has_reranker = reranker.model is not None
    except Exception as e:
        print(f"⚠️ Reranker 加载失败: {e}")
        reranker = None
        has_reranker = False

    # --- 执行评估 ---
    print("\n🚀 开始评估...\n")
    results = []

    print("[1/4] 评估: 纯 BM25")
    results.append(
        evaluate_retriever(bm25_retriever, EVAL_DATASET, top_k=retrieval_k, strategy_name="BM25")
    )

    print("[2/4] 评估: 纯 FAISS (Dense)")
    results.append(
        evaluate_retriever(faiss_retriever, EVAL_DATASET, top_k=retrieval_k, strategy_name="FAISS (Dense)")
    )

    print("[3/4] 评估: BM25 + FAISS (Hybrid)")
    results.append(
        evaluate_retriever(
            ensemble_retriever, EVAL_DATASET, top_k=retrieval_k, strategy_name="BM25 + FAISS (Hybrid)"
        )
    )

    if has_reranker:
        print("[4/4] 评估: BM25 + FAISS + Rerank")
        results.append(
            evaluate_retriever(
                ensemble_retriever,
                EVAL_DATASET,
                top_k=retrieval_k,
                strategy_name="BM25 + FAISS + Rerank",
                reranker=reranker,
                rerank_top_k=3,
            )
        )
    else:
        print("[4/4] 跳过: Reranker 不可用")

    # --- 输出报告 ---
    print_results_table(results)

    # 保存 JSON 报告
    report_path = "eval_rag_report.json"
    report_data = []
    for m in results:
        report_data.append({
            "strategy": m.strategy_name,
            "recall_at_k": round(m.recall_at_k, 4),
            "mrr": round(m.mrr, 4),
            "hit_rate": round(m.hit_rate, 4),
            "avg_latency_ms": round(m.avg_latency_ms, 2),
            "details": m.details,
        })

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n📄 评估报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
