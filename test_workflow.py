# -*- coding: utf-8 -*-
"""
端到端测试脚本

测试新 LangGraph workflow 的每个环节。
"""
import asyncio
import sys
import os
import time

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig


async def test_full_workflow():
    """完整链路测试"""

    print("=" * 60)
    print("Phase 1: 加载配置")
    print("=" * 60)
    config = AppConfig()
    print(f"  模型: {config.llm.model_path}")
    print(f"  知识库: {config.retrieval.knowledge_base_path}")
    print(f"  FAISS 索引: {config.retrieval.faiss_index_path}")

    print("\n" + "=" * 60)
    print("Phase 2: 加载模型 (可能需要 30-60 秒)")
    print("=" * 60)
    t0 = time.time()
    try:
        from langchain_community.chat_models import ChatLlamaCpp
        llm = ChatLlamaCpp(
            model_path=config.llm.model_path,
            n_gpu_layers=config.llm.n_gpu_layers,
            n_ctx=config.llm.n_ctx,
            temperature=config.llm.temperature,
            verbose=False,
            streaming=True,
        )
        print(f"  ✅ 模型加载完成 ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return

    print("\n" + "=" * 60)
    print("Phase 3: 测试 LLM 简单调用")
    print("=" * 60)
    t0 = time.time()
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        chain = ChatPromptTemplate.from_messages([
            ("system", "你是一个测试助手。"),
            ("user", "回答：1+1等于几？只回答数字。")
        ]) | llm | StrOutputParser()

        result = await chain.ainvoke({})
        print(f"  ✅ LLM 回答: {result.strip()} ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ❌ LLM 调用失败: {e}")
        return

    print("\n" + "=" * 60)
    print("Phase 4: 测试 Embedding + FAISS 索引构建")
    print("=" * 60)
    t0 = time.time()
    try:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=config.embedding.model_name,
            model_kwargs={'device': config.embedding.device},
            encode_kwargs={'normalize_embeddings': config.embedding.normalize_embeddings}
        )
        print(f"  ✅ Embedding 加载完成 ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ❌ Embedding 加载失败: {e}")
        return

    print("\n" + "=" * 60)
    print("Phase 5: 测试统一检索服务 (构建索引 + 检索)")
    print("=" * 60)
    t0 = time.time()
    try:
        from utils.reranker import RerankService
        from utils.query_rewriter import QueryRewriter
        from utils.retrieval_service import RetrievalService
        from langchain_community.retrievers import BM25Retriever
        try:
            from langchain.retrievers import EnsembleRetriever
        except ImportError:
            from langchain_classic.retrievers import EnsembleRetriever
        from langchain_community.vectorstores import FAISS
        from utils import parse_legal_document, annotate_documents

        # 加载 Reranker
        reranker = RerankService(
            model_name=config.reranker.model_name,
            device=config.reranker.device,
            batch_size=config.reranker.batch_size,
        )

        # 构建混合检索器
        knowledge_path = config.retrieval.knowledge_base_path
        index_path = config.retrieval.faiss_index_path
        metadata_index_path = index_path + "_with_metadata"

        if os.path.exists(metadata_index_path):
            print(f"  📦 加载已有索引...")
            vector_store = FAISS.load_local(
                metadata_index_path, embeddings,
                allow_dangerous_deserialization=True
            )
            docs = list(vector_store.docstore._dict.values())
            print(f"  📄 从索引恢复 {len(docs)} 个条款")
        else:
            print(f"  📄 首次构建索引...")
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs = parse_legal_document(text, source=knowledge_path)
            print(f"  📄 解析为 {len(docs)} 个条款")

            print(f"  🤖 元数据标注中...")
            docs = annotate_documents(docs, llm)
            print(f"  ✅ 标注完成")

            vector_store = FAISS.from_documents(docs, embeddings)
            vector_store.save_local(metadata_index_path)
            print(f"  💾 索引已保存")

        faiss_retriever = vector_store.as_retriever(
            search_kwargs={"k": config.retrieval.retrieval_k}
        )
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = config.retrieval.retrieval_k

        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[config.retrieval.bm25_weight, config.retrieval.faiss_weight]
        )

        query_rewriter = QueryRewriter(llm=llm, enabled=True)
        retrieval_service = RetrievalService(
            retriever=ensemble,
            reranker=reranker,
            query_rewriter=query_rewriter,
            top_k=config.reranker.top_k,
            rerank_enabled=config.retrieval.rerank_enabled,
        )

        # 测试检索
        test_query = "试用期工资怎么算"
        result = retrieval_service.retrieve_as_string(test_query, use_hyde=True)
        print(f"  ✅ 检索完成 ({time.time() - t0:.1f}s)")
        print(f"  📄 结果长度: {len(result)} 字符")
        print(f"  📄 前 200 字: {result[:200]}...")

    except Exception as e:
        print(f"  ❌ 检索失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("Phase 6: 测试合同审查图 (LangGraph)")
    print("=" * 60)
    t0 = time.time()
    try:
        from utils import GuardrailsPipeline
        from workflows import create_contract_review_graph

        guardrails = GuardrailsPipeline()
        contract_graph = create_contract_review_graph(llm, retrieval_service, guardrails)

        state = {
            "query": "试用期工资有什么规定？",
            "contract_text": "甲方同意在试用期内支付乙方工资为每月3000元。",
            "history": "",
            "tool_history": [],
            "policy_flags": [],
            "status": "running",
        }

        result = await contract_graph.ainvoke(state)
        print(f"  ✅ 合同审查完成 ({time.time() - t0:.1f}s)")
        print(f"  📄 状态: {result.get('status')}")
        print(f"  📄 答案前 300 字: {result.get('final_answer', '')[:300]}...")

    except Exception as e:
        print(f"  ❌ 合同审查失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Phase 7: 测试研究型任务图 (LangGraph)")
    print("=" * 60)
    t0 = time.time()
    try:
        from workflows import create_research_agent_graph

        research_graph = create_research_agent_graph(llm, retrieval_service, guardrails)

        state = {
            "query": "劳动法对加班费是怎么规定的？",
            "history": "",
            "tool_history": [],
            "search_results": [],
            "policy_flags": [],
            "status": "running",
        }

        result = await research_graph.ainvoke(state)
        print(f"  ✅ 研究型任务完成 ({time.time() - t0:.1f}s)")
        print(f"  📄 状态: {result.get('status')}")
        print(f"  📄 计划: {result.get('plan')}")
        print(f"  📄 答案前 300 字: {result.get('final_answer', '')[:300]}...")

    except Exception as e:
        print(f"  ❌ 研究型任务失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_full_workflow())
