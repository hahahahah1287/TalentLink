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
            score_threshold=config.reranker.score_threshold,
            hyde_enabled=config.retrieval.hyde_enabled,
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
    print("Phase 6: 构建统一法务图 (build_legal_graph)")
    print("=" * 60)
    t0 = time.time()
    try:
        from utils import GuardrailsPipeline
        from workflows import build_legal_graph, SkillSpec
        from utils.intent_router import route_intent

        # 真实 skill：先 init 再包成 SkillSpec（与 workflow_service.py 接线一致）
        from skills.risk_clause_detector import (
            risk_clause_skill, init_skill as init_risk_skill,
        )
        from skills.compliance_check import (
            compliance_skill, init_skill as init_compliance_skill,
        )
        from skills.legal_term_explainer import (
            legal_term_skill, init_skill as init_term_skill,
        )
        from skills.case_retriever import (
            case_retriever_skill, init_skill as init_case_skill,
        )

        init_risk_skill(llm)
        init_compliance_skill(llm)
        init_term_skill(llm)
        init_case_skill(embeddings)

        # 统一接口 fn(query, law_context="") -> str；这里挂最小集合（合规 + 风险条款）
        skill_specs = {
            "risk_clause_detector": SkillSpec(
                fn=risk_clause_skill, uses_law_context=False, label="合同风险条款",
            ),
            "compliance_check": SkillSpec(
                fn=compliance_skill, uses_law_context=False, label="合规检查",
            ),
        }

        guardrails = GuardrailsPipeline()
        # build_legal_graph 内部已自动 create_contract_chain / create_legal_qa_chain
        graph = build_legal_graph(
            llm=llm,
            retrieval_service=retrieval_service,
            skill_specs=skill_specs,
            guardrails=guardrails,
        )
        print(f"  ✅ 统一法务图构建完成 ({time.time() - t0:.1f}s)")

        # ---- 用例 A：合同审查（合同特征由 route_intent 按内容判定） ----
        print("\n  --- 用例 A：合同审查 ---")
        tA = time.time()
        queryA = "这份合同的试用期约定是否合法？"
        contractA = "甲方与乙方约定：试用期为6个月，劳动合同期限为1年。"
        routeA = route_intent(queryA, contract_text=contractA)
        print(f"  🧭 has_contract={routeA['has_contract']} skills={routeA['skills']}")

        stateA = {
            "conversation_id": "test-conv-A",
            "user_id": "test-user",
            "scene": "legal",
            "query": queryA,
            "contract_text": contractA,
            "history": "",
            "has_contract": routeA["has_contract"],
            "route_skills": routeA["skills"],
            "tool_history": [],
            "skill_outputs": {},
            "status": "running",
            "guard_issues": [],
            "guard_retry": 0,
        }
        resultA = await graph.ainvoke(stateA)
        print(f"  ✅ 合同审查完成 ({time.time() - tA:.1f}s)")
        print(f"  📄 final_answer 前 300 字: {(resultA.get('final_answer') or '')[:300]}...")
        print(f"  📄 skill_outputs keys: {list((resultA.get('skill_outputs') or {}).keys())}")
        print(f"  📄 law_context 非空: {bool(resultA.get('law_context'))}")

        # ---- 用例 B：法务咨询（无合同文本） ----
        print("\n  --- 用例 B：法务咨询 ---")
        tB = time.time()
        queryB = "劳动法对加班费是怎么规定的？"
        routeB = route_intent(queryB, contract_text=None)
        print(f"  🧭 has_contract={routeB['has_contract']} skills={routeB['skills']}")

        stateB = {
            "conversation_id": "test-conv-B",
            "user_id": "test-user",
            "scene": "legal",
            "query": queryB,
            "contract_text": None,
            "history": "",
            "has_contract": routeB["has_contract"],
            "route_skills": routeB["skills"],
            "tool_history": [],
            "skill_outputs": {},
            "status": "running",
            "guard_issues": [],
            "guard_retry": 0,
        }
        resultB = await graph.ainvoke(stateB)
        print(f"  ✅ 法务咨询完成 ({time.time() - tB:.1f}s)")
        print(f"  📄 final_answer 前 300 字: {(resultB.get('final_answer') or '')[:300]}...")
        print(f"  📄 skill_outputs keys: {list((resultB.get('skill_outputs') or {}).keys())}")
        print(f"  📄 law_context 非空: {bool(resultB.get('law_context'))}")

    except Exception as e:
        print(f"  ❌ 统一法务图测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_full_workflow())
