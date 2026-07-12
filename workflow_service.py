# -*- coding: utf-8 -*-
"""
WorkflowService — 确定性法务编排服务

替代旧的 UnifiedAgentService / 双图（contract + research）架构。

整个系统收敛为**一条参数化的确定性法务 workflow**：
    安全检测(确定性关键词) → checkpoint 精确缓存(前置快路)
      → 意图路由(关键词表 + 合同内容特征) → retrieve(法条,首节点)
      → run_skills(确定性 skill) → generate(LLM 只做合成,SSE 真流式)
      → guard(确定性 GuardrailsPipeline)

设计要点：
- 把 LLM 关进笼子：判定交给 skill 的规则引擎/状态机，LLM 只合成与改写。
- 编排是确定性 DAG，不靠 9B 自由调度（无 Planner / ReAct / recursion_limit）。
- 检索固化为图首节点，所有法务问答都先查法条。
- SSE 逐 token 真流式（generate 节点的合成链 astream）。
- 精确 checkpoint 缓存（无语义近义命中——法务要精确）；带历史的追问跳过缓存。
"""
import asyncio
import hashlib
import json
import os
import time
import traceback
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever

from config import AppConfig
from utils import (
    RerankService,
    QueryRewriter,
    RetrievalService,
    WorkflowCheckpoint,
    GuardrailsPipeline,
    PromptInjectionDetector,
    annotate_documents,
)
from utils.legal_corpus import corpus_fingerprint, parse_legal_corpus, resolve_knowledge_base_paths
from utils.embeddings import TransformerEmbeddings
from utils.intent_router import route_intent
from utils.llm_factory import create_chat_llm, describe_llm_backend, llm_supports_parallel_requests
from utils.tools.contract_tools import create_contract_chain, create_legal_qa_chain
from memory import ChatHistoryManager
from workflows import build_legal_graph, SkillSpec
from workflows.legal_graph import (
    build_generate_inputs,
    make_retrieve_node,
    make_run_skills_node,
    make_guard_node,
    check_guard_result,
)

from skills.registry import SkillRegistry
from skills.risk_clause_detector import risk_clause_skill, init_skill as init_risk_skill
from skills.compliance_check import compliance_skill, init_skill as init_compliance_skill
from skills.legal_term_explainer import legal_term_skill, init_skill as init_term_skill
from skills.statute_checker import statute_skill
from skills.case_retriever import case_retriever_skill, init_skill as init_case_skill
from skills.web_search import web_search_skill


def _sse(payload: dict) -> str:
    """统一 SSE data 帧编码（中文不转义）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class WorkflowService:
    """确定性法务编排服务。"""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()

        # --- 1. 初始化 LLM 客户端（仅用于合成 / HyDE / 摘要，不做编排决策） ---
        print(f"📦 [WorkflowService] 初始化 LLM 后端: {describe_llm_backend(self.config.llm)}...")
        self.llm = create_chat_llm(self.config.llm, streaming=True)

        # --- 2. 加载 Embedding ---
        print(f"📦 [WorkflowService] 加载 Embedding...")
        self.embeddings = TransformerEmbeddings(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            normalize_embeddings=self.config.embedding.normalize_embeddings,
            local_files_only=True,
        )

        # --- 3. 加载 Reranker ---
        print(f"📦 [WorkflowService] 加载 Reranker...")
        self.reranker = RerankService(
            model_name=self.config.reranker.model_name,
            device=self.config.reranker.device,
            batch_size=self.config.reranker.batch_size
        )

        # --- 4. 混合检索器 + 统一检索服务 ---
        self.legal_retriever = self._setup_hybrid_retriever()
        self.query_rewriter = QueryRewriter(llm=self.llm, enabled=True)
        self.retrieval_service = RetrievalService(
            retriever=self.legal_retriever,
            reranker=self.reranker,
            query_rewriter=self.query_rewriter,
            top_k=self.config.reranker.top_k,
            rerank_enabled=self.config.retrieval.rerank_enabled,
            score_threshold=self.config.reranker.score_threshold,
            hyde_enabled=self.config.retrieval.hyde_enabled,
        )

        # --- 5. 历史管理器 ---
        db_config = {
            "mysql_host": self.config.database.mysql_host,
            "mysql_port": self.config.database.mysql_port,
            "mysql_user": self.config.database.mysql_user,
            "mysql_password": self.config.database.mysql_password,
            "mysql_database": self.config.database.mysql_database,
        }
        redis_config = {
            "redis_host": self.config.database.redis_host,
            "redis_port": self.config.database.redis_port,
            "redis_password": self.config.database.redis_password,
        }
        context_config = {
            "max_tokens": self.config.context.max_tokens,
            "max_turns": self.config.context.max_turns,
            "chars_per_token": self.config.context.chars_per_token,
            "reserve_tokens": self.config.context.reserve_tokens,
        }
        self.history_manager = ChatHistoryManager(
            llm=self.llm,
            db_config=db_config,
            redis_config=redis_config,
            context_config=context_config,
        )

        # --- 6. 安全检测器（默认只用确定性关键词层，不每请求跑 9B） ---
        self.injection_detector = PromptInjectionDetector(
            llm=self.llm, enable_llm_detection=False
        )

        # --- 7. 精确结果缓存（无语义近义命中，法务要精确） ---
        redis_client = self.history_manager.redis if self.history_manager.use_redis else None
        checkpoint_ttl = getattr(self.config, 'checkpoint_ttl', 3600)
        self.checkpoint = WorkflowCheckpoint(
            redis_client=redis_client,
            ttl=checkpoint_ttl,
            enabled=checkpoint_ttl > 0,
        )

        # --- 8. 初始化 skill（复用已加载的 llm / embedding） ---
        init_risk_skill(self.llm)
        init_compliance_skill(self.llm)
        init_term_skill(self.llm)            # 术语 grounding 接线 (2A/M2)
        init_case_skill(self.embeddings)     # 案例检索接线，复用 bge (2A/M1)

        # --- 9. Skill Registry（声明每个 skill 是否消费 law_context） ---
        self.registry = SkillRegistry()
        self.registry.register(
            "risk_clause_detector", risk_clause_skill,
            "风险条款识别（抽取式 NLP + 规则引擎）",
            uses_law_context=False, label="合同风险条款",
        )
        self.registry.register(
            "compliance_check", compliance_skill,
            "劳动法合规检查（决策表 / Rules-as-Data）",
            uses_law_context=False, label="合规检查",
        )
        self.registry.register(
            "legal_term_explainer", legal_term_skill,
            "法律术语解释（知识图谱 + grounding 抗幻觉）",
            uses_law_context=False, label="术语解释",
        )
        self.registry.register(
            "statute_checker", statute_skill,
            "时效计算器（领域状态机 + 时间区间推理）",
            uses_law_context=False, label="时效计算",
        )
        self.registry.register(
            "case_retriever", case_retriever_skill,
            "相似案例检索（语义检索 + MMR 多样性）",
            uses_law_context=False, label="相似案例",
        )
        self.registry.register(
            "web_search", web_search_skill,
            "最新政策/地方标准外部检索（仅作 freshness 线索）",
            uses_law_context=False, label="外部 freshness 检索",
        )
        print(f"📦 [WorkflowService] 已注册 {len(self.registry)} 个 skills: {self.registry}")

        # --- 10. 组装 SkillSpec 表 + 法务节点（SSE 真流式直接驱动节点） ---
        self.guardrails = GuardrailsPipeline()
        llm_bound_skills = set() if llm_supports_parallel_requests(self.config.llm) else {
            "risk_clause_detector",
            "compliance_check",
            "legal_term_explainer",
        }
        skill_specs = {
            name: SkillSpec(
                fn=self.registry.get_skill_fn(name),
                uses_law_context=self.registry.uses_law_context(name),
                label=self.registry.get_label(name),
                llm_bound=name in llm_bound_skills,
            )
            for name in self.registry.get_all_skill_names()
        }
        self.skill_specs = skill_specs

        # 合成链（SSE astream 逐 token 用）
        self._contract_chain = create_contract_chain(self.llm)
        self._legal_qa_chain = create_legal_qa_chain(self.llm)

        # 确定性法务节点（检索→run_skills→guard 直接复用，generate 走 astream）
        self._retrieve_node = make_retrieve_node(self.retrieval_service)
        self._run_skills_node = make_run_skills_node(skill_specs)
        self._guard_node = make_guard_node(self.guardrails)

        # 同时编译一份完整图，供非流式调用 / 测试 / 审计（与流式同一批节点逻辑）
        print("📦 [WorkflowService] 构建统一法务图...")
        self.legal_graph = build_legal_graph(
            llm=self.llm,
            retrieval_service=self.retrieval_service,
            skill_specs=skill_specs,
            guardrails=self.guardrails,
        )

        print("✅ [WorkflowService] 所有组件初始化完成")

    def _setup_hybrid_retriever(self):
        """初始化混合检索器（BM25 + FAISS）。"""
        knowledge_paths = resolve_knowledge_base_paths(self.config.retrieval)
        corpus_id = corpus_fingerprint(knowledge_paths)
        index_path = self.config.retrieval.faiss_index_path
        metadata_index_path = index_path + "_with_metadata"

        if not knowledge_paths:
            print("⚠️ [Retriever] 未找到知识库文件，使用占位文档")
            docs = [Document(page_content="暂无法律数据")]
            vector_store = FAISS.from_documents(docs, self.embeddings)
        elif os.path.exists(metadata_index_path):
            print(f"📦 [Retriever] 加载带元数据的索引: {metadata_index_path}")
            vector_store = FAISS.load_local(
                metadata_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            docs = list(vector_store.docstore._dict.values())
            print(f"📄 [Retriever] 从索引恢复 {len(docs)} 个条款文档")
        else:
            print(f"📄 [Retriever] 结构化解析法律文档: {knowledge_paths}")
            docs = parse_legal_corpus(knowledge_paths)
            for doc in docs:
                doc.metadata["corpus_version"] = self.config.retrieval.corpus_version
                doc.metadata["corpus_fingerprint"] = corpus_id
            print(f"✅ [Retriever] 已解析为 {len(docs)} 个条款，corpus_fingerprint={corpus_id}")
            print(f"🤖 [Retriever] 开始元数据标注（共 {len(docs)} 个条款）...")
            docs = annotate_documents(docs, self.llm)
            print(f"✅ [Retriever] 元数据标注完成")
            vector_store = FAISS.from_documents(docs, self.embeddings)
            vector_store.save_local(metadata_index_path)
            print(f"💾 [Retriever] 索引已保存: {metadata_index_path}")

        faiss_retriever = vector_store.as_retriever(
            search_kwargs={"k": self.config.retrieval.retrieval_k}
        )
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = self.config.retrieval.retrieval_k

        return EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[self.config.retrieval.bm25_weight, self.config.retrieval.faiss_weight],
        )

    def _build_cache_fingerprint(self, query: str, contract_text: Optional[str], route: dict) -> dict:
        """构建结果缓存指纹，避免同 query 不同合同/路由串答案。"""
        contract_hash = hashlib.sha256((contract_text or "").encode("utf-8")).hexdigest()[:16]
        knowledge_paths = resolve_knowledge_base_paths(self.config.retrieval)
        return {
            "query": query,
            "contract_hash": contract_hash,
            "has_contract": route.get("has_contract"),
            "route_skills": route.get("skills", []),
            "model_path": self.config.llm.model_path,
            "faiss_index_path": self.config.retrieval.faiss_index_path,
            "corpus_version": self.config.retrieval.corpus_version,
            "corpus_fingerprint": corpus_fingerprint(knowledge_paths),
            "knowledge_base_paths": knowledge_paths,
            "route_version": route.get("route_version"),
            "workflow_version": "legal-dag-evidence-v2",
            "guard_version": "evidence-citation-v1",
        }

    # ==================== 请求处理（SSE 真流式） ====================

    async def process_request_stream(
        self,
        user_id: str,
        session_id: Optional[str],
        query: str,
        scene: str = "legal",
        contract_text: Optional[str] = None,
    ):
        """
        流式处理用户请求（单一确定性法务流水线）。

        scene 仅作语义标注：所有非闲聊请求都走同一条法务图，
        是否合同审查由意图路由按内容特征自动判定，不靠前端声明。

        Yields:
            SSE data 帧（统一 json.dumps 编码，逐 token 真流式）。
        """
        # 1. 安全检测（确定性关键词层）：同时检查 query 与合同文本
        safety_text = "\n".join([query or "", contract_text or ""])
        if not self.injection_detector.is_safe(safety_text):
            yield _sse({"text": "您的请求包含非法指令，已被安全网关拦截。"})
            yield "event: end\n" + _sse({})
            return

        # 2. 会话管理（含 session 归属校验）
        try:
            current_session_id = self.history_manager.get_or_create_session(
                user_id, session_id, query
            )
        except PermissionError:
            yield _sse({"text": "无权访问该会话。", "error": True})
            yield "event: end\n" + _sse({})
            return
        except ValueError:
            yield _sse({"text": "会话不存在或已失效。", "error": True})
            yield "event: end\n" + _sse({})
            return

        # 3. 获取历史（带历史的追问不复用缓存，避免上下文相关答案被错误命中）
        history_str = self.history_manager.get_history_str(
            current_session_id,
            limit=20,
            system_prompt="你是一名劳动法务助手。",
            current_query=query,
        )
        has_history = bool(history_str and history_str.strip())

        # 4. 确定性意图路由（关键词表 + 合同内容特征），缓存 key 依赖路由结果
        route = route_intent(query, contract_text=contract_text)
        cache_fingerprint = self._build_cache_fingerprint(query, contract_text, route)
        print(
            f"🧭 [Router] has_contract={route['has_contract']} "
            f"(来源={route['contract_source']}) skills={route['skills']}"
        )

        # 5. 精确 checkpoint 缓存（仅无历史的首轮提问走缓存）
        if not has_history:
            cached_result = self.checkpoint.get("legal", cache_fingerprint)
            if cached_result:
                full_response = cached_result.get("final_answer", "")
                if full_response:
                    print("🎯 [Cache] checkpoint 精确命中")
                    yield _sse({"text": full_response})
                    self.history_manager.add_message(current_session_id, "user", query)
                    self.history_manager.add_message(current_session_id, "assistant", full_response)
                    yield "event: end\n" + _sse({"session_id": current_session_id})
                    return

        # 6. 记录用户消息
        self.history_manager.add_message(current_session_id, "user", query)

        # 7. 构建初始 state
        initial_state = {
            "conversation_id": current_session_id,
            "user_id": user_id,
            "scene": scene,
            "query": query,
            "contract_text": contract_text,
            "history": history_str,
            "has_contract": route["has_contract"],
            "route_skills": route["skills"],
            "route_result": route,
            "tool_history": [{
                "step": "route",
                "tool": "intent_router",
                "route_version": route.get("route_version"),
                "confidence": route.get("confidence"),
                "skills": route.get("skills"),
                "trace": route.get("trace"),
                "timestamp": time.time(),
            }],
            "skill_outputs": {},
            "status": "running",
            "guard_issues": [],
            "guard_retry": 0,
        }

        full_response = ""
        try:
            full_response = ""
            async for chunk in self._run_pipeline_stream(initial_state):
                # chunk 为 ("token", text) 或 ("final", state)
                kind, payload = chunk
                if kind == "token":
                    full_response += payload
                    yield _sse({"text": payload})
                elif kind == "status":
                    yield _sse({"status": payload})
                elif kind == "final":
                    result_state = payload
                    final_answer = result_state.get("final_answer", "")
                    if final_answer:
                        yield _sse({"text": final_answer})
                        full_response = final_answer

                    # 8. 写入精确缓存（仅无历史、有效答案）
                    if (not has_history and final_answer
                            and len(final_answer.strip()) > 10):
                        self.checkpoint.set("legal", cache_fingerprint, {"final_answer": final_answer,
                                                                          **{k: result_state.get(k) for k in
                                                                             ("law_context", "evidence_items", "tool_history")}})
        except Exception as e:
            full_response = f"系统内部错误: {str(e)}"
            print(traceback.format_exc())
            yield _sse({"text": full_response, "error": True})

        # 9. 归档（有效性门槛：错误串/空答案不入库）
        if full_response and not full_response.startswith("系统内部错误") and len(full_response.strip()) > 5:
            self.history_manager.add_message(current_session_id, "assistant", full_response)
        else:
            print("ℹ️ [Archive] 答案无效（错误或过短），跳过归档")
        yield "event: end\n" + _sse({"session_id": current_session_id})

    async def _run_pipeline_stream(self, state: dict):
        """
        执行法务流水线，generate 阶段逐 token yield。

        前序确定性节点（retrieve / run_skills）通过 ainvoke 子图执行（无 LLM 流式需求），
        generate 用合成链 astream 逐 token 产出，guard 跑确定性管线。
        若 guard 触发一次 revise，则再 astream 一轮。

        Yields: ("status", str) | ("token", str) | ("final", state)
        """
        # --- 阶段 A：检索 + 并行 specialist agents（确定性，无需流式） ---
        yield ("status", "检索法条…" if not state["has_contract"] else "审查合同（检索→风险/合规）…")

        upd = await self._retrieve_node(state)
        state.update(upd)
        yield ("status", "正在并行运行法务专家分析…")
        upd = await self._run_skills_node(state)
        state.update(upd)

        # --- 阶段 B：合成（guarded streaming：先缓冲草稿，guard 通过后再输出） ---
        for attempt in range(2):
            yield ("status", "正在合成答案…" if attempt == 0 else "正在重新合成答案…")
            inputs, chain = build_generate_inputs(
                state, self._contract_chain, self._legal_qa_chain, self.skill_specs
            )
            draft_parts = []
            async for token in chain.astream(inputs):
                # astream 产出 AIMessageChunk 或 str（取决于链尾），统一取文本
                text = token if isinstance(token, str) else getattr(token, "content", "")
                if text:
                    draft_parts.append(text)
            draft = "".join(draft_parts)
            state["draft_answer"] = draft
            state["final_answer"] = draft
            state["guard_issues"] = []

            # guard：确定性输出防护（引用校验/脱敏/免责声明）
            guard_upd = await self._guard_node(state)
            state.update(guard_upd)

            if check_guard_result(state) == "revise" and attempt == 0:
                yield ("status", "校验到引用问题，正在修正…")
                continue
            break

        state["status"] = "done"
        yield ("final", state)

    def get_system_metrics(self) -> dict:
        """系统运行指标。"""
        return {
            "checkpoint": self.checkpoint.get_metrics(),
        }

    def shutdown(self):
        """优雅关闭。"""
        print("⏳ [WorkflowService] 正在关闭...")
        self.history_manager.shutdown()
        print("✅ [WorkflowService] 已关闭")
