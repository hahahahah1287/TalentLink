# -*- coding: utf-8 -*-
"""
LangGraph Workflow Service

替代旧的 UnifiedAgentService，使用 LangGraph 状态图编排业务流程。

架构：
- 底层能力（LLM, Embedding, Retriever, Reranker）保持不变
- 业务流程由 LangGraph StateGraph 编排
- 不做内部 Router，由 API 参数决定进入哪个 workflow
- 普通聊天不进 LangGraph，直接 LLM 调用
"""
import os
import time
import traceback
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
try:
    from langchain_experimental.chat_models import ChatLlamaCpp
except ImportError:
    from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from config import AppConfig
from utils import (
    RerankService,
    CircuitBreaker,
    CircuitBreakerOpenError,
    QueryRewriter,
    SemanticCache,
    RetrievalService,
    WorkflowCheckpoint,
    parse_legal_document,
    annotate_documents,
)
from utils.tool_call_parser import ToolCallParserLLM
from memory import ChatHistoryManager
from workflows import create_contract_review_graph, create_research_agent_graph
from skills.registry import SkillRegistry
from skills.web_search import web_search, job_search, web_search_skill, job_search_skill
from skills.risk_clause_detector import risk_clause_skill, init_skill as init_risk_skill
from skills.compliance_check import compliance_skill, init_skill as init_compliance_skill
from skills.legal_term_explainer import legal_term_skill
from skills.statute_checker import statute_skill
from workflows.research_agent import make_react_step_executor


class WorkflowService:
    """
    LangGraph Workflow Service

    职责：
    - 初始化所有底层组件
    - 创建 LangGraph workflow 实例
    - 提供流式对话接口（根据 scene 参数路由到不同 workflow）
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()

        # --- 1. 加载 LLM ---
        print(f"📦 [WorkflowService] 加载模型: {self.config.llm.model_path}...")
        self.llm = ChatLlamaCpp(
            model_path=self.config.llm.model_path,
            n_gpu_layers=self.config.llm.n_gpu_layers,
            n_ctx=self.config.llm.n_ctx,
            temperature=self.config.llm.temperature,
            verbose=self.config.llm.verbose,
            streaming=True,
        )

        # 包装 LLM：自动解析 content 中的 tool call XML → tool_calls 属性
        # 只影响 create_react_agent（Review Agent、复杂 skill ReAct），
        # Plan-and-Execute 链路使用原始 self.llm，不受影响。
        self.agent_llm = ToolCallParserLLM(self.llm)

        # --- 2. 加载 Embedding ---
        print(f"📦 [WorkflowService] 加载 Embedding...")
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.config.embedding.model_name,
            model_kwargs={'device': self.config.embedding.device},
            encode_kwargs={'normalize_embeddings': self.config.embedding.normalize_embeddings}
        )

        # --- 3. 加载 Reranker ---
        print(f"📦 [WorkflowService] 加载 Reranker...")
        self.reranker = RerankService(
            model_name=self.config.reranker.model_name,
            device=self.config.reranker.device,
            batch_size=self.config.reranker.batch_size
        )

        # --- 4. 初始化混合检索器 ---
        self.legal_retriever = self._setup_hybrid_retriever()

        # --- 5. 初始化统一检索服务 ---
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

        # --- 6. 初始化历史管理器 ---
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

        # --- 7. 初始化安全检测器 ---
        from services import PromptInjectionDetector
        self.injection_detector = PromptInjectionDetector(
            llm=self.llm, enable_llm_detection=True
        )

        # --- 9. 初始化语义缓存 ---
        redis_client = self.history_manager.redis if self.history_manager.use_redis else None
        self.semantic_cache = SemanticCache(
            embeddings=self.embeddings,
            redis_client=redis_client,
            similarity_threshold=0.93,
            ttl=3600,
        )

        # --- 9.1 初始化 Workflow Checkpoint 缓存 ---
        checkpoint_ttl = getattr(self.config, 'checkpoint_ttl', 3600)
        self.checkpoint = WorkflowCheckpoint(
            redis_client=redis_client,
            ttl=checkpoint_ttl,
            enabled=checkpoint_ttl > 0,
        )

        # --- 10. 初始化 LLM 驱动的 Skills ---
        init_risk_skill(self.llm)
        init_compliance_skill(self.llm)

        # --- 11. 初始化 Skill Registry ---
        self.registry = SkillRegistry()
        # 简单 skill：直接调用，无需推理
        self.registry.register("web_search", web_search_skill, "联网搜索最新信息（查最新政策、新闻、行业动态等）", category="search", complexity="simple")
        self.registry.register("job_search", job_search_skill, "搜索招聘信息（查职位、薪资、公司招聘等）", category="search", complexity="simple")
        self.registry.register("legal_term_explainer", legal_term_skill, "法律术语解释（解释合同或法律中的专业术语，如经济补偿金、竞业限制等）", complexity="simple")
        self.registry.register("statute_checker", statute_skill, "时效计算器（计算仲裁/诉讼时效是否届满，需要提供事件发生时间）", complexity="simple")
        # 复杂 skill：ReAct Agent 执行，可自主调用检索工具
        self.registry.register("risk_clause_detector", risk_clause_skill, "风险条款识别（分析合同中的高风险条款，如试用期违规、违约金过高等）", complexity="complex")
        self.registry.register("compliance_check", compliance_skill, "劳动法合规检查（检查用工场景是否合法，如加班、解除合同、社保等）", complexity="complex")
        print(f"📦 [WorkflowService] 已注册 {len(self.registry)} 个 skills: {self.registry}")

        # --- 12. 创建 ReAct 执行器（用于复杂 step） ---
        # 使用 agent_llm（带 tool call 解析），让 ReAct Agent 正确调度工具
        self.react_executor = make_react_step_executor(
            llm=self.agent_llm,
            retrieval_service=self.retrieval_service,
            registry=self.registry,
            extra_tools=[web_search, job_search],
        )

        # --- 13. 创建 LangGraph workflows ---
        # agent_llm 用于 Review Agent（create_react_agent）
        # 原始 self.llm 用于 Plan-and-Execute 链路（文本生成，不依赖 tool_calls）
        print("📦 [WorkflowService] 创建 LangGraph workflows...")
        self.contract_graph = create_contract_review_graph(
            llm=self.agent_llm,
            retrieval_service=self.retrieval_service,
        )
        self.research_graph = create_research_agent_graph(
            llm=self.agent_llm,
            retrieval_service=self.retrieval_service,
            registry=self.registry,
            react_executor=self.react_executor,
        )

        print("✅ [WorkflowService] 所有组件初始化完成")

    def _setup_hybrid_retriever(self):
        """初始化混合检索器（与原 UnifiedAgentService 相同）"""
        knowledge_path = self.config.retrieval.knowledge_base_path
        index_path = self.config.retrieval.faiss_index_path
        metadata_index_path = index_path + "_with_metadata"

        if not os.path.exists(knowledge_path):
            print(f"⚠️ [Retriever] 知识库文件不存在: {knowledge_path}，使用占位文档")
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
            print(f"📄 [Retriever] 结构化解析法律文档...")
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs = parse_legal_document(text, source=knowledge_path)
            print(f"✅ [Retriever] 已解析为 {len(docs)} 个条款")
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


    async def process_request_stream(
        self,
        user_id: str,
        session_id: Optional[str],
        query: str,
        scene: str = "chat",
        contract_text: Optional[str] = None,
    ):
        """
        流式处理用户请求

        根据 scene 参数路由到不同 workflow：
        - "contract" → 合同审查图（直接 RAG 链路）
        - 其他       → 研究型 Agent（Planner 自动规划工具）

        Args:
            user_id: 用户 ID
            session_id: 会话 ID（新对话传 None）
            query: 用户查询
            scene: 业务场景 ("contract" | 其他)
            contract_text: 合同文本（仅 contract 场景需要）

        Yields:
            SSE 格式的响应数据
        """
        # 1. 安全检测
        if not self.injection_detector.is_safe(query):
            yield "data: 您的请求包含非法指令，已被安全网关拦截。\n\n"
            yield 'event: end\ndata: {}\n\n'
            return

        # 2. 会话管理
        current_session_id = self.history_manager.get_or_create_session(
            user_id, session_id, query
        )

        # 3. 语义缓存查询
        cache_hit, cached_response = self.semantic_cache.get(query)
        if cache_hit and cached_response:
            print(f"🎯 [Cache] 语义缓存命中")
            yield f"data: {cached_response}\n\n"
            self.history_manager.add_message(current_session_id, "user", query)
            self.history_manager.add_message(current_session_id, "assistant", cached_response)
            yield f'event: end\ndata: {{"session_id": "{current_session_id}"}}\n\n'
            return

        # 3.1 Workflow Checkpoint 缓存查询
        checkpoint_key_scene = scene if scene in ("contract", "research") else "research"
        cached_result = self.checkpoint.get(checkpoint_key_scene, query)
        if cached_result:
            full_response = cached_result.get("final_answer", "")
            if full_response:
                yield f"data: {full_response}\n\n"
                self.history_manager.add_message(current_session_id, "user", query)
                self.history_manager.add_message(current_session_id, "assistant", full_response)
                yield f'event: end\ndata: {{"session_id": "{current_session_id}"}}\n\n'
                return

        # 4. 获取对话历史
        history_str = self.history_manager.get_history_str(
            current_session_id,
            limit=20,
            system_prompt="你是一个有用的AI助手。",
            current_query=query,
        )

        # 5. 记录用户消息
        self.history_manager.add_message(current_session_id, "user", query)

        # 6. 构建初始 state
        initial_state = {
            "conversation_id": current_session_id,
            "user_id": user_id,
            "scene": scene,
            "query": query,
            "contract_text": contract_text,
            "history": history_str,
            "tool_history": [],
            "search_results": [],
            "policy_flags": [],
            "status": "running",
            "review_status": "approve",
            "review_issues": [],
            "review_retry": 0,
        }

        # 7. 根据 scene 路由到对应 workflow
        full_response = ""
        result = {}
        try:
            if scene == "contract":
                yield "data: [正在进行合同审查（检索法律条文 → 对比分析 → 合规检查）...]\n\n"
                result = await self.contract_graph.ainvoke(initial_state)
                full_response = result.get("final_answer", "")
            else:
                yield "data: [正在为您搜索相关信息...]\n\n"
                result = await self.research_graph.ainvoke(initial_state)
                full_response = result.get("final_answer", "")

            yield f"data: {full_response}\n\n"

            # 7.1 写入 Workflow Checkpoint 缓存
            if result and full_response and len(full_response.strip()) > 10:
                self.checkpoint.set(checkpoint_key_scene, query, result)

        except Exception as e:
            full_response = f"系统内部错误: {str(e)}"
            print(traceback.format_exc())
            yield f"data: {full_response}\n\n"

        # 8. 写入语义缓存
        if len(full_response) > 50:
            self.semantic_cache.put(query, full_response)

        # 10. 存档
        self.history_manager.add_message(current_session_id, "assistant", full_response)
        yield f'event: end\ndata: {{"session_id": "{current_session_id}"}}\n\n'

    def get_system_metrics(self) -> dict:
        """获取系统运行指标"""
        return {
            "semantic_cache": self.semantic_cache.get_metrics(),
            "checkpoint": self.checkpoint.get_metrics(),
        }

    def shutdown(self):
        """优雅关闭服务"""
        print("⏳ [WorkflowService] 正在关闭...")
        self.history_manager.shutdown()
        print("✅ [WorkflowService] 已关闭")
