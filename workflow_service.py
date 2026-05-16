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
    GuardrailsPipeline,
    RetrievalService,
    parse_legal_document,
    annotate_documents,
)
from memory import ChatHistoryManager
from workflows import create_contract_review_graph, create_research_agent_graph


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

        # --- 7. 初始化 Guardrails ---
        self.guardrails = GuardrailsPipeline()

        # --- 8. 初始化安全检测器 ---
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

        # --- 10. 创建 LangGraph workflows ---
        print("📦 [WorkflowService] 创建 LangGraph workflows...")
        self.contract_graph = create_contract_review_graph(
            llm=self.llm,
            retrieval_service=self.retrieval_service,
            guardrails=self.guardrails,
        )
        self.research_graph = create_research_agent_graph(
            llm=self.llm,
            retrieval_service=self.retrieval_service,
            guardrails=self.guardrails,
        )

        # --- 11. 创建通用对话链 ---
        self.general_chain = self._create_general_chain()

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

    def _create_general_chain(self):
        """创建通用对话链"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有用、友好的 AI 助手。请简洁、准确地回答用户问题。"),
            ("user", """
【历史对话】
{history}

【用户新问题】
{query}

请回答：""")
        ])
        return prompt | self.llm | StrOutputParser()

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
        - "contract" → 合同审查图
        - "job"      → 求职推荐图
        - 其他       → 直接 LLM 对话

        Args:
            user_id: 用户 ID
            session_id: 会话 ID（新对话传 None）
            query: 用户查询
            scene: 业务场景 ("contract" | "job" | "chat")
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
        }

        # 7. 根据 scene 路由到对应 workflow
        full_response = ""
        try:
            if scene == "contract":
                yield "data: [正在进行合同审查（检索法律条文 → 对比分析 → 合规检查）...]\n\n"
                result = await self.contract_graph.ainvoke(initial_state)
                full_response = result.get("final_answer", "")

            elif scene == "job":
                yield "data: [正在为您搜索相关信息...]\n\n"
                result = await self.research_graph.ainvoke(initial_state)
                full_response = result.get("final_answer", "")

            else:  # chat — 不进 LangGraph，直接 LLM
                async for chunk in self.general_chain.astream({
                    "history": history_str,
                    "query": query,
                }):
                    full_response += chunk
                    yield f"data: {chunk}\n\n"

            # 对于 workflow 场景，一次性输出结果
            if scene in ("contract", "job"):
                yield f"data: {full_response}\n\n"

        except Exception as e:
            full_response = f"系统内部错误: {str(e)}"
            print(traceback.format_exc())
            yield f"data: {full_response}\n\n"

        # 8. Guardrails（chat 场景已有的 guardrails 逻辑）
        if scene not in ("contract", "job"):
            # workflow 场景已在图内做 policy_check，这里只处理 chat
            guardrails_result = self.guardrails.run(
                full_response, context={"intent": scene}
            )
            full_response = guardrails_result["output"]
            if guardrails_result["modified"]:
                for guard_info in guardrails_result["guards_triggered"]:
                    if guard_info["guard"] == "免责声明":
                        yield f"data: {self.guardrails.guards[-1].DISCLAIMER}\n\n"

        # 9. 写入语义缓存
        if scene != "chat" and len(full_response) > 50:
            self.semantic_cache.put(query, full_response)

        # 10. 存档
        self.history_manager.add_message(current_session_id, "assistant", full_response)
        yield f'event: end\ndata: {{"session_id": "{current_session_id}"}}\n\n'

    def get_system_metrics(self) -> dict:
        """获取系统运行指标"""
        return {
            "semantic_cache": self.semantic_cache.get_metrics(),
        }

    def shutdown(self):
        """优雅关闭服务"""
        print("⏳ [WorkflowService] 正在关闭...")
        self.history_manager.shutdown()
        print("✅ [WorkflowService] 已关闭")
