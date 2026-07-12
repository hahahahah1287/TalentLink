# -*- coding: utf-8 -*-
"""
TalentLink 应用配置模块

集中管理所有配置项，包括：
- LLM 模型路径
- Embedding 模型配置
- 数据库连接配置
- 上下文窗口参数
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class LLMConfig:
    """大语言模型配置"""
    # 后端类型：local=进程内 ChatLlamaCpp；server=OpenAI-compatible llama_cpp.server
    backend: str = "local"
    # GGUF 模型路径
    model_path: str = "./Qwen3.5-9B-IQ4_XS.gguf"
    # llama_cpp.server / OpenAI-compatible 服务配置
    server_base_url: str = "http://127.0.0.1:8000/v1"
    server_model: str = "local-9b"
    server_api_key: str = "not-needed"
    server_timeout: float = 600.0
    # 上下文窗口
    n_ctx: int = 4096
    # GPU 层数 (-1 表示全部卸载到 GPU)
    n_gpu_layers: int = -1

    # 参数
    temperature: float = 0.1
    verbose: bool = False

@dataclass
class EmbeddingConfig:
    """向量嵌入模型配置"""
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"  # 节省显存，CPU 够快
    normalize_embeddings: bool = True


@dataclass
class DatabaseConfig:
    """数据库连接配置"""
    # MySQL 配置
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "123456"
    mysql_database: str = "talentlink"
    mysql_charset: str = "utf8mb4"
    
    # Redis 配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None  # 无密码
    redis_db: int = 0


@dataclass
class ContextConfig:
    """上下文窗口管理配置"""
    max_tokens: int = 2048
    max_turns: int = 10
    chars_per_token: float = 2.5  # 中文约 2-2.5
    reserve_tokens: int = 512


@dataclass
class RerankerConfig:
    """重排序模型配置"""
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"  # 节省显存
    batch_size: int = 32
    top_k: int = 5  # rerank 后保留数量
    score_threshold: float = 0.3  # rerank 分数阈值，低于此值的文档被过滤


@dataclass
class RetrievalConfig:
    """检索配置"""
    faiss_index_path: str = "faiss_legal_index"
    knowledge_base_path: str = "labor_law.txt"  # 兼容旧单文件配置
    knowledge_base_paths: List[str] = field(default_factory=lambda: [
        "labor_law.txt",
        "labor_law_contaract.txt",
        "Paid_Annual_Leave.txt",
        "work_related_injury.txt",
        "data/legal_sources/**/*.txt",
    ])
    corpus_version: str = "legal-corpus-v1"
    bm25_weight: float = 0.4  # BM25 权重（法律文本关键词匹配更重要）
    faiss_weight: float = 0.6  # FAISS 权重
    retrieval_k: int = 8  # 粗排返回数量（给 reranker 更多候选）
    rerank_enabled: bool = True
    hyde_enabled: bool = True  # HyDE 查询改写开关，统一控制所有路径


@dataclass
class AppConfig:
    """
    应用总配置
    
    用法:
        config = AppConfig()
        print(config.llm.model_path)
        print(config.database.mysql_host)
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # Workflow Checkpoint 缓存 TTL（秒），0 表示禁用
    checkpoint_ttl: int = 3600

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典，方便日志记录"""
        return {
            "llm": {
                "backend": self.llm.backend,
                "model_path": self.llm.model_path,
                "server_base_url": self.llm.server_base_url,
                "server_model": self.llm.server_model,
                "n_ctx": self.llm.n_ctx,
            },
            "embedding": {
                "model": self.embedding.model_name,
                "device": self.embedding.device,
            },
            "database": {
                "mysql_host": self.database.mysql_host,
                "redis_host": self.database.redis_host,
            },
            "reranker": {
                "model": self.reranker.model_name,
                "device": self.reranker.device,
            },
            "retrieval": {
                "faiss_index_path": self.retrieval.faiss_index_path,
                "knowledge_base_paths": self.retrieval.knowledge_base_paths,
                "corpus_version": self.retrieval.corpus_version,
            }
        }
