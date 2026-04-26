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
from typing import Dict, Any, Optional


@dataclass
class LLMConfig:
    """大语言模型配置"""
    # GGUF 模型路径
    model_path: str = "./Qwen3.5-9B-Q5_K_M.gguf"
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
    top_k: int = 3


@dataclass
class RetrievalConfig:
    """检索配置"""
    faiss_index_path: str = "faiss_legal_index"
    knowledge_base_path: str = "labor_law.txt"
    bm25_weight: float = 0.5
    faiss_weight: float = 0.5
    retrieval_k: int = 5  # 粗排返回数量
    rerank_enabled: bool = True


@dataclass
class AppConfig:
    """
    应用总配置
    
    用法:
        config = AppConfig()
        print(config.llm.main_model_path)
        print(config.database.mysql_host)
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # 分词器类型: legal / contract / general
    text_splitter_type: str = "legal"

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典，方便日志记录"""
        return {
            "llm": {
                "model_name": self.llm.model_name,
                "base_url": self.llm.base_url,
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
            }
        }
