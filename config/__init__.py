# -*- coding: utf-8 -*-
"""
TalentLink 应用配置模块

集中管理所有配置项，包括：
- LLM 模型路径
- Embedding 模型配置
- 数据库连接配置
- 上下文窗口参数
- LLM-as-Judge 配置（从 .env 读取，避免 key 泄露）
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# 从仓库根目录 .env 加载环境变量（.env 已在 .gitignore 中，不会进 git）
load_dotenv()


@dataclass
class LLMConfig:
    """大语言模型配置"""
    # GGUF 模型路径
    model_path: str = "./Qwen3.5-9B-IQ4_XS.gguf"
    # 上下文窗口
    n_ctx: int = 4096
    # GPU 层数 (-1 表示全部卸载到 GPU)
    n_gpu_layers: int = -1

    # 参数
    temperature: float = 0.1
    verbose: bool = False

@dataclass
class JudgeConfig:
    """LLM-as-Judge 配置（可选层，对应 RAGAS_TEST_PLAN.md §17.3）

    从 .env 读取 API key，避免写进代码或推到 GitHub。
    未配置时 is_configured=False，确定性评测核心仍可独立运行。
    注意：刻意不放入 to_dict()，防止 key 被日志记录。
    """
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", ""))
    judge_model: str = field(default_factory=lambda: os.environ.get("JUDGE_MODEL", "gemini-2.5-flash"))

    @property
    def is_configured(self) -> bool:
        return bool(self.gemini_api_key) or bool(
            self.openai_api_key and self.openai_base_url
        )


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
    # 建议在 .env 设 MYSQL_PASSWORD 覆盖此默认值，避免硬编码进 git
    mysql_password: str = field(default_factory=lambda: os.environ.get("MYSQL_PASSWORD", "123456"))
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
    knowledge_base_path: str = "labor_law.txt"
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
        print(config.judge.is_configured)   # .env 是否填了 key
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # Workflow Checkpoint 缓存 TTL（秒），0 表示禁用
    checkpoint_ttl: int = 3600

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典，方便日志记录（刻意不含 judge，避免泄露 key）"""
        return {
            "llm": {
                "model_path": self.llm.model_path,
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
            }
        }
