# -*- coding: utf-8 -*-
"""
聊天记录管理模块

架构:
- Redis: 写入缓冲层 (高性能)
- MySQL: 持久化存储层
- ContextWindowManager: 智能上下文截断

特性:
- 后台异步落库 (不阻塞用户请求)
- 异步标题生成 (不阻塞首字输出)
- 基于 Token 的上下文管理
"""
import json
import time
import uuid
import threading
from typing import List, Tuple, Optional
from contextlib import contextmanager

import redis
import pymysql
from pymysql.cursors import DictCursor
from dbutils.pooled_db import PooledDB

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==================== 上下文窗口管理器 ====================

class ContextWindowManager:
    """
    基于 Token 估算的上下文窗口管理器
    
    功能:
    - 估算文本 token 数量
    - 智能截断历史对话
    - 保留最近对话的完整性
    """
    
    def __init__(
        self, 
        max_tokens: int = 2048, 
        max_turns: int = 10, 
        chars_per_token: float = 2.5, 
        reserve_tokens: int = 512
    ):
        """
        Args:
            max_tokens: 上下文窗口最大 token 数
            max_turns: 最大对话轮数
            chars_per_token: 字符/token 比例（中文约 2-2.5）
            reserve_tokens: 为模型回复预留的 token
        """
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.chars_per_token = chars_per_token
        self.reserve_tokens = reserve_tokens
        self.effective_max_tokens = max_tokens - reserve_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数量"""
        if not text:
            return 0
        return int(len(text) / self.chars_per_token)
    
    def truncate_message(self, content: str, max_chars: int = 500) -> str:
        """截断单条消息，保留首尾"""
        if len(content) <= max_chars:
            return content
        half = max_chars // 2 - 10
        return content[:half] + "\n...[中间部分已省略]...\n" + content[-half:]
    
    def manage_context(
        self, 
        messages: List[Tuple[str, str]], 
        system_prompt: str = "", 
        current_query: str = ""
    ) -> str:
        """
        智能管理上下文，确保不超出 token 限制
        
        Args:
            messages: [(role, content), ...] 历史消息列表
            system_prompt: 系统提示词
            current_query: 当前用户查询
        
        Returns:
            截断后的历史对话字符串
        """
        if not messages:
            return ""
        
        # 计算已使用的 token
        used_tokens = self.estimate_tokens(system_prompt) + self.estimate_tokens(current_query)
        available_tokens = self.effective_max_tokens - used_tokens
        
        # 按对话轮次分组
        turns = []
        current_turn = []
        for role, content in messages:
            current_turn.append((role, content))
            if role == "assistant":
                turns.append(current_turn)
                current_turn = []
        if current_turn:
            turns.append(current_turn)
        
        # 从最近的对话开始选择
        selected_turns = []
        total_tokens = 0
        
        for turn in reversed(turns[-self.max_turns:]):
            turn_text = "\n".join([f"{r}: {c}" for r, c in turn])
            turn_tokens = self.estimate_tokens(turn_text)
            
            if total_tokens + turn_tokens <= available_tokens:
                selected_turns.insert(0, turn)
                total_tokens += turn_tokens
            else:
                # 尝试截断后加入
                remaining_tokens = available_tokens - total_tokens
                if remaining_tokens > 100:
                    truncated_turn = [
                        (r, self.truncate_message(c, int(remaining_tokens * self.chars_per_token // 2))) 
                        for r, c in turn
                    ]
                    selected_turns.insert(0, truncated_turn)
                break
        
        # 拼接结果
        result_lines = []
        for turn in selected_turns:
            for role, content in turn:
                result_lines.append(f"{role}: {content}")
        
        return "\n".join(result_lines)


# ==================== 数据库连接池 ====================

class DatabasePool:
    """
    MySQL 连接池管理器
    
    使用 DBUtils 实现连接池，避免频繁创建/销毁连接。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "123456",
        database: str = "talentlink",
        charset: str = "utf8mb4"
    ):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.pool = PooledDB(
            creator=pymysql,
            maxconnections=10,
            mincached=2,
            maxcached=5,
            blocking=True,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset=charset,
            cursorclass=DictCursor,
            autocommit=True
        )
        self._initialized = True
        print(f"✅ [DB] MySQL 连接池已初始化: {host}:{port}/{database}")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = self.pool.connection()
        try:
            yield conn
        finally:
            conn.close()


# ==================== 聊天历史管理器 ====================

class ChatHistoryManager:
    """
    聊天历史管理器
    
    架构:
    - 写入: User Request -> Redis Buffer -> Background Worker -> MySQL
    - 读取: MySQL (Redis 中的数据可能有延迟)
    """
    
    def __init__(
        self, 
        llm,
        db_config: Optional[dict] = None,
        redis_config: Optional[dict] = None,
        context_config: Optional[dict] = None
    ):
        """
        Args:
            llm: LLM 实例（用于生成标题，可传 None 使用默认标题）
            db_config: MySQL 配置字典
            redis_config: Redis 配置字典
            context_config: 上下文管理配置
        """
        self.llm = llm
        
        # 1. 初始化 MySQL 连接池
        db_cfg = db_config or {}
        self.db_pool = DatabasePool(
            host=db_cfg.get("mysql_host", "localhost"),
            port=db_cfg.get("mysql_port", 3306),
            user=db_cfg.get("mysql_user", "root"),
            password=db_cfg.get("mysql_password", "123456"),
            database=db_cfg.get("mysql_database", "talentlink"),
            charset=db_cfg.get("mysql_charset", "utf8mb4")
        )
        
        # 2. 初始化 Redis
        redis_cfg = redis_config or {}
        try:
            self.redis = redis.Redis(
                host=redis_cfg.get("redis_host", "localhost"),
                port=redis_cfg.get("redis_port", 6379),
                password=redis_cfg.get("redis_password", None),
                db=redis_cfg.get("redis_db", 0),
                decode_responses=True
            )
            self.redis.ping()
            self.use_redis = True
            print("✅ [Memory] Redis 连接成功")
        except Exception as e:
            print(f"⚠️ [Memory] Redis 连接失败: {e}，降级为直接写库模式")
            self.use_redis = False
            self.redis = None
        
        self.BUFFER_KEY = "chat_msg_buffer"
        
        # 3. 初始化数据库表
        self._setup_database()
        
        # 4. 初始化上下文管理器
        ctx_cfg = context_config or {}
        self.context_manager = ContextWindowManager(
            max_tokens=ctx_cfg.get("max_tokens", 2048),
            max_turns=ctx_cfg.get("max_turns", 10),
            chars_per_token=ctx_cfg.get("chars_per_token", 2.5),
            reserve_tokens=ctx_cfg.get("reserve_tokens", 512)
        )
        
        # 5. 启动后台落库线程
        self.running = True
        if self.use_redis:
            self.sync_thread = threading.Thread(
                target=self._background_sync_worker, 
                daemon=True,
                name="RedisToMySQLSync"
            )
            self.sync_thread.start()
    
    def _setup_database(self):
        """初始化数据库表结构"""
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cursor:
                # 会话表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id VARCHAR(36) PRIMARY KEY,
                        user_id VARCHAR(64) NOT NULL,
                        title VARCHAR(255) DEFAULT '新对话',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_user_id (user_id),
                        INDEX idx_created_at (created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                
                # 消息表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id VARCHAR(36) PRIMARY KEY,
                        session_id VARCHAR(36) NOT NULL,
                        role ENUM('user', 'assistant', 'system') NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_session_id (session_id),
                        INDEX idx_created_at (created_at),
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                
                # 摘要表（保留，未来可用于长对话摘要）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_summaries (
                        session_id VARCHAR(36) PRIMARY KEY,
                        summary TEXT,
                        summarized_until TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
        
        print("✅ [Memory] 数据库表结构已初始化")
    
    def _background_sync_worker(self):
        """后台线程：将 Redis 缓冲数据持久化到 MySQL"""
        while self.running:
            try:
                if self.redis is None:
                    break
                
                # 阻塞式弹出，减少 CPU 空转
                raw_msg = self.redis.brpop(self.BUFFER_KEY, timeout=1)
                if raw_msg:
                    _, data_json = raw_msg
                    data = json.loads(data_json)
                    self._persist_message(data)
            except Exception as e:
                print(f"⚠️ [Sync] 后台落库错误: {e}")
                time.sleep(1)
    
    def _persist_message(self, data: dict):
        """将消息持久化到 MySQL"""
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO messages (message_id, session_id, role, content) 
                       VALUES (%s, %s, %s, %s)""",
                    (data['message_id'], data['session_id'], data['role'], data['content'])
                )
    
    def add_message(self, session_id: str, role: str, content: str):
        """
        添加消息（优先写 Redis，降级写 MySQL）
        """
        msg_data = {
            "message_id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        if self.use_redis and self.redis:
            try:
                self.redis.lpush(self.BUFFER_KEY, json.dumps(msg_data, ensure_ascii=False))
                return
            except Exception as e:
                print(f"⚠️ [Memory] Redis 写入失败: {e}，降级写库")
        
        # 降级直接写 MySQL
        self._persist_message(msg_data)
    
    def get_history_str(
        self, 
        session_id: str, 
        limit: int = 50, 
        system_prompt: str = "", 
        current_query: str = ""
    ) -> str:
        """
        获取上下文字符串（经过智能截断）
        
        Args:
            session_id: 会话 ID
            limit: 从数据库获取的最大消息数
            system_prompt: 系统提示词（用于 token 计算）
            current_query: 当前查询（用于 token 计算）
        
        Returns:
            截断后的历史对话字符串
        """
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT role, content FROM messages 
                       WHERE session_id = %s 
                       ORDER BY created_at ASC 
                       LIMIT %s""",
                    (session_id, limit)
                )
                rows = cursor.fetchall()
                messages = [(r['role'], r['content']) for r in rows]
        
        return self.context_manager.manage_context(messages, system_prompt, current_query)
    
    def get_or_create_session(
        self, 
        user_id: str, 
        session_id: Optional[str], 
        first_query: str
    ) -> str:
        """
        获取或创建会话
        
        如果 session_id 为 None，创建新会话并返回新 ID。
        标题生成在后台异步执行，不阻塞主线程。
        """
        if session_id:
            return session_id
        
        new_session_id = str(uuid.uuid4())
        
        # 先用默认标题创建会话（不阻塞）
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO sessions (session_id, user_id, title) VALUES (%s, %s, %s)""",
                    (new_session_id, user_id, "新对话")
                )
        
        # 后台异步生成标题
        if self.llm:
            threading.Thread(
                target=self._async_generate_title,
                args=(new_session_id, first_query),
                daemon=True,
                name=f"TitleGen-{new_session_id[:8]}"
            ).start()
        
        return new_session_id
    
    def _async_generate_title(self, session_id: str, query: str):
        """后台异步生成标题"""
        try:
            prompt = PromptTemplate.from_template(
                "根据以下用户问题，生成一个简短的对话标题（不超过15个字）：\n{query}\n标题："
            )
            chain = prompt | self.llm | StrOutputParser()
            title = chain.invoke({"query": query}).strip()
            
            # 截断过长的标题
            if len(title) > 50:
                title = title[:50] + "..."
            
            # 更新数据库
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE sessions SET title = %s WHERE session_id = %s",
                        (title, session_id)
                    )
            
            print(f"✅ [Title] 会话 {session_id[:8]}... 标题已更新: {title}")
        except Exception as e:
            print(f"⚠️ [Title] 标题生成失败: {e}")
    
    def get_user_sessions(self, user_id: str, limit: int = 20) -> List[dict]:
        """获取用户的会话列表（用于侧边栏）"""
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT session_id, title, created_at 
                       FROM sessions 
                       WHERE user_id = %s 
                       ORDER BY created_at DESC 
                       LIMIT %s""",
                    (user_id, limit)
                )
                rows = cursor.fetchall()
        
        return [
            {
                "session_id": r['session_id'],
                "title": r['title'],
                "created_at": r['created_at'].isoformat() if r['created_at'] else None
            }
            for r in rows
        ]
    
    def shutdown(self):
        """优雅关闭（确保 Redis 缓冲数据落库）"""
        self.running = False
        print("⏳ [Memory] 正在关闭，等待缓冲数据落库...")
        
        if self.use_redis and self.redis:
            # 处理剩余的 Redis 数据
            while True:
                try:
                    raw_msg = self.redis.rpop(self.BUFFER_KEY)
                    if not raw_msg:
                        break
                    data = json.loads(raw_msg)
                    self._persist_message(data)
                except:
                    break
        
        print("✅ [Memory] 已安全关闭")