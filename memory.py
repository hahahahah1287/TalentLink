# memory.py
import sqlite3
import uuid
import traceback
from typing import List, Tuple, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ContextWindowManager:
    """
    上下文窗口管理器：智能管理对话历史，防止超出 LLM 的上下文长度限制。
    
    策略：
    1. 滑动窗口：保留最近 N 轮对话
    2. Token 估算：基于字符数粗略估算 token 数量
    3. 摘要压缩：对过长历史生成摘要（可选）
    4. 重要性权重：系统消息和最近消息优先保留
    """
    
    def __init__(self, 
                 max_tokens: int = 2048,
                 max_turns: int = 10,
                 chars_per_token: float = 2.5,  # 中文约 1.5-2 字符/token，英文约 4 字符/token
                 reserve_tokens: int = 512):     # 为新回复预留的 token 数
        """
        Args:
            max_tokens: 上下文窗口最大 token 数
            max_turns: 最大对话轮数（一问一答算一轮）
            chars_per_token: 字符到 token 的转换比例
            reserve_tokens: 为模型输出预留的 token 数
        """
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.chars_per_token = chars_per_token
        self.reserve_tokens = reserve_tokens
        self.effective_max_tokens = max_tokens - reserve_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """粗略估算文本的 token 数量"""
        if not text:
            return 0
        return int(len(text) / self.chars_per_token)
    
    def truncate_message(self, content: str, max_chars: int = 500) -> str:
        """截断过长的单条消息，保留头尾"""
        if len(content) <= max_chars:
            return content
        half = max_chars // 2 - 10
        return content[:half] + "\n...[中间部分已省略]...\n" + content[-half:]
    
    def manage_context(self, 
                       messages: List[Tuple[str, str]], 
                       system_prompt: str = "",
                       current_query: str = "") -> str:
        """
        智能管理上下文，返回优化后的历史字符串。
        
        Args:
            messages: [(role, content), ...] 按时间顺序排列的消息列表
            system_prompt: 系统提示词（需要计入 token 预算）
            current_query: 当前用户查询（需要计入 token 预算）
            
        Returns:
            优化后的历史对话字符串
        """
        if not messages:
            return ""
        
        # 1. 计算已用 token 预算
        used_tokens = self.estimate_tokens(system_prompt) + self.estimate_tokens(current_query)
        available_tokens = self.effective_max_tokens - used_tokens
        
        # 2. 按轮次限制 + Token 限制进行筛选
        # 将消息按轮次分组（user + assistant 为一轮）
        turns = []
        current_turn = []
        for role, content in messages:
            current_turn.append((role, content))
            if role == "assistant":
                turns.append(current_turn)
                current_turn = []
        if current_turn:  # 处理未完成的轮次
            turns.append(current_turn)
        
        # 3. 从最新的轮次开始，逆向选择
        selected_turns = []
        total_tokens = 0
        
        for turn in reversed(turns[-self.max_turns:]):  # 最多取最近 max_turns 轮
            turn_text = "\n".join([f"{r}: {c}" for r, c in turn])
            turn_tokens = self.estimate_tokens(turn_text)
            
            if total_tokens + turn_tokens <= available_tokens:
                selected_turns.insert(0, turn)  # 插入到开头，保持时间顺序
                total_tokens += turn_tokens
            else:
                # Token 预算不足，尝试截断最旧的这一轮
                remaining_tokens = available_tokens - total_tokens
                if remaining_tokens > 100:  # 至少保留一点内容
                    truncated_turn = [(r, self.truncate_message(c, int(remaining_tokens * self.chars_per_token // 2))) 
                                      for r, c in turn]
                    selected_turns.insert(0, truncated_turn)
                break
        
        # 4. 格式化输出
        result_lines = []
        for turn in selected_turns:
            for role, content in turn:
                result_lines.append(f"{role}: {content}")
        
        return "\n".join(result_lines)


class ChatHistoryManager:
    """
    负责管理所有的会话历史、消息记录以及自动生成会话标题。
    底层使用 SQLite 存储。
    """
    def __init__(self, db_path: str, llm, context_config: Optional[dict] = None):
        self.db_path = db_path
        self.llm = llm  # 注入 LLM 实例，用于生成标题
        self._setup_database()
        
        # 初始化上下文管理器
        config = context_config or {}
        self.context_manager = ContextWindowManager(
            max_tokens=config.get("max_tokens", 2048),
            max_turns=config.get("max_turns", 10),
            chars_per_token=config.get("chars_per_token", 2.5),
            reserve_tokens=config.get("reserve_tokens", 512)
        )

    def _get_conn(self):
        """获取数据库连接上下文"""
        return sqlite3.connect(self.db_path)

    def _setup_database(self):
        """初始化数据库表结构"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # 1. 会话表 (Sessions): 存储会话元数据
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            # 2. 消息表 (Messages): 存储具体的对话内容
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL, 
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )""")
            # 3. 会话摘要表 (Summaries): 存储压缩后的历史摘要
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                summarized_until TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )""")
            conn.commit()

    def _generate_session_title(self, initial_query: str) -> str:
        """核心功能：调用 LLM 为新对话生成一个不超过8个字的标题"""
        try:
            title_prompt = PromptTemplate.from_template(
                "任务：为对话生成标题。\n用户第一句话：\"{query}\"\n要求：不超过8个字，概括核心意图，不要引号，不要标点。\n标题："
            )
            title_chain = title_prompt | self.llm | StrOutputParser()
            title = title_chain.invoke({"query": initial_query}).strip()
            return title if title else "新对话"
        except Exception as e:
            print(f"[Memory Warning] 标题生成失败: {e}")
            return "新对话"

    def get_or_create_session(self, user_id: str, session_id: str | None, first_query: str) -> str:
        """
        会话管理的入口：
        - 如果提供了 session_id，直接返回。
        - 如果没提供，创建新会话 ID，生成新标题，存入数据库。
        """
        if session_id:
            return session_id
        
        new_session_id = str(uuid.uuid4())
        title = self._generate_session_title(first_query)
        
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, user_id, title) VALUES (?, ?, ?)",
                (new_session_id, user_id, title)
            )
            conn.commit()
        return new_session_id

    def add_message(self, session_id: str, role: str, content: str):
        """插入一条聊天记录"""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO messages (message_id, session_id, role, content) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), session_id, role, content)
            )
            conn.commit()

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        """获取消息列表 [(role, content), ...]"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if limit:
                cursor.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?", 
                    (session_id, limit)
                )
                rows = cursor.fetchall()
                return list(reversed(rows))  # 反转回时间顺序
            else:
                cursor.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC", 
                    (session_id,)
                )
                return cursor.fetchall()

    def get_history_str(self, session_id: str, 
                        limit: Optional[int] = None,
                        system_prompt: str = "",
                        current_query: str = "") -> str:
        """
        获取优化后的历史字符串，用于喂给 LLM 作为上下文。
        
        Args:
            session_id: 会话 ID
            limit: 最大消息数限制（可选，会与 context_manager 配合使用）
            system_prompt: 系统提示词（用于计算 token 预算）
            current_query: 当前查询（用于计算 token 预算）
        """
        # 1. 获取原始消息
        messages = self.get_messages(session_id, limit=limit or 50)  # 最多取 50 条
        
        if not messages:
            return ""
        
        # 2. 使用上下文管理器优化
        return self.context_manager.manage_context(
            messages=messages,
            system_prompt=system_prompt,
            current_query=current_query
        )

    def get_user_sessions(self, user_id: str) -> list[dict]:
        """获取用户的会话列表（用于前端侧边栏展示）"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT session_id, title, created_at FROM sessions WHERE user_id = ? ORDER BY created_at DESC", 
                (user_id,)
            )
            return [{"session_id": r[0], "title": r[1], "created_at": r[2]} for r in cursor.fetchall()]

    def save_session_summary(self, session_id: str, summary: str):
        """保存会话摘要（用于长对话压缩）"""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO session_summaries (session_id, summary, summarized_until) 
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (session_id, summary))
            conn.commit()

    def get_session_summary(self, session_id: str) -> Optional[str]:
        """获取会话摘要"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT summary FROM session_summaries WHERE session_id = ?", 
                (session_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None