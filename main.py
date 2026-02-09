# -*- coding: utf-8 -*-
"""
TalentLink API 入口

FastAPI 应用，提供流式对话和历史记录接口。
"""
# --- Fix for GLIBCXX version issue in Conda ---
import os
import sys

# 强制将系统库路径加入 LD_LIBRARY_PATH，解决 Miniconda libstdc++ 版本过旧的问题
system_lib_path = "/usr/lib/x86_64-linux-gnu"
if os.path.exists(system_lib_path):
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if system_lib_path not in current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{system_lib_path}:{current_ld}"
        # 注意：对于已经加载的动态库，os.environ 修改可能无效，
        # 但 llama_cpp 是在 import 时动态加载的，所以通常有效。
        # 如果无效，需要用户在终端执行 export LD_LIBRARY_PATH=...
        try:
            # 尝试重新加载 ctypes 以应用新的环境（虽然 Python 进程启动后很难完全变更）
            import ctypes
            ctypes.CDLL(os.path.join(system_lib_path, "libstdc++.so.6"), mode=ctypes.RTLD_GLOBAL)
        except Exception as e:
            print(f"⚠️ [System] 尝试预加载系统 libstdc++ 失败: {e}")

# ----------------------------------------------

import signal
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from services import UnifiedAgentService


# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """对话请求"""
    user_id: str
    session_id: Optional[str] = None  # 新对话传 None
    query: str
    contract_text: Optional[str] = None  # 仅审合同时需要


class HistoryResponse(BaseModel):
    """会话列表响应"""
    sessions: List[dict]


# ==================== 全局服务 ====================

agent_service: Optional[UnifiedAgentService] = None


# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    - 启动时：初始化服务
    - 关闭时：优雅关闭（确保数据落库）
    """
    global agent_service
    
    # 启动
    print("🚀 [App] 正在初始化服务...")
    agent_service = UnifiedAgentService()
    print("✅ [App] 服务已就绪")
    
    yield
    
    # 关闭
    if agent_service:
        agent_service.shutdown()


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="TalentLink AI API",
    version="6.0.0",
    description="本地化 AI 法务助手 API",
    lifespan=lifespan
)

# CORS 配置（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 路由 ====================

@app.post("/chat/stream", summary="流式对话接口")
async def chat_stream(req: ChatRequest):
    """
    主对话接口
    
    客户端应使用 EventSource 或 fetch 读取 SSE 流。
    """
    return StreamingResponse(
        agent_service.process_request_stream(
            req.user_id,
            req.session_id,
            req.query,
            req.contract_text
        ),
        media_type="text/event-stream"
    )


@app.get("/history/{user_id}", response_model=HistoryResponse, summary="获取会话历史")
async def get_history(user_id: str):
    """获取用户的会话列表（用于侧边栏）"""
    sessions = agent_service.history_manager.get_user_sessions(user_id)
    return HistoryResponse(sessions=sessions)


@app.get("/health", summary="健康检查")
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "service": "TalentLink AI",
        "components": {
            "llm": agent_service is not None,
            "database": agent_service.history_manager.db_pool is not None if agent_service else False,
            "redis": agent_service.history_manager.use_redis if agent_service else False,
        }
    }


# ==================== 入口 ====================

if __name__ == "__main__":
    # 检查知识库文件是否存在
    if not os.path.exists("labor_law.txt"):
        print("⚠️ [App] 未找到知识库文件 labor_law.txt，请准备好法律知识库")
    else:
        print("📄 [App] 已找到知识库文件: labor_law.txt")
    
    print("🚀 服务器启动中: http://0.0.0.0:8000")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境关闭热重载
        workers=1      # 单进程（LLM 占用大量内存）
    )