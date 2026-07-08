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

from workflow_service import WorkflowService


# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """对话请求"""
    user_id: str
    session_id: Optional[str] = None  # 新对话传 None
    query: str
    scene: str = "legal"              # "legal"（法务问答/合同审查统一入口）| "chat"
    # 合同文本可选：是否走合同审查由后端按内容特征判定，不依赖此字段（无 OCR）
    contract_text: Optional[str] = None


class HistoryResponse(BaseModel):
    """会话列表响应"""
    sessions: List[dict]


# ==================== 全局服务 ====================

workflow_service: Optional[WorkflowService] = None


# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    - 启动时：初始化服务
    - 关闭时：优雅关闭（确保数据落库）
    """
    global workflow_service

    # 启动
    print("🚀 [App] 正在初始化服务...")
    workflow_service = WorkflowService()
    print("✅ [App] 服务已就绪")

    yield

    # 关闭
    if workflow_service:
        workflow_service.shutdown()


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
    主对话接口（单一确定性法务流水线）。

    所有非闲聊请求都走同一条法务图：
      安全检测 → 缓存 → 意图路由 → 检索法条 → 确定性 skill → 合成 → 输出防护。
    是否合同审查由后端按内容特征自动判定（不依赖前端 scene/contract_text 声明）。

    客户端应使用 EventSource 或 fetch 读取 SSE 流（data 帧为 JSON：{"text": "..."}）。
    """
    return StreamingResponse(
        workflow_service.process_request_stream(
            user_id=req.user_id,
            session_id=req.session_id,
            query=req.query,
            scene=req.scene,
            contract_text=req.contract_text,
        ),
        media_type="text/event-stream"
    )


@app.get("/history/{user_id}", response_model=HistoryResponse, summary="获取会话历史")
async def get_history(user_id: str):
    """获取用户的会话列表（用于侧边栏）"""
    sessions = workflow_service.history_manager.get_user_sessions(user_id)
    return HistoryResponse(sessions=sessions)


@app.get("/health", summary="健康检查")
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "service": "TalentLink AI",
        "components": {
            "llm": workflow_service is not None,
            "database": workflow_service.history_manager.db_pool is not None if workflow_service else False,
            "redis": workflow_service.history_manager.use_redis if workflow_service else False,
        }
    }


@app.get("/metrics", summary="系统运行指标")
async def system_metrics():
    """
    暴露 checkpoint 缓存命中率等运行时指标。
    可对接 Prometheus/Grafana 监控。
    """
    if workflow_service:
        return workflow_service.get_system_metrics()
    return {"error": "service not initialized"}


@app.post("/cache/invalidate", summary="清除 Workflow Checkpoint 缓存")
async def invalidate_cache(workflow: Optional[str] = None):
    """
    手动清除 checkpoint 缓存。

    Args:
        workflow: 指定工作流类型（默认 "legal"），不传则清除全部
    """
    if workflow_service:
        workflow_service.checkpoint.invalidate_all(workflow)
        return {"status": "ok", "cleared_workflow": workflow or "all"}
    return {"error": "service not initialized"}


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