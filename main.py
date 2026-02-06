# main_app.py
import uvicorn
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services import UnifiedAgentService

# --- 数据模型 (Schema) ---
class ChatRequest(BaseModel):
    user_id: str
    session_id: str | None = None # 如果是新对话，传 None
    query: str
    contract_text: str | None = None # 仅审合同时需要

class HistoryResponse(BaseModel):
    sessions: list[dict]

# --- FastAPI 应用 ---
app = FastAPI(title="Local AI Agent API", version="5.0.0")

# 初始化服务 (这会触发模型加载)
agent_service = UnifiedAgentService()

@app.post("/chat/stream", summary="流式对话接口")
async def chat_stream(req: ChatRequest):
    """
    主对话接口。
    客户端应使用 EventSource 或 fetch 读取 SSE 流。
    """
    return StreamingResponse(
        agent_service.process_request_stream(
            req.user_id, req.session_id, req.query, req.contract_text
        ),
        media_type="text/event-stream"
    )

@app.get("/history/{user_id}", response_model=HistoryResponse, summary="获取侧边栏历史")
async def get_history(user_id: str):
    """获取用户的会话列表 (ID, 标题, 时间)"""
    sessions = agent_service.history_manager.get_user_sessions(user_id)
    return HistoryResponse(sessions=sessions)

if __name__ == "__main__":
    # 创建一个示例法律文件，确保第一次运行不报错
    if not os.path.exists("labor_laws.txt"):
        with open("labor_laws.txt", "w", encoding="utf-8") as f:
            f.write("《劳动合同法》第十九条：劳动合同期限三个月以上不满一年的，试用期不得超过一个月...")
    
    print("🚀 服务器启动中: http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)