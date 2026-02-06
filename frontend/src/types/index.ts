// ========================================
// TypeScript 类型定义
// ========================================

/** 消息类型 */
export interface Message {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;
}

/** 会话类型 */
export interface Session {
    id: string;
    title: string;
    updated_at: string;
}

/** 聊天请求 */
export interface ChatRequest {
    user_id: string;
    session_id: string | null;
    query: string;
    contract_text?: string;
}

/** 历史响应 */
export interface HistoryResponse {
    sessions: Session[];
}

/** 聊天状态 */
export interface ChatState {
    messages: Message[];
    sessions: Session[];
    currentSessionId: string | null;
    isLoading: boolean;
    error: string | null;
}
