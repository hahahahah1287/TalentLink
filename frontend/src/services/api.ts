// ========================================
// API 调用层
// ========================================

import { ChatRequest, HistoryResponse } from '../types';

const API_BASE = '';  // 使用 Vite 代理

/** 获取用户会话历史 */
export async function fetchHistory(userId: string): Promise<HistoryResponse> {
    const response = await fetch(`${API_BASE}/history/${userId}`);
    if (!response.ok) {
        throw new Error(`获取历史失败: ${response.statusText}`);
    }
    return response.json();
}

/** 发送消息并处理 SSE 流 */
export async function sendMessageStream(
    request: ChatRequest,
    onChunk: (text: string) => void,
    onEnd: (sessionId: string) => void,
    onError: (error: string) => void
): Promise<void> {
    try {
        const response = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            throw new Error(`请求失败: ${response.statusText}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
            throw new Error('无法读取响应流');
        }

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    onChunk(data);
                } else if (line.startsWith('event: end')) {
                    // 下一行是 end 事件的数据
                    continue;
                } else if (line.includes('"session_id"')) {
                    // 解析 session_id
                    try {
                        const match = line.match(/"session_id":\s*"([^"]+)"/);
                        if (match) {
                            onEnd(match[1]);
                        }
                    } catch {
                        // 忽略解析错误
                    }
                }
            }
        }
    } catch (error) {
        onError(error instanceof Error ? error.message : '未知错误');
    }
}
