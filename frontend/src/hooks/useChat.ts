// ========================================
// 聊天状态管理 Hook
// ========================================

import { useState, useCallback, useEffect } from 'react';
import { Message, ChatState } from '../types';
import { fetchHistory, sendMessageStream } from '../services/api';

// 生成唯一 ID
const generateId = () => Math.random().toString(36).substring(2, 11);

// 默认用户 ID (后续可扩展登录功能)
const DEFAULT_USER_ID = 'demo-user';

export function useChat() {
    const [state, setState] = useState<ChatState>({
        messages: [],
        sessions: [],
        currentSessionId: null,
        isLoading: false,
        error: null,
    });

    const userId = DEFAULT_USER_ID;

    // 加载会话历史
    const loadHistory = useCallback(async () => {
        try {
            const { sessions } = await fetchHistory(userId);
            setState(prev => ({ ...prev, sessions }));
        } catch (error) {
            console.error('加载历史失败:', error);
        }
    }, [userId]);

    // 初始加载
    useEffect(() => {
        loadHistory();
    }, [loadHistory]);

    // 发送消息
    const sendMessage = useCallback(async (query: string, contractText?: string) => {
        if (!query.trim() || state.isLoading) return;

        // 添加用户消息
        const userMessage: Message = {
            id: generateId(),
            role: 'user',
            content: query,
            timestamp: new Date(),
        };

        // 创建 AI 消息占位
        const aiMessage: Message = {
            id: generateId(),
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            isStreaming: true,
        };

        setState(prev => ({
            ...prev,
            messages: [...prev.messages, userMessage, aiMessage],
            isLoading: true,
            error: null,
        }));

        // 发送请求
        await sendMessageStream(
            {
                user_id: userId,
                session_id: state.currentSessionId,
                query,
                contract_text: contractText,
            },
            // onChunk
            (text) => {
                setState(prev => ({
                    ...prev,
                    messages: prev.messages.map(msg =>
                        msg.id === aiMessage.id
                            ? { ...msg, content: msg.content + text }
                            : msg
                    ),
                }));
            },
            // onEnd
            (sessionId) => {
                setState(prev => ({
                    ...prev,
                    currentSessionId: sessionId,
                    messages: prev.messages.map(msg =>
                        msg.id === aiMessage.id
                            ? { ...msg, isStreaming: false }
                            : msg
                    ),
                    isLoading: false,
                }));
                // 刷新历史列表
                loadHistory();
            },
            // onError
            (error) => {
                setState(prev => ({
                    ...prev,
                    messages: prev.messages.map(msg =>
                        msg.id === aiMessage.id
                            ? { ...msg, content: `错误: ${error}`, isStreaming: false }
                            : msg
                    ),
                    isLoading: false,
                    error,
                }));
            }
        );
    }, [state.currentSessionId, state.isLoading, userId, loadHistory]);

    // 创建新对话
    const createNewSession = useCallback(() => {
        setState(prev => ({
            ...prev,
            currentSessionId: null,
            messages: [],
            error: null,
        }));
    }, []);

    // 切换会话 (后续可扩展加载历史消息)
    const switchSession = useCallback((sessionId: string) => {
        setState(prev => ({
            ...prev,
            currentSessionId: sessionId,
            messages: [],  // 暂时清空，后续可加载该会话的历史消息
            error: null,
        }));
    }, []);

    return {
        ...state,
        userId,
        sendMessage,
        createNewSession,
        switchSession,
        loadHistory,
    };
}
