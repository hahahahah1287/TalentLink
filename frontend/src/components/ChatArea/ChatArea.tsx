import { useRef, useEffect } from 'react';
import { Message } from '../../types';
import { MessageBubble } from '../MessageBubble/MessageBubble';
import './ChatArea.css';

interface ChatAreaProps {
    messages: Message[];
    onSuggestionClick: (text: string) => void;
}

const SUGGESTIONS = [
    {
        icon: '💼',
        title: '找工作',
        desc: '搜索最新招聘信息',
        query: '帮我查找北京的前端开发岗位',
    },
    {
        icon: '📜',
        title: '审合同',
        desc: '分析劳动合同条款',
        query: '请帮我分析这份劳动合同的试用期条款是否合规',
    },
    {
        icon: '⚖️',
        title: '法律咨询',
        desc: '了解劳动法相关问题',
        query: '试用期最长可以多久？',
    },
    {
        icon: '💬',
        title: '自由对话',
        desc: '随便聊聊',
        query: '你好，介绍一下你自己',
    },
];

export function ChatArea({ messages, onSuggestionClick }: ChatAreaProps) {
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // 自动滚动到底部
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // 显示欢迎屏幕
    if (messages.length === 0) {
        return (
            <div className="chat-area">
                <div className="chat-header">
                    <div>
                        <div className="chat-title">新对话</div>
                        <div className="chat-subtitle">开始一段新的对话</div>
                    </div>
                </div>

                <div className="welcome-screen">
                    <div className="welcome-icon">🤖</div>
                    <h1 className="welcome-title">你好，我是 AI 智能助手</h1>
                    <p className="welcome-subtitle">
                        我可以帮你搜索招聘信息、分析劳动合同、解答法律问题，或者只是聊聊天
                    </p>

                    <div className="suggestion-grid">
                        {SUGGESTIONS.map((item, index) => (
                            <div
                                key={index}
                                className="suggestion-card"
                                onClick={() => onSuggestionClick(item.query)}
                            >
                                <div className="suggestion-icon">{item.icon}</div>
                                <div className="suggestion-title">{item.title}</div>
                                <div className="suggestion-desc">{item.desc}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="chat-area">
            <div className="chat-header">
                <div>
                    <div className="chat-title">对话中</div>
                    <div className="chat-subtitle">{messages.length} 条消息</div>
                </div>
            </div>

            <div className="chat-messages">
                {messages.map((message) => (
                    <MessageBubble key={message.id} message={message} />
                ))}
                <div ref={messagesEndRef} />
            </div>
        </div>
    );
}
