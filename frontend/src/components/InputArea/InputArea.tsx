import { useState, useRef, KeyboardEvent } from 'react';
import './InputArea.css';

interface InputAreaProps {
    onSend: (query: string, contractText?: string) => void;
    isLoading: boolean;
    initialValue?: string;
}

export function InputArea({ onSend, isLoading, initialValue = '' }: InputAreaProps) {
    const [message, setMessage] = useState(initialValue);
    const [showContractInput, setShowContractInput] = useState(false);
    const [contractText, setContractText] = useState('');
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // 自动调整高度
    const adjustHeight = () => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
        }
    };

    const handleSend = () => {
        if (!message.trim() || isLoading) return;

        onSend(message.trim(), showContractInput ? contractText : undefined);
        setMessage('');
        setContractText('');
        setShowContractInput(false);

        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    // 如果有初始值，清除它
    if (initialValue && message === initialValue) {
        handleSend();
    }

    return (
        <div className="input-area">
            <div className="input-container">
                <div className="input-actions">
                    <button
                        className={`action-btn ${showContractInput ? 'active' : ''}`}
                        onClick={() => setShowContractInput(!showContractInput)}
                        title="添加合同文本"
                    >
                        📎
                    </button>
                </div>

                <div className="input-wrapper">
                    {showContractInput && (
                        <div className="contract-input-area">
                            <label className="contract-label">
                                <span>📜</span>
                                <span>合同文本（用于审合同功能）</span>
                            </label>
                            <textarea
                                className="contract-textarea"
                                placeholder="粘贴合同内容..."
                                value={contractText}
                                onChange={(e) => setContractText(e.target.value)}
                            />
                        </div>
                    )}

                    <textarea
                        ref={textareaRef}
                        className="message-input"
                        placeholder="输入消息... (Shift+Enter 换行)"
                        value={message}
                        onChange={(e) => {
                            setMessage(e.target.value);
                            adjustHeight();
                        }}
                        onKeyDown={handleKeyDown}
                        rows={1}
                    />
                </div>

                <button
                    className={`send-btn ${isLoading ? 'loading' : ''}`}
                    onClick={handleSend}
                    disabled={!message.trim() || isLoading}
                    title="发送消息"
                >
                    {isLoading ? '⏳' : '➤'}
                </button>
            </div>

            <div className="input-hint">
                按 Enter 发送 · 支持求职搜索、合同审核、法律咨询
            </div>
        </div>
    );
}
