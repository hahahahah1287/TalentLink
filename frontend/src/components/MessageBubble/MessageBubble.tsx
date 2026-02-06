import { Message } from '../../types';
import './MessageBubble.css';

interface MessageBubbleProps {
    message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
    const roleClass = message.role;
    const streamingClass = message.isStreaming ? 'streaming' : '';

    return (
        <div className={`message ${roleClass} ${streamingClass}`}>
            <div className="message-avatar">
                {message.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-content">
                <div className="message-bubble">
                    {message.content || (message.isStreaming ? '' : '...')}
                </div>
                <div className="message-time">
                    {message.timestamp.toLocaleTimeString('zh-CN', {
                        hour: '2-digit',
                        minute: '2-digit'
                    })}
                </div>
            </div>
        </div>
    );
}
