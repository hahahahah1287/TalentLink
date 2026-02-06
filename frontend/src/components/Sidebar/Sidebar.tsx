import { Session } from '../../types';
import './Sidebar.css';

interface SidebarProps {
    sessions: Session[];
    currentSessionId: string | null;
    onNewChat: () => void;
    onSelectSession: (sessionId: string) => void;
}

// 格式化时间
function formatTime(dateStr: string): string {
    try {
        const date = new Date(dateStr);
        const now = new Date();
        const diff = now.getTime() - date.getTime();
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));

        if (days === 0) {
            return '今天';
        } else if (days === 1) {
            return '昨天';
        } else if (days < 7) {
            return `${days}天前`;
        } else {
            return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
        }
    } catch {
        return '';
    }
}

export function Sidebar({ sessions, currentSessionId, onNewChat, onSelectSession }: SidebarProps) {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="sidebar-logo">
                    <span className="sidebar-logo-icon">🤖</span>
                    <span>AI 智能助手</span>
                </div>
                <button className="new-chat-btn" onClick={onNewChat}>
                    <span>✨</span>
                    <span>新对话</span>
                </button>
            </div>

            <div className="sidebar-content">
                <div className="section-title">历史对话</div>

                {sessions.length === 0 ? (
                    <div className="empty-state">
                        暂无对话记录
                    </div>
                ) : (
                    <div className="session-list">
                        {sessions.map((session) => (
                            <div
                                key={session.id}
                                className={`session-item ${session.id === currentSessionId ? 'active' : ''}`}
                                onClick={() => onSelectSession(session.id)}
                            >
                                <span className="session-icon">💬</span>
                                <div className="session-info">
                                    <div className="session-title">{session.title || '新对话'}</div>
                                    <div className="session-time">{formatTime(session.updated_at)}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </aside>
    );
}
