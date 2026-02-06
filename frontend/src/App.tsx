import { useState } from 'react';
import { useChat } from './hooks/useChat';
import { Sidebar } from './components/Sidebar/Sidebar';
import { ChatArea } from './components/ChatArea/ChatArea';
import { InputArea } from './components/InputArea/InputArea';
import './App.css';

function App() {
    const {
        messages,
        sessions,
        currentSessionId,
        isLoading,
        error,
        sendMessage,
        createNewSession,
        switchSession,
    } = useChat();

    const [pendingQuery, setPendingQuery] = useState<string | null>(null);

    // 处理推荐问题点击
    const handleSuggestionClick = (query: string) => {
        sendMessage(query);
    };

    // 处理发送消息
    const handleSend = (query: string, contractText?: string) => {
        sendMessage(query, contractText);
        setPendingQuery(null);
    };

    return (
        <div className="app">
            <Sidebar
                sessions={sessions}
                currentSessionId={currentSessionId}
                onNewChat={createNewSession}
                onSelectSession={switchSession}
            />

            <main className="main-content">
                <ChatArea
                    messages={messages}
                    onSuggestionClick={handleSuggestionClick}
                />

                <InputArea
                    onSend={handleSend}
                    isLoading={isLoading}
                    initialValue={pendingQuery || ''}
                />
            </main>

            {error && (
                <div className="error-toast">
                    ⚠️ {error}
                </div>
            )}
        </div>
    );
}

export default App;
