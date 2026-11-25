import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function Dashboard() {
    // === 原有的 State ===
    const [files, setFiles] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);

    // === [新增] 搜尋功能的 State ===
    const [query, setQuery] = useState('');
    const [searching, setSearching] = useState(false);
    const [mode, setMode] = useState('rag'); // 'rag': streamed RAG answer, 'agent': tool-calling agent
    const [chatHistory, setChatHistory] = useState([
        { role: 'ai', text: '你好！我已經分析完文件了，有什麼我可以幫你的嗎？' }
    ]);

    // 載入文件列表 (保持不變)
    const fetchDocuments = async () => {
        try {
            const res = await axios.get(`${API_URL}/documents`);
            setFiles(res.data);
        } catch (err) {
            console.error(err);
        }
    };

    useEffect(() => {
        fetchDocuments();
        const interval = setInterval(fetchDocuments, 3000);
        return () => clearInterval(interval);
    }, []);

    // 處理上傳 (保持不變)
    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            await axios.post(`${API_URL}/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            fetchDocuments();
        } catch (err) {
            alert("Upload failed");
        } finally {
            setUploading(false);
        }
    };

    const handleSearch = async (e) => {
        if (e.key !== 'Enter' || !query.trim() || searching) return;

        const userQ = query;
        setQuery('');
        setSearching(true);

        setChatHistory(prev => [...prev, { role: 'user', text: userQ }]);

        if (mode === 'agent') {
            try {
                const res = await axios.post(`${API_URL}/agent`, { question: userQ });
                const { answer, steps, critiques, elapsed_ms } = res.data;
                setChatHistory(prev => [...prev, {
                    role: 'ai',
                    text: `${answer}\n\n⏱ ${(elapsed_ms / 1000).toFixed(1)}s ・ ${steps.length} 次工具呼叫`,
                    steps,
                    critiques,
                }]);
            } catch (err) {
                console.error(err);
                setChatHistory(prev => [...prev, { role: 'ai', text: "Agent 執行失敗，請確認 Ollama 是否已啟動。" }]);
            } finally {
                setSearching(false);
            }
            return;
        }

        try {
            // Stream the answer with Server-Sent Events: sources first, then tokens as they are generated.
            const res = await fetch(`${API_URL}/search/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: userQ, top_k: 5 }),
            });
            if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

            setChatHistory(prev => [...prev, { role: 'ai', text: '' }]);
            const setLastAiText = (text) => setChatHistory(prev => {
                const next = [...prev];
                next[next.length - 1] = { role: 'ai', text };
                return next;
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let answer = '';
            let sources = [];

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const events = buffer.split('\n\n');
                buffer = events.pop(); // keep an incomplete event for the next read
                for (const raw of events) {
                    if (!raw.startsWith('data: ')) continue;
                    const event = JSON.parse(raw.slice(6));

                    if (event.type === 'sources') {
                        sources = event.sources;
                    } else if (event.type === 'token') {
                        answer += event.text;
                        setLastAiText(answer);
                    } else if (event.type === 'error') {
                        answer += `\n\n⚠️ 生成中斷：${event.message}`;
                        setLastAiText(answer);
                    } else if (event.type === 'done') {
                        const sourceText = sources.map((doc, idx) =>
                            `[${idx + 1}] ${doc.filename}${doc.page ? ` 第 ${doc.page} 頁` : ''} (相似度: ${(doc.similarity_score * 100).toFixed(1)}%)`
                        ).join('\n');
                        const timing = `⏱ 首字 ${(event.ttft_ms / 1000).toFixed(1)}s ・ 完成 ${(event.total_ms / 1000).toFixed(1)}s`;
                        setLastAiText(sources.length
                            ? `${answer}\n\n---\n📑 參考來源：\n${sourceText}\n${timing}`
                            : answer);
                    }
                }
            }

        } catch (err) {
            console.error(err);
            setChatHistory(prev => [...prev, { role: 'ai', text: "系統處理時發生錯誤，請確認 Ollama 是否已啟動。" }]);
        } finally {
            setSearching(false);
        }
    };

    return (
        <div className="flex h-screen bg-gray-50 text-gray-800 font-sans">
            {/* Sidebar (保持不變) */}
            <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
                <div className="h-16 flex items-center px-6 border-b border-gray-200">
                    <span className="text-lg font-bold text-blue-800 tracking-tight">AI FINANCIAL</span>
                </div>

                <div className="p-4">
                    <label className={`flex items-center justify-center w-full h-10 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 transition-colors cursor-pointer shadow-sm ${uploading ? 'opacity-50 cursor-not-allowed' : ''}`}>
                        {uploading ? 'Uploading...' : 'Upload Report'}
                        <input type="file" className="hidden" onChange={handleUpload} disabled={uploading} />
                    </label>
                </div>

                <nav className="flex-1 px-2 space-y-1 overflow-y-auto">
                    <div className="px-3 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
                        Reports Repository
                    </div>
                    {files.map((file) => (
                        <button
                            key={file.id}
                            onClick={() => setSelectedFile(file)}
                            className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${selectedFile?.id === file.id ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-100'
                                }`}
                        >
                            <div className="flex-1 text-left truncate">{file.name}</div>
                            <StatusBadge status={file.status} />
                        </button>
                    ))}
                </nav>
            </aside>

            {/* Main Content */}
            <main className="flex-1 flex flex-col overflow-hidden">
                <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-8">
                    <h1 className="text-xl font-semibold text-gray-800">
                        {selectedFile ? selectedFile.name : 'System Overview'}
                    </h1>
                    <div className="flex items-center space-x-2">
                        <span className="h-2 w-2 rounded-full bg-green-500"></span>
                        <span className="text-sm text-gray-500">System Online</span>
                    </div>
                </header>

                <div className="flex-1 p-8 overflow-y-auto">
                    {!selectedFile ? (
                        <div className="flex flex-col items-center justify-center h-full text-gray-400">
                            <p className="text-lg">Select a document to view ETL details</p>
                        </div>
                    ) : (
                        // === [修改] 這裡改成兩欄式佈局 (Grid) ===
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">

                            {/* 左邊：文件詳情 (保留原本的) */}
                            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6 h-fit">
                                <h3 className="text-lg font-bold mb-4">Document Details</h3>
                                <div className="space-y-2">
                                    <p><strong>Status:</strong> <StatusBadge status={selectedFile.status} /></p>
                                    <p><strong>Upload Date:</strong> {selectedFile.date}</p>
                                    <p className="mt-4 text-sm text-gray-500 bg-gray-50 p-3 rounded">
                                        * Vector embeddings generated.
                                        <br />
                                        * Stored in PostgreSQL (pgvector).
                                    </p>
                                </div>
                            </div>

                            {/* 右邊：[新增] AI 對話視窗 */}
                            <div className="bg-white rounded-lg border border-gray-200 shadow-sm flex flex-col h-[600px]">
                                <div className="p-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                                    <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wide">
                                        AI Assistant
                                    </h3>
                                    <div className="flex text-xs border border-gray-300 rounded-lg overflow-hidden">
                                        {[['rag', 'RAG 串流'], ['agent', 'Agent（多步驟）']].map(([value, label]) => (
                                            <button
                                                key={value}
                                                onClick={() => setMode(value)}
                                                className={`px-3 py-1 ${mode === value ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-100'}`}
                                            >
                                                {label}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* 聊天紀錄區 */}
                                <div className="flex-1 p-4 overflow-y-auto space-y-3 bg-gray-50">
                                    {chatHistory.map((msg, idx) => (
                                        <ChatMessage key={idx} {...msg} />
                                    ))}
                                    {searching && (
                                        <div className="text-xs text-gray-400 text-center animate-pulse">
                                            {mode === 'agent' ? 'Agent 正在查資料與計算...' : 'Thinking...'}
                                        </div>
                                    )}
                                </div>

                                {/* 輸入框 */}
                                <div className="p-4 border-t border-gray-200 bg-white">
                                    <input
                                        type="text"
                                        value={query}
                                        onChange={(e) => setQuery(e.target.value)}
                                        onKeyDown={handleSearch}
                                        placeholder={mode === 'agent' ? '例如：毛利率比前一年增加幾個百分點？' : '輸入問題，例如：營收表現如何？'}
                                        className="w-full pl-4 pr-10 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm shadow-sm"
                                    />
                                </div>
                            </div>

                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}

// === [新增] 聊天訊息元件 ===
function ChatMessage({ role, text, steps = [], critiques = [] }) {
    const isAi = role === 'ai';
    return (
        <div className={`flex ${isAi ? 'justify-start' : 'justify-end'}`}>
            <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-sm whitespace-pre-wrap ${isAi ? 'bg-white text-gray-800 border border-gray-200' : 'bg-blue-600 text-white'
                }`}>
                {text}
                {steps.length > 0 && <AgentTrace steps={steps} critiques={critiques} />}
            </div>
        </div>
    );
}

// Agent 的執行過程：每一次工具呼叫的參數與結果，以及 harness 退回重做的原因
function AgentTrace({ steps, critiques }) {
    const icons = { search_report: '🔍', get_page: '📄', calculator: '🧮' };
    return (
        <details className="mt-3 border-t border-gray-200 pt-2 text-xs text-gray-600">
            <summary className="cursor-pointer select-none text-gray-500">查看執行過程（{steps.length} 步）</summary>
            <ol className="mt-2 space-y-2 list-none">
                {steps.map((step, i) => (
                    <li key={i} className="bg-gray-50 rounded p-2">
                        <div className="font-medium text-gray-700">
                            {i + 1}. {icons[step.tool] || '🔧'} {step.tool}
                            <span className="font-mono text-gray-500"> {JSON.stringify(step.arguments)}</span>
                        </div>
                        <div className="mt-1 text-gray-500 line-clamp-4">{step.result.slice(0, 400)}</div>
                    </li>
                ))}
                {critiques.map((c, i) => (
                    <li key={`c${i}`} className="bg-amber-50 text-amber-800 rounded p-2">↩ 自我檢查退回：{c}</li>
                ))}
            </ol>
        </details>
    );
}

// 狀態標籤 (保持不變)
function StatusBadge({ status }) {
    const styles = {
        completed: 'bg-green-100 text-green-800',
        processing: 'bg-yellow-100 text-yellow-800',
        pending: 'bg-gray-100 text-gray-600',
        failed: 'bg-red-100 text-red-800',
    };
    return (
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${styles[status] || 'bg-gray-100'}`}>
            {status}
        </span>
    );
}