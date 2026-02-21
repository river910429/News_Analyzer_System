import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function Dashboard() {
    // === 原有的 State ===
    const [files, setFiles] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);

    // === [新增] 搜尋功能的 State ===
    const [query, setQuery] = useState('');
    const [searching, setSearching] = useState(false);
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

        try {
            const res = await axios.post(`${API_URL}/search`, {
                query: userQ,
                top_k: 3
            });

            // 解構後端傳來的 ChatResponse
            const { answer, sources } = res.data;

            let aiResponse = answer;

            // 如果有參考來源，把它們折疊或附在下方
            if (sources && sources.length > 0) {
                const sourceText = sources.map((doc, idx) =>
                    `[${idx + 1}] ${doc.filename} (相似度: ${(doc.similarity_score * 100).toFixed(1)}%)`
                ).join('\n');

                aiResponse += `\n\n---\n📑 參考來源：\n${sourceText}`;
            }

            setChatHistory(prev => [...prev, { role: 'ai', text: aiResponse }]);

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
                                <div className="p-4 border-b border-gray-200 bg-gray-50">
                                    <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wide">
                                        AI Assistant (RAG Search)
                                    </h3>
                                </div>

                                {/* 聊天紀錄區 */}
                                <div className="flex-1 p-4 overflow-y-auto space-y-3 bg-gray-50">
                                    {chatHistory.map((msg, idx) => (
                                        <ChatMessage key={idx} role={msg.role} text={msg.text} />
                                    ))}
                                    {searching && <div className="text-xs text-gray-400 text-center animate-pulse">Thinking...</div>}
                                </div>

                                {/* 輸入框 */}
                                <div className="p-4 border-t border-gray-200 bg-white">
                                    <input
                                        type="text"
                                        value={query}
                                        onChange={(e) => setQuery(e.target.value)}
                                        onKeyDown={handleSearch}
                                        placeholder="輸入問題，例如：營收表現如何？"
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
function ChatMessage({ role, text }) {
    const isAi = role === 'ai';
    return (
        <div className={`flex ${isAi ? 'justify-start' : 'justify-end'}`}>
            <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-sm whitespace-pre-wrap ${isAi ? 'bg-white text-gray-800 border border-gray-200' : 'bg-blue-600 text-white'
                }`}>
                {text}
            </div>
        </div>
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