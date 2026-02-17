import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function Dashboard() {
    const [files, setFiles] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);

    // 載入文件列表
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
        // 簡單的輪詢，每 3 秒更新一次狀態
        const interval = setInterval(fetchDocuments, 3000);
        return () => clearInterval(interval);
    }, []);

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
            fetchDocuments(); // 更新列表
        } catch (err) {
            alert("Upload failed");
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="flex h-screen bg-gray-50 text-gray-800 font-sans">
            {/* Sidebar */}
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
            <main className="flex-1 flex flex-col">
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
                        <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6">
                            <h3 className="text-lg font-bold mb-4">Document Details</h3>
                            <p><strong>Status:</strong> {selectedFile.status}</p>
                            <p><strong>Upload Date:</strong> {selectedFile.date}</p>
                            <p className="mt-4 text-sm text-gray-500">
                                * Vector embeddings have been generated and stored in PostgreSQL.
                                <br />
                                * RAG Query interface coming soon.
                            </p>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}

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