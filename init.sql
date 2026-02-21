-- 啟用向量擴充功能
CREATE EXTENSION IF NOT EXISTS vector;

-- 建立文件表
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    s3_key TEXT NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending'
);

-- 建立向量表 (Chunks)
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    chunk_text TEXT NOT NULL,
    -- ⚠️ 關鍵修改：從 384 改成 1024 維
    embedding vector(1024) 
);