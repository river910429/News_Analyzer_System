from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # <--- 引入 Pydantic
from typing import List, Optional
import requests
import boto3
import psycopg2
import redis
import json
import os
import uuid
from botocore.client import Config
from sentence_transformers import SentenceTransformer # <--- 引入 AI 模型

app = FastAPI()

# 1. CORS 設定 (允許前端呼叫)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 載入模型 (這是比較耗時的操作，放在全域只做一次)
print("正在載入 Embedding 模型...")
# 注意：這裡必須跟 Worker 用一模一樣的模型，不然向量空間會對不起來
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = SentenceTransformer('BAAI/bge-m3')
print("模型載入完成！")

# 3. 連線設定 (Redis, DB, S3)
redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=6379, db=0)

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=f"http://{os.getenv('MINIO_ENDPOINT')}",
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="使用者的搜尋問題")
    top_k: int = Field(3, ge=1, le=10, description="要回傳幾筆最相關的結果")

class SearchResult(BaseModel):
    filename: str
    content: str
    similarity_score: float

class SearchResult(BaseModel):
    filename: str
    content: str
    similarity_score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SearchResult]

# 初始化 MinIO Bucket
@app.on_event("startup")
async def startup_event():
    s3 = get_s3_client()
    bucket_name = os.getenv("MINIO_BUCKET")
    try:
        s3.create_bucket(Bucket=bucket_name)
    except:
        pass # Bucket 已存在

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    s3 = get_s3_client()
    conn = get_db_connection()
    cursor = conn.cursor()
    bucket = os.getenv("MINIO_BUCKET")

    try:
        # 1. 上傳到 MinIO
        file_content = await file.read()
        # 產生唯一檔名避免衝突
        s3_key = f"{uuid.uuid4()}_{file.filename}"
        s3.put_object(Bucket=bucket, Key=s3_key, Body=file_content)

        # 2. 寫入 DB (狀態: pending)
        cursor.execute(
            "INSERT INTO documents (filename, s3_key, status) VALUES (%s, %s, %s) RETURNING id",
            (file.filename, s3_key, "pending")
        )
        doc_id = cursor.fetchone()[0]
        conn.commit()

        # 3. 發送任務給 Worker
        task = {"doc_id": doc_id, "s3_key": s3_key, "filename": file.filename}
        redis_client.lpush("etl_queue", json.dumps(task))

        return {"status": "success", "doc_id": doc_id, "filename": file.filename}

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/documents")
async def list_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, status, upload_date FROM documents ORDER BY upload_date DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    docs = []
    for row in rows:
        docs.append({
            "id": row[0],
            "name": row[1],
            "status": row[2],
            "date": row[3].strftime("%Y-%m-%d %H:%M")
        })
    return docs

@app.post("/search", response_model=ChatResponse)
async def search_and_generate(request: SearchRequest):
    """
    RAG 完整流程：
    1. 檢索 (Retrieval): 從資料庫找出最相關的片段
    2. 生成 (Generation): 把片段丟給 Ollama 整理出答案
    """
    try:
        # ----- 1. 檢索階段 (Retrieval) -----
        query_vector = model.encode(request.query).tolist()
        conn = get_db_connection()
        cursor = conn.cursor()
        
        sql = """
        SELECT d.filename, c.chunk_text, (c.embedding <=> %s::vector) as distance
        FROM document_chunks c
        JOIN documents d ON c.document_id = d.id
        ORDER BY distance ASC LIMIT %s;
        """
        cursor.execute(sql, (str(query_vector), request.top_k))
        rows = cursor.fetchall()
        
        sources = []
        context_text = "" # 用來餵給 LLM 的上下文
        
        for idx, row in enumerate(rows):
            score = 1 - row[2]
            # 設定 Threshold: 如果相似度太低 (例如低於 0.4)，就不要把它當作參考資料
            if score > 0.4: 
                sources.append(SearchResult(filename=row[0], content=row[1], similarity_score=round(score, 4)))
                context_text += f"[參考資料 {idx+1}]\n{row[1]}\n\n"

        # ----- 2. 生成階段 (Generation) -----
        if not context_text:
            return ChatResponse(
                answer="抱歉，根據目前上傳的文件，我找不到與您問題相關的資訊。",
                sources=[]
            )

        # 組合給 LLM 的終極 Prompt (提示詞)
        prompt = f"""
你是一位專業的財報與企業分析師。請「嚴格」根據以下提供的[參考資料]來回答使用者的問題。
如果參考資料中沒有提到，請回答「根據目前的文件，無法得知此資訊」，絕不能自己編造答案。
請用繁體中文，語氣專業、簡潔地回答。

[參考資料]
{context_text}

[使用者的問題]
{request.query}

請開始你的回答：
"""

        # 呼叫 Ollama API (host.docker.internal 讓 Docker 可以連到你的 Windows 宿主機)
        ollama_url = "http://host.docker.internal:11434/api/generate"
        ollama_payload = {
            "model": "qwen2.5", # 如果你剛剛抓的是別的模型，請改這裡
            "prompt": prompt,
            "stream": False
        }
        
        print("正在呼叫 Ollama 生成回答...")
        response = requests.post(ollama_url, json=ollama_payload)
        response.raise_for_status()
        
        llm_answer = response.json().get("response", "無法生成回答")

        return ChatResponse(
            answer=llm_answer.strip(),
            sources=sources
        )

    except Exception as e:
        print(f"RAG Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()