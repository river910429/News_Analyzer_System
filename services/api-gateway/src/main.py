"""API gateway: document upload, document listing, and RAG search."""

import json
import logging
import os
import uuid
from typing import List

import boto3
import psycopg2
import redis
import requests
from botocore.client import Config
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("api-gateway")

# ----- Configuration (all from environment, see .env.example) -----
REDIS_HOST = os.getenv("REDIS_HOST", "redis-queue")
DB_HOST = os.getenv("DB_HOST", "db")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "documents")
# Must be the same model the nlp-worker uses, otherwise the vector spaces won't match.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))
ETL_QUEUE = "etl_queue"

app = FastAPI(title="Financial Report RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Loading embedding model %s ...", EMBEDDING_MODEL)
model = SentenceTransformer(EMBEDDING_MODEL)
logger.info("Embedding model loaded")

redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"http://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="使用者的搜尋問題")
    top_k: int = Field(3, ge=1, le=10, description="要回傳幾筆最相關的結果")


class SearchResult(BaseModel):
    filename: str
    content: str
    similarity_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SearchResult]


@app.on_event("startup")
def ensure_bucket() -> None:
    s3 = get_s3_client()
    try:
        s3.create_bucket(Bucket=MINIO_BUCKET)
        logger.info("Created bucket %s", MINIO_BUCKET)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            logger.info("Bucket %s already exists", MINIO_BUCKET)
        else:
            raise


# Endpoints below are plain `def` (not `async def`) on purpose: they use blocking
# libraries (psycopg2, boto3, requests), and FastAPI runs `def` endpoints in a
# thread pool so they don't block the event loop.


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    s3 = get_s3_client()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        s3_key = f"{uuid.uuid4()}_{file.filename}"
        s3.put_object(Bucket=MINIO_BUCKET, Key=s3_key, Body=file.file.read())

        cursor.execute(
            "INSERT INTO documents (filename, s3_key, status) VALUES (%s, %s, %s) RETURNING id",
            (file.filename, s3_key, "pending"),
        )
        doc_id = cursor.fetchone()[0]
        conn.commit()

        task = {"doc_id": doc_id, "s3_key": s3_key, "filename": file.filename}
        redis_client.lpush(ETL_QUEUE, json.dumps(task))
        logger.info("Queued document %s (%s)", doc_id, file.filename)

        return {"status": "success", "doc_id": doc_id, "filename": file.filename}
    except Exception as e:
        conn.rollback()
        logger.exception("Upload failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.get("/documents")
def list_documents():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, filename, status, upload_date FROM documents ORDER BY upload_date DESC"
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        {"id": r[0], "name": r[1], "status": r[2], "date": r[3].strftime("%Y-%m-%d %H:%M")}
        for r in rows
    ]


def retrieve(query: str, top_k: int) -> List[SearchResult]:
    """Return the top_k most similar chunks above SIMILARITY_THRESHOLD."""
    query_vector = model.encode(query).tolist()
    sql = """
        SELECT d.filename, c.chunk_text, (c.embedding <=> %s::vector) AS distance
        FROM document_chunks c
        JOIN documents d ON c.document_id = d.id
        ORDER BY distance ASC
        LIMIT %s;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (str(query_vector), top_k))
            rows = cursor.fetchall()
    finally:
        conn.close()

    results = []
    for filename, chunk_text, distance in rows:
        score = 1 - distance  # cosine distance -> cosine similarity
        if score > SIMILARITY_THRESHOLD:
            results.append(
                SearchResult(filename=filename, content=chunk_text, similarity_score=round(score, 4))
            )
    return results


def build_prompt(query: str, sources: List[SearchResult]) -> str:
    # Numbering follows the `sources` list, so [參考資料 N] matches the N-th source shown in the UI.
    context = "\n\n".join(f"[參考資料 {i}]\n{s.content}" for i, s in enumerate(sources, start=1))
    return f"""
你是一位專業的財報與企業分析師。請「嚴格」根據以下提供的[參考資料]來回答使用者的問題。
如果參考資料中沒有提到，請回答「根據目前的文件，無法得知此資訊」，絕不能自己編造答案。
請用繁體中文，語氣專業、簡潔地回答。

[參考資料]
{context}

[使用者的問題]
{query}

請開始你的回答：
"""


def generate(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("response", "無法生成回答").strip()


@app.post("/search", response_model=ChatResponse)
def search_and_generate(request: SearchRequest):
    """RAG: retrieve relevant chunks from pgvector, then ask Ollama to answer from them."""
    try:
        sources = retrieve(request.query, request.top_k)
        if not sources:
            return ChatResponse(
                answer="抱歉，根據目前上傳的文件，我找不到與您問題相關的資訊。", sources=[]
            )

        logger.info("Calling Ollama (%s) with %d sources", OLLAMA_MODEL, len(sources))
        answer = generate(build_prompt(request.query, sources))
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.exception("RAG search failed")
        raise HTTPException(status_code=500, detail=str(e))
