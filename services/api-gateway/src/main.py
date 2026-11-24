"""API gateway: document upload, document listing, and RAG search."""

import json
import logging
import os
import uuid
from typing import List, Literal, Optional

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

from lexical import LexicalIndex, reciprocal_rank_fusion

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
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # hybrid | vector | bm25
CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "30"))  # per retriever, before fusion
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


Mode = Literal["hybrid", "vector", "bm25"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="使用者的搜尋問題")
    top_k: int = Field(5, ge=1, le=10, description="要回傳幾筆最相關的結果")
    mode: Optional[Mode] = Field(None, description="檢索方式，預設用 RETRIEVAL_MODE")


class SearchResult(BaseModel):
    filename: str
    page: Optional[int] = None
    content: str
    similarity_score: float  # cosine similarity between question and chunk


class ChatResponse(BaseModel):
    answer: str
    sources: List[SearchResult]


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)
    mode: Optional[Mode] = None
    apply_threshold: bool = Field(
        False, description="評估時預設不套用相似度門檻，才能看到完整的排序"
    )


@app.on_event("startup")
def ensure_schema() -> None:
    """Idempotent migration for databases created before page numbers existed."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER")
        conn.commit()
    finally:
        conn.close()


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


@app.post("/documents/{doc_id}/reindex")
def reindex_document(doc_id: int):
    """Re-run parsing/chunking/embedding for an uploaded document (e.g. after changing the chunker)."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE documents SET status = 'pending' WHERE id = %s RETURNING s3_key, filename",
                (doc_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found")
        conn.commit()
    finally:
        conn.close()

    redis_client.lpush(ETL_QUEUE, json.dumps({"doc_id": doc_id, "s3_key": row[0], "filename": row[1]}))
    logger.info("Queued re-index of document %s", doc_id)
    return {"status": "queued", "doc_id": doc_id}


def _chunk_signature():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(*), coalesce(max(id), 0) FROM document_chunks")
            return cursor.fetchone()
    finally:
        conn.close()


def _all_chunks():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, chunk_text FROM document_chunks ORDER BY id")
            return cursor.fetchall()
    finally:
        conn.close()


lexical_index = LexicalIndex(_chunk_signature, _all_chunks)


def retrieve(
    query: str, top_k: int, apply_threshold: bool = True, mode: Optional[str] = None
) -> List[SearchResult]:
    """Hybrid retrieval: vector search + BM25, combined with reciprocal rank fusion.

    mode="vector" or "bm25" uses a single retriever (useful for evaluation).
    With apply_threshold, a chunk is kept if its cosine similarity passes the
    threshold or BM25 ranked it within top_k (exact keyword matches such as
    numbers often have modest cosine similarity).
    """
    mode = mode or RETRIEVAL_MODE
    query_vector = str(model.encode(query, normalize_embeddings=True).tolist())
    n_candidates = max(CANDIDATES, top_k)

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            vector_ids: List[int] = []
            if mode in ("hybrid", "vector"):
                cursor.execute(
                    "SELECT id FROM document_chunks ORDER BY embedding <=> %s::vector LIMIT %s",
                    (query_vector, n_candidates),
                )
                vector_ids = [r[0] for r in cursor.fetchall()]
            bm25_ids = lexical_index.search(query, n_candidates) if mode in ("hybrid", "bm25") else []

            rankings = [r for r in (vector_ids, bm25_ids) if r]
            fused = reciprocal_rank_fusion(rankings)
            ranked_ids = sorted(fused, key=lambda i: -fused[i])[:top_k]
            if not ranked_ids:
                return []

            cursor.execute(
                """
                SELECT c.id, d.filename, c.page_number, c.chunk_text,
                       1 - (c.embedding <=> %s::vector) AS similarity
                FROM document_chunks c JOIN documents d ON c.document_id = d.id
                WHERE c.id = ANY(%s)
                """,
                (query_vector, ranked_ids),
            )
            rows = {r[0]: r for r in cursor.fetchall()}
    finally:
        conn.close()

    bm25_top = set(bm25_ids[:top_k])
    results = []
    for chunk_id in ranked_ids:
        _, filename, page, text, similarity = rows[chunk_id]
        if apply_threshold and similarity <= SIMILARITY_THRESHOLD and chunk_id not in bm25_top:
            continue
        results.append(
            SearchResult(filename=filename, page=page, content=text, similarity_score=round(similarity, 4))
        )
    return results


def build_prompt(query: str, sources: List[SearchResult]) -> str:
    # Numbering follows the `sources` list, so [參考資料 N] matches the N-th source shown in the UI.
    def header(i: int, s: SearchResult) -> str:
        page = f"，第 {s.page} 頁" if s.page else ""
        return f"[參考資料 {i}]（{s.filename}{page}）"

    context = "\n\n".join(f"{header(i, s)}\n{s.content}" for i, s in enumerate(sources, start=1))
    return f"""
你是一位專業的財報與企業分析師。請「嚴格」根據以下提供的[參考資料]來回答使用者的問題。
如果參考資料中沒有提到，請回答「根據目前的文件，無法得知此資訊」，絕不能自己編造答案。

注意事項：
1. 數字必須逐字照抄參考資料，不要自行換算或四捨五入，並註明單位（例如新台幣仟元）。
2. 財報比較表中，一列數字的欄位通常依序為「本年度、前一年度、增減金額、增減比例」，請勿混淆。
3. 同一份年報可能同時有「合併」與「個體」財務報表，若問題沒有指定，以合併報表為主並說明。
4. 回答時以 [參考資料 N] 標註依據。
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
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},  # factual Q&A: keep answers deterministic
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("response", "無法生成回答").strip()


@app.post("/search", response_model=ChatResponse)
def search_and_generate(request: SearchRequest):
    """RAG: retrieve relevant chunks from pgvector, then ask Ollama to answer from them."""
    try:
        sources = retrieve(request.query, request.top_k, mode=request.mode)
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


@app.post("/retrieve", response_model=List[SearchResult])
def retrieve_only(request: RetrieveRequest):
    """Retrieval without calling the LLM. Used by eval/run_eval.py to measure Hit@k / MRR."""
    try:
        return retrieve(request.query, request.top_k, request.apply_threshold, request.mode)
    except Exception as e:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=500, detail=str(e))
