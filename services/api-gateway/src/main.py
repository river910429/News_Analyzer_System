"""API gateway: document upload, document listing, and RAG search."""

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import boto3
import psycopg2
import redis
import requests
from botocore.client import Config
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from .agent import calculator as calculator_tool
from .agent import run_agent
from .companies import DISPLAY_NAMES, companies_in, document_company, link
from .lexical import reciprocal_rank_fusion
from .query_expansion import expand_query
from .tokenizer import tokenize

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
# Cross-encoder reranker, loaded only when first used. RERANK_BY_DEFAULT applies to /search.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANK_BY_DEFAULT = os.getenv("RERANK_BY_DEFAULT", "false").lower() == "true"
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
QUERY_EXPANSION = os.getenv("QUERY_EXPANSION", "true").lower() == "true"  # synonym expansion for BM25
ETL_QUEUE = "etl_queue"

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_schema()
    backfill_companies()
    ensure_bucket()
    yield


app = FastAPI(title="Financial Report RAG API", lifespan=lifespan)
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
    rerank: Optional[bool] = Field(None, description="是否用 reranker 重新排序，預設依 RERANK_BY_DEFAULT")


class SearchResult(BaseModel):
    filename: str
    document_id: Optional[int] = None
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
    rerank: Optional[bool] = None
    expand: Optional[bool] = Field(None, description="BM25 查詢是否做同義詞擴展，預設依 QUERY_EXPANSION")
    company_filter: bool = Field(True, description="依問題中提到的公司過濾文件（字典式 NER）")
    apply_threshold: bool = Field(
        False, description="評估時預設不套用相似度門檻，才能看到完整的排序"
    )


SCHEMA_SQL = """
ALTER TABLE documents ADD COLUMN IF NOT EXISTS company TEXT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS token_count INTEGER;
CREATE TABLE IF NOT EXISTS postings (
    term TEXT NOT NULL,
    chunk_id INTEGER NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    tf INTEGER NOT NULL,
    PRIMARY KEY (term, chunk_id)
);
CREATE INDEX IF NOT EXISTS postings_chunk_id_idx ON postings (chunk_id);
"""


def ensure_schema() -> None:
    """Idempotent migration for databases created before page numbers existed."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def backfill_companies() -> None:
    """Tag documents indexed before company detection existed, from their file names."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, filename FROM documents WHERE company IS NULL")
            for doc_id, filename in cursor.fetchall():
                company = document_company(filename)
                if company:
                    cursor.execute("UPDATE documents SET company = %s WHERE id = %s", (company, doc_id))
                    logger.info("Document %s (%s) tagged as %s", doc_id, filename, company)
        conn.commit()
    finally:
        conn.close()


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


@app.get("/health")
def health():
    """Liveness/readiness check used by docker-compose and monitoring."""
    checks = {}
    try:
        conn = get_db_connection()
        conn.close()
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"error: {e.__class__.__name__}"
    try:
        redis_client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e.__class__.__name__}"
    healthy = all(v == "ok" for v in checks.values())
    if not healthy:
        raise HTTPException(status_code=503, detail=checks)
    return {"status": "ok", **checks}


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
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        cursor.close()
        conn.close()


@app.get("/documents")
def list_documents():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, filename, status, upload_date, company FROM documents ORDER BY upload_date DESC"
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        {"id": r[0], "name": r[1], "status": r[2], "date": r[3].strftime("%Y-%m-%d %H:%M"), "company": r[4]}
        for r in rows
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    """Delete a document, its chunks and postings (ON DELETE CASCADE) and the stored PDF."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM documents WHERE id = %s RETURNING s3_key", (doc_id,))
            row = cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found")
        conn.commit()
    finally:
        conn.close()
    try:
        get_s3_client().delete_object(Bucket=MINIO_BUCKET, Key=row[0])
    except Exception:
        logger.warning("Could not delete %s from object storage", row[0])
    logger.info("Deleted document %s", doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/documents/reindex")
def reindex_all():
    """Queue every document for re-indexing (after changing chunking or the inverted index)."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE documents SET status = 'pending' RETURNING id, s3_key, filename")
            rows = cursor.fetchall()
        conn.commit()
    finally:
        conn.close()
    for doc_id, s3_key, filename in rows:
        redis_client.lpush(ETL_QUEUE, json.dumps({"doc_id": doc_id, "s3_key": s3_key, "filename": filename}))
    logger.info("Queued re-index of %d documents", len(rows))
    return {"status": "queued", "documents": len(rows)}


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


BM25_K1 = 1.5
BM25_B = 0.75

# BM25 over the inverted index. df and avgdl are computed per query, so new documents
# are searchable as soon as the worker commits them.
BM25_SQL = """
WITH q AS (SELECT DISTINCT unnest(%(terms)s::text[]) AS term),
stats AS (
    SELECT count(*)::float AS n, avg(token_count)::float AS avgdl
    FROM document_chunks WHERE token_count IS NOT NULL
),
df AS (
    SELECT p.term, count(*)::float AS df
    FROM postings p JOIN q ON p.term = q.term
    GROUP BY p.term
)
SELECT p.chunk_id,
       sum(ln(1 + (s.n - df.df + 0.5) / (df.df + 0.5))
           * p.tf * (%(k1)s + 1)
           / (p.tf + %(k1)s * (1 - %(b)s + %(b)s * c.token_count / s.avgdl))) AS score
FROM postings p
JOIN df ON df.term = p.term
JOIN document_chunks c ON c.id = p.chunk_id
CROSS JOIN stats s
WHERE %(doc_ids)s::int[] IS NULL OR c.document_id = ANY(%(doc_ids)s::int[])
GROUP BY p.chunk_id
ORDER BY score DESC, p.chunk_id
LIMIT %(limit)s
"""


def bm25_search(cursor, query: str, limit: int, doc_ids: Optional[List[int]] = None) -> List[int]:
    """Chunk ids ranked by BM25 over the postings table, optionally only from some documents."""
    terms = sorted(set(tokenize(query)))
    if not terms:
        return []
    params = {"terms": terms, "k1": BM25_K1, "b": BM25_B, "limit": limit, "doc_ids": doc_ids}
    cursor.execute(BM25_SQL, params)
    return [row[0] for row in cursor.fetchall()]

_reranker = None
_reranker_lock = threading.Lock()


def get_reranker():
    """Load the cross-encoder on first use, so startup stays fast when it is unused."""
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker %s ...", RERANKER_MODEL)
            _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
            logger.info("Reranker loaded")
    return _reranker


def _company_doc_ids(cursor, company: str) -> List[int]:
    cursor.execute("SELECT id FROM documents WHERE company = %s", (company,))
    return [r[0] for r in cursor.fetchall()]


def _ranked_ids(cursor, query: str, query_vector: str, mode: str, expand: bool, n: int, doc_ids) -> tuple:
    """Fused (vector + BM25) chunk ids for one scope; doc_ids=None searches everything."""
    vector_ids: List[int] = []
    if mode in ("hybrid", "vector"):
        if doc_ids is None:
            cursor.execute(
                "SELECT id FROM document_chunks ORDER BY embedding <=> %s::vector LIMIT %s", (query_vector, n)
            )
        else:
            cursor.execute(
                "SELECT id FROM document_chunks WHERE document_id = ANY(%s)"
                " ORDER BY embedding <=> %s::vector LIMIT %s",
                (doc_ids, query_vector, n),
            )
        vector_ids = [r[0] for r in cursor.fetchall()]
    bm25_ids: List[int] = []
    if mode in ("hybrid", "bm25"):
        bm25_ids = bm25_search(cursor, expand_query(query) if expand else query, n, doc_ids)
    fused = reciprocal_rank_fusion([r for r in (vector_ids, bm25_ids) if r])
    return sorted(fused, key=lambda i: -fused[i]), bm25_ids


def _interleave(lists: List[List[int]]) -> List[int]:
    """Round-robin merge, so every company asked about gets passages near the top."""
    out: List[int] = []
    for i in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if i < len(x) and x[i] not in out:
                out.append(x[i])
    return out


def retrieve(
    query: str,
    top_k: int,
    apply_threshold: bool = True,
    mode: Optional[str] = None,
    rerank: Optional[bool] = None,
    expand: Optional[bool] = None,
    companies: Optional[List[str]] = None,
) -> List[SearchResult]:
    """Hybrid retrieval: vector search + BM25, combined with reciprocal rank fusion.

    Company filter: companies=None detects the companies named in the query (dictionary NER);
    an empty list searches every document. With several companies (a comparison), each is
    retrieved separately and the results interleaved, so no company is crowded out.
    mode="vector" or "bm25" uses a single retriever (useful for evaluation).
    With apply_threshold, a chunk is kept if its cosine similarity passes the
    threshold or BM25 ranked it within top_k (exact keyword matches such as
    numbers often have modest cosine similarity).
    With rerank, the top RERANK_CANDIDATES fused chunks are re-scored by a
    cross-encoder that reads question and chunk together.
    """
    mode = mode or RETRIEVAL_MODE
    rerank = RERANK_BY_DEFAULT if rerank is None else rerank
    expand = QUERY_EXPANSION if expand is None else expand
    companies = companies_in(query) if companies is None else companies
    query_vector = str(model.encode(query, normalize_embeddings=True).tolist())
    n_candidates = max(CANDIDATES, top_k)
    n_keep = max(RERANK_CANDIDATES, top_k) if rerank else top_k

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            scopes = [d for d in (_company_doc_ids(cursor, c) for c in companies) if d] or [None]
            per_scope = [_ranked_ids(cursor, query, query_vector, mode, expand, n_candidates, d) for d in scopes]
            ranked_ids = _interleave([ids for ids, _ in per_scope])[:n_keep]
            bm25_top = {i for _, bm25 in per_scope for i in bm25[:top_k]}
            if not ranked_ids:
                return []

            cursor.execute(
                """
                SELECT c.id, d.filename, c.page_number, c.chunk_text,
                       1 - (c.embedding <=> %s::vector) AS similarity, c.document_id
                FROM document_chunks c JOIN documents d ON c.document_id = d.id
                WHERE c.id = ANY(%s)
                """,
                (query_vector, ranked_ids),
            )
            rows = {r[0]: r for r in cursor.fetchall()}
    finally:
        conn.close()

    if rerank:
        scores = get_reranker().predict([(query, rows[i][3]) for i in ranked_ids])
        order = sorted(range(len(ranked_ids)), key=lambda j: -scores[j])
        ranked_ids = [ranked_ids[j] for j in order[:top_k]]

    results = []
    for chunk_id in ranked_ids:
        _, filename, page, text, similarity, document_id = rows[chunk_id]
        if apply_threshold and similarity <= SIMILARITY_THRESHOLD and chunk_id not in bm25_top:
            continue
        results.append(
            SearchResult(
                filename=filename,
                document_id=document_id,
                page=page,
                content=text,
                similarity_score=round(similarity, 4),
            )
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
2. 財報比較表的年度欄位順序因公司而異（有的先列本年度，有的先列前一年度），請依表頭的年度標示讀取，勿混淆。
3. 同一份年報可能同時有「合併」與「個體」財務報表，若問題沒有指定，以合併報表為主並說明。
4. 參考資料可能來自不同公司的年報，只使用問題所問公司的資料，並注意檔名。
5. 回答時以 [參考資料 N] 標註依據。
請用繁體中文，語氣專業、簡潔地回答。

[參考資料]
{context}

[使用者的問題]
{query}

請開始你的回答：
"""


NO_RESULT_ANSWER = "抱歉，根據目前上傳的文件，我找不到與您問題相關的資訊。"


def _ollama_payload(prompt: str, stream: bool) -> dict:
    return {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {"temperature": 0.1},  # factual Q&A: keep answers deterministic
    }


def generate(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate", json=_ollama_payload(prompt, False), timeout=OLLAMA_TIMEOUT
    )
    response.raise_for_status()
    return response.json().get("response", "無法生成回答").strip()


def generate_stream(prompt: str):
    """Yield answer text pieces as Ollama produces them (it streams one JSON object per line)."""
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=_ollama_payload(prompt, True),
        stream=True,
        timeout=OLLAMA_TIMEOUT,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("response"):
                yield data["response"]
            if data.get("done"):
                break


def sse(event: dict) -> str:
    """Format one Server-Sent Event."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/search", response_model=ChatResponse)
def search_and_generate(request: SearchRequest):
    """RAG: retrieve relevant chunks from pgvector, then ask Ollama to answer from them."""
    try:
        sources = retrieve(request.query, request.top_k, mode=request.mode, rerank=request.rerank)
        if not sources:
            return ChatResponse(answer=NO_RESULT_ANSWER, sources=[])

        logger.info("Calling Ollama (%s) with %d sources", OLLAMA_MODEL, len(sources))
        answer = generate(build_prompt(request.query, sources))
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.exception("RAG search failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/search/stream")
def search_stream(request: SearchRequest):
    """Same as /search, but streams the answer as Server-Sent Events.

    Events (JSON in each `data:` line):
      {"type": "sources", "sources": [...]}   once, before the answer
      {"type": "token", "text": "..."}        many, as the LLM generates
      {"type": "done", "ttft_ms": int, "total_ms": int}
      {"type": "error", "message": "..."}     if generation fails midway
    """
    started = time.perf_counter()
    try:
        sources = retrieve(request.query, request.top_k, mode=request.mode, rerank=request.rerank)
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    def events():
        yield sse({"type": "sources", "sources": [s.model_dump() for s in sources]})
        if not sources:
            yield sse({"type": "token", "text": NO_RESULT_ANSWER})
            yield sse({"type": "done", "ttft_ms": 0, "total_ms": round((time.perf_counter() - started) * 1000)})
            return

        first_token_at = None
        try:
            for piece in generate_stream(build_prompt(request.query, sources)):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                yield sse({"type": "token", "text": piece})
        except Exception as e:
            logger.exception("Streaming generation failed")
            yield sse({"type": "error", "message": str(e)})
            return

        now = time.perf_counter()
        ttft = round(((first_token_at or now) - started) * 1000)
        total = round((now - started) * 1000)
        logger.info("Streamed answer: time to first token %d ms, total %d ms", ttft, total)
        yield sse({"type": "done", "ttft_ms": ttft, "total_ms": total})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def merge_chunks(chunks: List[str]) -> str:
    """Join consecutive chunks of one page, dropping the lines repeated by the chunk overlap."""
    lines: List[str] = []
    for chunk in chunks:
        chunk_lines = chunk.split("\n")
        overlap = 0
        for n in range(min(len(chunk_lines), len(lines)), 0, -1):
            if lines[-n:] == chunk_lines[:n]:
                overlap = n
                break
        lines.extend(chunk_lines[overlap:])
    return "\n".join(lines)


def page_text(doc_id: int, page: int) -> str:
    """Rebuild a page's text from its chunks."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT chunk_text FROM document_chunks WHERE document_id = %s AND page_number = %s ORDER BY id",
                (doc_id, page),
            )
            chunks = [r[0] for r in cursor.fetchall()]
    finally:
        conn.close()
    return merge_chunks(chunks)


@app.get("/documents/{doc_id}/chunks")
def list_chunks(doc_id: int):
    """All chunks of a document (for offline experiments such as the tokenizer benchmark)."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, page_number, chunk_text FROM document_chunks WHERE document_id = %s ORDER BY id",
                (doc_id,),
            )
            return [{"id": r[0], "page": r[1], "text": r[2]} for r in cursor.fetchall()]
    finally:
        conn.close()


@app.get("/documents/{doc_id}/pages/{page}")
def get_page(doc_id: int, page: int):
    text = page_text(doc_id, page)
    if not text:
        raise HTTPException(status_code=404, detail="Page not found (or document not indexed)")
    return {"document_id": doc_id, "page": page, "text": text}


# ----- Agent -----
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1)
    max_steps: int = Field(AGENT_MAX_STEPS, ge=1, le=12)
    model: Optional[str] = Field(None, description="Ollama 模型，預設用 OLLAMA_MODEL（方便比較不同模型）")


class AgentStep(BaseModel):
    tool: str
    arguments: object
    result: str


class AgentResponse(BaseModel):
    answer: str
    steps: List[AgentStep]
    critiques: List[str]
    stopped_reason: str
    elapsed_ms: int


def ollama_chat(messages, tools, model: Optional[str] = None):
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    if tools:
        payload["tools"] = tools
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    return response.json().get("message", {})


def search_report_tool(query: str, top_k: int = 5, company: str = "") -> str:
    canonical = link(company) if company else None
    if company and canonical is None:
        return f"不認識公司「{company}」，可用的公司：{', '.join(DISPLAY_NAMES.values())}。"
    companies = [canonical] if canonical else None  # None: detect companies named in the query
    results = retrieve(query, min(max(int(top_k), 1), 10), companies=companies)
    if not results:
        return "沒有找到相關段落。"
    return "\n\n".join(
        f"[{i}] 文件 {r.document_id}（{r.filename}）第 {r.page} 頁\n{r.content}"
        for i, r in enumerate(results, start=1)
    )


def get_page_tool(document_id: int, page: int) -> str:
    return page_text(int(document_id), int(page)) or f"文件 {document_id} 第 {page} 頁沒有內容。"


AGENT_TOOLS = {
    "search_report": search_report_tool,
    "get_page": get_page_tool,
    "calculator": calculator_tool,
}


@app.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest):
    """Answer with a tool-calling agent: it can search several times, read pages and calculate."""
    try:
        def chat(messages, tools):
            return ollama_chat(messages, tools, request.model)

        return run_agent(request.question, chat, AGENT_TOOLS, max_steps=request.max_steps)
    except Exception as e:
        logger.exception("Agent failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/retrieve", response_model=List[SearchResult])
def retrieve_only(request: RetrieveRequest):
    """Retrieval without calling the LLM. Used by eval/run_eval.py to measure Hit@k / MRR."""
    try:
        return retrieve(
            request.query,
            request.top_k,
            request.apply_threshold,
            request.mode,
            request.rerank,
            request.expand,
            None if request.company_filter else [],
        )
    except Exception as e:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
