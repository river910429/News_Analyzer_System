"""ETL worker: pull tasks from Redis, parse PDFs, chunk, embed, store in pgvector."""

import io
import json
import logging
import os
import signal
import time
from typing import List

import boto3
import psycopg2
import redis
from psycopg2.extras import execute_values
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from chunking import chunk_pages
from companies import document_company
from indexing import SCHEMA_SQL, write_postings
from tables import extract_table_chunks

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("nlp-worker")

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
# Must be the same model the api-gateway uses.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
ETL_QUEUE = "etl_queue"

logger.info("Loading embedding model %s ...", EMBEDDING_MODEL)
model = SentenceTransformer(EMBEDDING_MODEL)
logger.info("Embedding model loaded")

redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, health_check_interval=30)


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
    )


def extract_pages(pdf_bytes: bytes) -> List[str]:
    """Extract text per page. Raises ValueError if nothing readable is found."""
    pdf = PdfReader(io.BytesIO(pdf_bytes))
    if pdf.is_encrypted:
        logger.warning("PDF is encrypted, trying empty password")
        pdf.decrypt("")

    pages = []
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            logger.warning("Page %d has no extractable text (image-only?)", i)
        pages.append(text)  # keep empty pages so page numbers stay correct

    if not any(p.strip() for p in pages):
        raise ValueError("Extracted text is empty (image-only or empty PDF)")
    return pages


def ensure_schema() -> None:
    """Idempotent migration for databases created before page numbers existed."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def set_status(conn, doc_id: int, status: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute("UPDATE documents SET status = %s WHERE id = %s", (status, doc_id))
    conn.commit()


def process_document(conn, s3, doc_id: int, s3_key: str) -> None:
    set_status(conn, doc_id, "processing")

    obj = s3.get_object(Bucket=MINIO_BUCKET, Key=s3_key)
    pdf_bytes = obj["Body"].read()
    pages = extract_pages(pdf_bytes)
    logger.info("Doc %s: extracted %d pages", doc_id, len(pages))
    with conn.cursor() as cursor:
        cursor.execute("SELECT filename FROM documents WHERE id = %s", (doc_id,))
        filename = cursor.fetchone()[0]
    company = document_company(filename, "\n".join(pages[:5]))
    logger.info("Doc %s: company %s", doc_id, company)

    chunks = chunk_pages(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    table_chunks = extract_table_chunks(pdf_bytes)
    chunks += table_chunks
    logger.info("Doc %s: %d chunks (%d table chunks), embedding...", doc_id, len(chunks), len(table_chunks))
    texts = [text for _, text in chunks]
    embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True) if texts else []

    # Old chunks are replaced and the new ones committed together with the 'completed'
    # status in one transaction: re-indexing is safe, and a failure never leaves
    # half a document searchable.
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
        inserted = execute_values(
            cursor,
            "INSERT INTO document_chunks (document_id, page_number, chunk_text, embedding) VALUES %s RETURNING id",
            [(doc_id, page, text, vec.tolist()) for (page, text), vec in zip(chunks, embeddings, strict=True)],
            fetch=True,
        )
        n_postings = write_postings(cursor, [row[0] for row in inserted], texts)
        cursor.execute("UPDATE documents SET status = 'completed', company = %s WHERE id = %s", (company, doc_id))
    conn.commit()
    logger.info("Doc %s: %d postings in the inverted index", doc_id, n_postings)
    logger.info("Doc %s: completed", doc_id)


def handle_task(raw: bytes) -> None:
    task = json.loads(raw)
    doc_id = task["doc_id"]
    logger.info("Processing document %s (%s)", doc_id, task.get("filename"))

    conn = get_db_connection()
    try:
        process_document(conn, get_s3_client(), doc_id, task["s3_key"])
    except Exception:
        logger.exception("Doc %s: failed", doc_id)
        conn.rollback()  # discard any partially inserted chunks
        try:
            set_status(conn, doc_id, "failed")
        except Exception:
            logger.exception("Doc %s: could not mark as failed", doc_id)
    finally:
        conn.close()


_stopping = False


def _request_stop(signum, _frame) -> None:
    """On SIGTERM (Kubernetes scale-in, docker stop) finish the current document, then exit."""
    global _stopping
    _stopping = True
    logger.info("Received signal %s: finishing the current task, then exiting", signum)


def main() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    ensure_schema()
    logger.info("Worker listening on queue '%s'", ETL_QUEUE)
    while not _stopping:
        try:
            # Wake up every 5 s instead of blocking forever, so an idle connection
            # never hits a socket timeout.
            item = redis_client.brpop(ETL_QUEUE, timeout=5)
            if item is None:
                continue
            handle_task(item[1])
        except (redis.exceptions.TimeoutError, redis.exceptions.ConnectionError) as e:
            # An idle Redis connection can time out; redis-py reconnects on the next call.
            logger.warning("Redis connection issue (%s), retrying", e.__class__.__name__)
            time.sleep(2)
        except Exception:
            # Keep the worker alive on unexpected errors (e.g. Redis/DB briefly down).
            logger.exception("Unexpected error in worker loop")
            time.sleep(2)
    logger.info("Worker stopped")


if __name__ == "__main__":
    main()
