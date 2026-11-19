"""ETL worker: pull tasks from Redis, parse PDFs, chunk, embed, store in pgvector."""

import io
import json
import logging
import os
import time
from typing import List

import boto3
import psycopg2
import redis
from psycopg2.extras import execute_values
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

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
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MIN_CHUNK_LENGTH = 50
ETL_QUEUE = "etl_queue"

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
    )


def extract_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF. Raises ValueError if nothing readable is found."""
    pdf = PdfReader(io.BytesIO(pdf_bytes))
    if pdf.is_encrypted:
        logger.warning("PDF is encrypted, trying empty password")
        pdf.decrypt("")

    pages = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            pages.append(text)
        else:
            logger.warning("Page %d has no extractable text (image-only?)", i)

    content = "\n".join(pages)
    if not content.strip():
        raise ValueError("Extracted text is empty (image-only or empty PDF)")
    return content


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Fixed-size character chunks with overlap; drops very short trailing pieces."""
    chunks = []
    for start in range(0, len(text), size - overlap):
        chunk = text[start:start + size]
        if len(chunk) > MIN_CHUNK_LENGTH:
            chunks.append(chunk)
    return chunks


def set_status(conn, doc_id: int, status: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute("UPDATE documents SET status = %s WHERE id = %s", (status, doc_id))
    conn.commit()


def process_document(conn, s3, doc_id: int, s3_key: str) -> None:
    set_status(conn, doc_id, "processing")

    obj = s3.get_object(Bucket=MINIO_BUCKET, Key=s3_key)
    text = extract_text(obj["Body"].read())
    logger.info("Doc %s: extracted %d chars", doc_id, len(text))

    chunks = chunk_text(text)
    logger.info("Doc %s: %d chunks, embedding...", doc_id, len(chunks))
    embeddings = model.encode(chunks) if chunks else []

    # Chunks and the 'completed' status are committed in one transaction,
    # so a failure never leaves half a document searchable.
    with conn.cursor() as cursor:
        execute_values(
            cursor,
            "INSERT INTO document_chunks (document_id, chunk_text, embedding) VALUES %s",
            [(doc_id, c, v.tolist()) for c, v in zip(chunks, embeddings)],
        )
        cursor.execute("UPDATE documents SET status = 'completed' WHERE id = %s", (doc_id,))
    conn.commit()
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


def main() -> None:
    logger.info("Worker listening on queue '%s'", ETL_QUEUE)
    while True:
        try:
            _, raw = redis_client.brpop(ETL_QUEUE, timeout=0)
            handle_task(raw)
        except Exception:
            # Keep the worker alive on unexpected errors (e.g. Redis/DB briefly down).
            logger.exception("Unexpected error in worker loop")
            time.sleep(2)


if __name__ == "__main__":
    main()
