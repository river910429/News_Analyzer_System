from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
import boto3
import psycopg2
import redis
import json
import os
import uuid
from botocore.client import Config

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 連線設定
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