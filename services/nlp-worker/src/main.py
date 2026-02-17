import redis
import json
import os
import psycopg2
import boto3
import io
import traceback
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# 初始化模型 (這會花一點時間下載)
print("Loading Embedding Model...")
model = SentenceTransformer('all-MiniLM-L6-v2') 
print("Model Loaded!")

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
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
    )

def process_etl():
    while True:
        # 監聽 etl_queue
        item = redis_client.brpop("etl_queue", timeout=0)
        if item:
            task = json.loads(item[1])
            doc_id = task['doc_id']
            s3_key = task['s3_key']
            
            print(f"Processing Document ID: {doc_id}...")
            
            conn = get_db_connection()
            cursor = conn.cursor()
            s3 = get_s3_client()
            bucket = os.getenv("MINIO_BUCKET")
            print(f"🚀 [Start] Processing Document ID: {doc_id}...")

            try:
                # 更新狀態為 processing
                cursor.execute("UPDATE documents SET status = 'processing' WHERE id = %s", (doc_id,))
                conn.commit()

                # 1. 下載
                print(f"   Downloading S3 Key: {s3_key}")
                obj = s3.get_object(Bucket=bucket, Key=s3_key)
                file_stream = io.BytesIO(obj['Body'].read())

                # 2. 解析 PDF
                text_content = ""
                try:
                    print("   Parsing PDF...")
                    pdf = PdfReader(file_stream)
                    # 檢查是否加密
                    if pdf.is_encrypted:
                        print("   ⚠️ PDF is encrypted! Trying to decrypt...")
                        try:
                            pdf.decrypt("") # 嘗試空密碼
                        except:
                            raise Exception("PDF Encrypted and cannot be read.")
                            
                    for i, page in enumerate(pdf.pages):
                        extracted = page.extract_text()
                        if extracted:
                            text_content += extracted + "\n"
                        else:
                            print(f"   ⚠️ Page {i} extracted empty text (might be image-only PDF)")

                except Exception as pdf_err:
                    print(f"   ❌ PDF Parse Error: {pdf_err}")
                    # 備案：如果是純文字檔誤傳為 PDF，試著用 utf-8 硬讀
                    file_stream.seek(0)
                    text_content = file_stream.read().decode('utf-8', errors='ignore')

                if not text_content.strip():
                    raise Exception("Extracted text is empty! (File might be image-only PDF or empty)")

                print(f"   ✅ Extracted {len(text_content)} chars.")

                # 3. 切分文字 (Chunking) 
                # 簡單切分：每 500 字切一塊，重疊 50 字 (這比單純換行切分更好)
                chunk_size = 500
                overlap = 50
                chunks = []
                for i in range(0, len(text_content), chunk_size - overlap):
                    chunk = text_content[i:i + chunk_size]
                    if len(chunk) > 50: # 太短的不要
                        chunks.append(chunk)
                
                print(f"切分成 {len(chunks)} 個區塊，開始向量化...")

                # 4. 向量化 (Embedding)
                if chunks:
                    embeddings = model.encode(chunks)
                    
                    # 5. 存入 pgvector
                    for text, vector in zip(chunks, embeddings):
                        cursor.execute(
                            "INSERT INTO document_chunks (document_id, chunk_text, embedding) VALUES (%s, %s, %s)",
                            (doc_id, text, vector.tolist())
                        )
                
                # 更新狀態為 completed
                cursor.execute("UPDATE documents SET status = 'completed' WHERE id = %s", (doc_id,))
                conn.commit()
                print(f"Document {doc_id} processed successfully.")

            except Exception as e:
                print(f"Error: {e}")
                cursor.execute("UPDATE documents SET status = 'failed' WHERE id = %s", (doc_id,))
                conn.commit()
            finally:
                cursor.close()
                conn.close()

if __name__ == "__main__":
    process_etl()