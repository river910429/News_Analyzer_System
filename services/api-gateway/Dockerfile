# 使用輕量級的 Python Image
FROM python:3.9-slim

WORKDIR /app

# 先複製 requirements 以利用 Docker Cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY src/ ./src/

# 啟動 FastAPI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]