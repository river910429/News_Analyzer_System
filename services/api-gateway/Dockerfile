# 使用輕量級的 Python Image
FROM python:3.10-slim

WORKDIR /app

# 先複製 requirements 以利用 Docker Cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY src/ ./src/

# 啟動 FastAPI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]