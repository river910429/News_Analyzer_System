# 1. 改用 Python 3.10 解決 Boto3 警告
FROM python:3.10-slim

WORKDIR /app

# 2. 安裝系統依賴 (pypdf 有時需要這些)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. 先複製 requirements 確保 cache 機制運作
COPY requirements.txt .

# 4. 安裝 Python 套件 (顯示詳細輸出)
RUN pip install --no-cache-dir -r requirements.txt

# 5. 複製程式碼
COPY src/ ./src/

# 6. 設定 Python 路徑
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1 

CMD ["python", "src/main.py"]