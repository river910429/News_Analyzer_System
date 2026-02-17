# 📈 Financial Sentiment Analysis System (財經新聞情緒分析儀)

這是一個基於 **Event-Driven Microservices (事件驅動微服務)** 架構的 NLP 系統。
使用者可以輸入財經新聞，系統會透過非同步排程，使用 BERT 模型進行情緒判斷（利多/利空）。

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)
![Python](https://img.shields.io/badge/Python-3.9-green)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React-61DAFB)
![Redis](https://img.shields.io/badge/MessageQueue-Redis-red)

## 🏗 System Architecture (系統架構)

本專案採用 **Producer-Consumer** 模式，確保系統在高流量下仍能穩定運作，避免因 AI 模型運算耗時而阻塞 API。

```mermaid
graph LR
    User[Web UI (React)] -- HTTP POST --> API[API Gateway (FastAPI)]
    API -- Push Task --> Redis[(Redis Queue)]
    API -- Return TaskID --> User
    
    Worker[NLP Worker (Python)] -- Pop Task --> Redis
    Worker -- Load Model --> BERT[HuggingFace Model]
    Worker -- Save Result --> Redis
    
    User -- Polling Result --> API
    API -- Get Result --> Redis

🚀 Tech Stack (技術堆疊)
Frontend: React, Vite

Backend: FastAPI (Async Web Framework)

Message Queue: Redis (作為 Task Queue 與 Result Store)

NLP Engine: Pytorch, HuggingFace Transformers (DistilBERT Multilingual)

Infrastructure: Docker, Docker Compose

Configuration: Pydantic Settings (.env management)

📂 Project Structure (專案結構)
.
├── docker-compose.yml      # 容器編排
├── .env                    # 環境變數設定
├── services/
│   ├── api-gateway/        # [Service] 接收請求，派發任務
│   ├── nlp-worker/         # [Worker] 背景執行 AI 運算
│   └── frontend/           # [Web] 使用者介面
🛠️ How to Run (如何執行)
Prerequisites
Docker & Docker Compose installed.

Start the System
Bash
# 1. Clone the repository
git clone <your-repo-url>

# 2. Build and Run services
docker-compose up --build
Access the Application
Web UI: http://localhost:5173

API Docs: http://localhost:8000/docs

🧪 Testing
Manual Test
打開 Web UI。

輸入新聞：「台積電營收創新高，股價大漲」。

點擊分析，查看結果。

API Test (Curl)
Bash
curl -X POST http://localhost:8000/submit-task \
     -H "Content-Type: application/json" \
     -d '{"text": "Sample text", "user_id": "test"}'