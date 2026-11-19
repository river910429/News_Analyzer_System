# Financial Report RAG Q&A

A Retrieval-Augmented Generation (RAG) system for financial reports. Upload PDF reports, let the system index them in the background, then ask questions in natural language and get answers from a **local LLM** together with the source passages they were based on.

財報問答系統：上傳財報 PDF，系統在背景建立向量索引，使用者以自然語言提問，由本地 LLM 根據檢索到的段落回答並附上來源。

<!-- TODO: add a screenshot / GIF of the UI here, e.g. docs/demo.gif -->

## Features

- **PDF upload** with asynchronous indexing (status: `pending → processing → completed / failed`)
- **Semantic search** over document chunks using `BAAI/bge-m3` embeddings (1024-dim, multilingual) in PostgreSQL + pgvector
- **Grounded answers** from a local LLM (Ollama + `qwen2.5`), with cited source chunks and similarity scores
- **No external API keys** — everything runs locally with Docker Compose

## Architecture

```mermaid
flowchart LR
    UI["React (Vite)"] -- "HTTP" --> API["api-gateway (FastAPI)"]
    API -- "PDF" --> S3[("MinIO")]
    API -- "lpush task" --> Q[("Redis etl_queue")]
    Q -- "brpop" --> W["nlp-worker"]
    W -- "download" --> S3
    W -- "parse → chunk → embed" --> PG[("PostgreSQL + pgvector")]
    API -- "query embedding + vector search" --> PG
    API -- "prompt with retrieved chunks" --> LLM["Ollama (qwen2.5, on host)"]
```

**Indexing (async)**
1. `POST /upload` stores the PDF in MinIO, inserts a `pending` row in `documents`, and pushes a task to Redis.
2. `nlp-worker` pops the task, extracts text with `pypdf`, splits it into 500-char chunks with 50-char overlap, embeds them with `bge-m3`, and writes them to `document_chunks`. Chunks and the `completed` status are committed in one transaction, so a failed document never becomes partially searchable.

**Question answering**
1. `POST /search` embeds the question with the same model and retrieves the top-k chunks by cosine similarity.
2. Chunks below the similarity threshold (default 0.4) are dropped. If nothing is left, the API answers "not found" without calling the LLM.
3. The remaining chunks are put into a Traditional Chinese prompt that instructs the model to answer **only** from the provided context.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS |
| API | FastAPI, Pydantic |
| Queue | Redis (list as a simple task queue) |
| Worker | Python, pypdf, sentence-transformers |
| Storage | PostgreSQL 16 + pgvector, MinIO (S3-compatible) |
| LLM | Ollama (`qwen2.5`) running on the host |
| Infra | Docker Compose |

## Getting Started

### Prerequisites

- Docker and Docker Compose
- [Ollama](https://ollama.com/) running on the host machine, with the model pulled:

  ```bash
  ollama pull qwen2.5
  ```

- About 4 GB of free RAM for the embedding model (the first start downloads `bge-m3`, ~2 GB)

### Run

```bash
git clone https://github.com/river910429/News_Analyzer_System.git
cd News_Analyzer_System
cp .env.example .env        # adjust passwords if you like
docker-compose up --build
```

| Service | URL |
|---|---|
| Web UI | http://localhost:5173 |
| API docs (Swagger) | http://localhost:8000/docs |
| MinIO console | http://localhost:9001 |

Upload a PDF in the sidebar, wait until its status becomes `completed`, select it, and ask a question such as 「營收表現如何？」.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Upload a PDF (multipart `file`); queues it for indexing |
| `GET` | `/documents` | List documents and their indexing status |
| `POST` | `/search` | `{"query": "...", "top_k": 3}` → `{"answer": "...", "sources": [...]}` |

## Configuration

All settings come from environment variables; see [`.env.example`](.env.example). Notable ones:

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Must be the same for gateway and worker; the DB column is `vector(1024)` |
| `SIMILARITY_THRESHOLD` | `0.4` | Minimum cosine similarity for a chunk to be used as context |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Characters per chunk / overlap |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `http://host.docker.internal:11434` / `qwen2.5` | LLM endpoint and model |
| `VITE_API_URL` | `http://localhost:8000` | API URL used by the browser |

## Design Decisions

- **Queue-based indexing.** PDF parsing and embedding take seconds to minutes. Moving them to a worker keeps `/upload` fast and lets the worker scale independently of the API.
- **pgvector instead of a dedicated vector DB.** Documents, metadata and vectors live in one PostgreSQL database, so joins (chunk → filename) are plain SQL and there is one less service to operate.
- **`bge-m3` embeddings.** Financial reports here are mostly Traditional Chinese with English terms mixed in; `bge-m3` is a strong multilingual model that runs on CPU.
- **Local LLM via Ollama.** No API keys or per-request cost, and documents never leave the machine.
- **Similarity threshold + strict prompt.** Both aim to reduce hallucination: weak matches are not sent to the LLM, and the prompt tells it to say "unknown" rather than guess.

## Project Structure

```
.
├── docker-compose.yml
├── init.sql                    # documents + document_chunks (vector(1024))
├── .env.example
└── services/
    ├── api-gateway/src/main.py # /upload, /documents, /search
    ├── nlp-worker/src/main.py  # ETL loop: download → parse → chunk → embed → store
    └── frontend/src/App.jsx    # upload sidebar, document status, chat panel
```

## Roadmap

- [x] Cleanup: remove dead code, env-based config, logging, `.env.example`
- [ ] Retrieval quality: paragraph-aware chunking, hybrid search (BM25 + vector), reranker, page numbers in citations, delete endpoint
- [ ] Evaluation: 30–50 question QA set with Hit@k / MRR, baseline vs. improvements
- [ ] Engineering: pytest + GitHub Actions, async DB/HTTP, streaming answers (SSE), health checks, hosted demo
