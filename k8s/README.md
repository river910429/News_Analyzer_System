# Kubernetes deployment (Docker Desktop)

Why Kubernetes here: indexing is the bottleneck. Embedding one annual report takes about
3 minutes on CPU, so uploading ten reports keeps a single worker busy for half an hour.
**KEDA** watches the Redis job queue and scales `nlp-worker` from 1 to 3 pods while there is
a backlog, then back to 1.

```
browser ──► frontend (NodePort 30173)
        └─► api-gateway (NodePort 30080) ──► db (pgvector, PVC)
MCP client ─► mcp-server (NodePort 30801)  ├► minio (PVC)
                                           └► redis-queue ◄── KEDA watches etl_queue length
                                                   │
                                     nlp-worker × 1..3 (shared HF model cache PVC)
Ollama runs on the host: http://host.docker.internal:11434
```

| File | Contents |
|---|---|
| `storage.yaml` | PVCs for PostgreSQL, MinIO and the shared Hugging Face model cache |
| `infra.yaml` | PostgreSQL + pgvector, Redis, MinIO with readiness probes |
| `apps.yaml` | api-gateway (startup/readiness/liveness probes on `/health`), nlp-worker, mcp-server, frontend |
| `scaledobject.yaml` | KEDA `ScaledObject`: one worker per waiting document, 1-3 replicas |
| `deploy.ps1` | builds images, creates the secret from `.env`, applies everything, installs KEDA |

Design notes:

- **Same service names as docker-compose** (`db`, `redis-queue`, `minio`), so `.env` is reused
  unchanged; it becomes the `rag-env` Secret.
- **Graceful scale-in**: a job leaves the Redis list when a worker takes it, so the queue can be
  empty while workers are still embedding. KEDA then scales in; the worker handles SIGTERM by
  finishing the current document, and `terminationGracePeriodSeconds: 600` gives it time.
  Re-indexing is idempotent (old chunks are replaced in one transaction), so a killed worker
  never leaves half a document.
- **CPU sharing**: each worker is limited to 6 PyTorch threads so three workers do not
  oversubscribe the CPU; requests reserve 2 CPUs and 2.5 GiB per worker (bge-m3 in RAM).
- **Shared model cache**: new worker pods load bge-m3 from the PVC instead of downloading 2 GB.

## Run

Enable Kubernetes in Docker Desktop, stop the compose stack (`docker compose down`) to free
memory, then:

```powershell
powershell -ExecutionPolicy Bypass -File k8s\deploy.ps1
kubectl -n financial-rag get pods
```

Open http://localhost:30173. Docker Desktop should have at least 12 GB of memory
(Settings → Resources) for the API plus three workers.

## Measure scaling

```powershell
# autoscaling (KEDA, up to 3 workers)
python eval/ingest_benchmark.py --api http://localhost:30080 --pdf "report.pdf" --copies 6 --watch-k8s --tag ingest-keda

# baseline: pin one worker
kubectl -n financial-rag annotate scaledobject nlp-worker autoscaling.keda.sh/paused-replicas=1 --overwrite
python eval/ingest_benchmark.py --api http://localhost:30080 --pdf "report.pdf" --copies 6 --watch-k8s --tag ingest-1worker
kubectl -n financial-rag annotate scaledobject nlp-worker autoscaling.keda.sh/paused-replicas-
```

Watch scaling live with `kubectl -n financial-rag get pods -w`.

## Remove

```powershell
kubectl delete namespace financial-rag
```
