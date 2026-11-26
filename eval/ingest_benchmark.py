"""Ingestion throughput benchmark: upload N documents at once and time the indexing.

Run it with one worker and with KEDA autoscaling to measure what scaling buys:

    # Kubernetes, autoscaling 1-3 workers (KEDA)
    python eval/ingest_benchmark.py --pdf report.pdf --copies 6 --api http://localhost:30080 --watch-k8s
    # Kubernetes, a single worker: pause KEDA and pin one replica first
    kubectl -n financial-rag annotate scaledobject nlp-worker autoscaling.keda.sh/paused-replicas=1 --overwrite

The uploaded copies are deleted afterwards (use --keep to keep them), so they do not
skew the retrieval benchmarks.
"""

import argparse
import json
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent


def request(method, url, data=None, headers=None, timeout=120):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload(api, pdf_bytes, filename):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8")
    body += pdf_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    return request("POST", f"{api}/upload", body, headers)["doc_id"]


def worker_replicas():
    """Ready nlp-worker pods according to Kubernetes (None if kubectl is unavailable)."""
    try:
        out = subprocess.run(
            [
                "kubectl",
                "-n",
                "financial-rag",
                "get",
                "deployment",
                "nlp-worker",
                "-o",
                "jsonpath={.status.readyReplicas}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(out.stdout or 0)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--pdf", required=True, help="PDF to upload")
    parser.add_argument("--copies", type=int, default=6)
    parser.add_argument("--poll", type=float, default=5)
    parser.add_argument("--watch-k8s", action="store_true", help="record nlp-worker replicas over time")
    parser.add_argument("--keep", action="store_true", help="keep the uploaded documents")
    parser.add_argument("--tag", default="ingest")
    args = parser.parse_args()

    pdf = Path(args.pdf).read_bytes()
    start = time.time()
    ids = [upload(args.api, pdf, f"bench-{i:02d}-{Path(args.pdf).name}") for i in range(args.copies)]
    print(f"uploaded {len(ids)} documents in {time.time() - start:.1f}s")

    timeline = []
    done_at = {}
    while True:
        docs = {d["id"]: d["status"] for d in request("GET", f"{args.api}/documents")}
        now = time.time() - start
        for doc_id in ids:
            if docs.get(doc_id) in ("completed", "failed") and doc_id not in done_at:
                done_at[doc_id] = now
        replicas = worker_replicas() if args.watch_k8s else None
        timeline.append({"t": round(now), "done": len(done_at), "workers": replicas})
        print(f"t={now:6.0f}s  done {len(done_at)}/{len(ids)}" + (f"  workers={replicas}" if args.watch_k8s else ""))
        if len(done_at) == len(ids):
            break
        time.sleep(args.poll)

    total = max(done_at.values())
    failed = [i for i in ids if docs.get(i) == "failed"]
    summary = {
        "documents": len(ids),
        "failed": len(failed),
        "total_s": round(total),
        "docs_per_min": round(len(ids) / total * 60, 2),
        "max_workers": max((p["workers"] or 0) for p in timeline) if args.watch_k8s else None,
    }
    print("\n" + "  ".join(f"{k}={v}" for k, v in summary.items()))

    out = HERE / "results" / f"{datetime.now():%Y%m%d-%H%M}_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "timeline": timeline}, indent=2), encoding="utf-8")
    print(f"saved {out}")

    if not args.keep:
        for doc_id in ids:
            request("DELETE", f"{args.api}/documents/{doc_id}")
        print(f"deleted the {len(ids)} benchmark documents")


if __name__ == "__main__":
    main()
