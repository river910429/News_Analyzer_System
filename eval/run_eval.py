"""Evaluate the RAG system against a hand-written QA set.

Retrieval metrics (no LLM call, fast):
    Hit@k  - share of questions where a relevant chunk appears in the top k
    MRR    - mean of 1 / rank of the first relevant chunk (0 if none in top K)

A retrieved chunk counts as relevant when it contains ALL of the question's
`keywords` (exact substrings taken from the PDF, e.g. a number or a term).

Answer check (optional, --answers, calls the LLM so it is slow on CPU):
    answer_hit - share of answers that contain every string in `answer_keywords`

Only the Python standard library is used, so it runs on the host:
    python eval/run_eval.py                       # retrieval only
    python eval/run_eval.py --answers             # + LLM answer check
    python eval/run_eval.py --mode vector         # compare retrievers: vector | bm25 | hybrid
"""

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
KS = (1, 3, 5, 10)


def post(url: str, payload: dict, timeout: float) -> object:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize(text: str) -> str:
    # PDF extraction often inserts line breaks / spaces inside numbers and words.
    return "".join(text.split())


def first_relevant_rank(chunks: list, keywords: list) -> int:
    """1-based rank of the first chunk containing all keywords, or 0."""
    kws = [normalize(k) for k in keywords]
    for rank, chunk in enumerate(chunks, start=1):
        content = normalize(chunk["content"])
        if all(k in content for k in kws):
            return rank
    return 0


def summarize(rows: list, ks, with_answers: bool = False) -> dict:
    """Hit@k and MRR from per-question ranks (0 = not found)."""
    n = len(rows)
    summary = {f"hit@{k}": round(sum(1 for r in rows if 0 < r["rank"] <= k) / n, 3) for k in ks}
    summary["mrr"] = round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3)
    if with_answers:
        summary["answer_hit"] = round(sum(r["answer_hit"] for r in rows) / n, 3)
    return summary


def load_qa(path: Path) -> list:
    items = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            item = json.loads(line)
            if not item.get("keywords"):
                sys.exit(f"{path}:{line_no}: 'keywords' is required")
            items.append(item)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--qa", default=str(HERE / "qa.jsonl"))
    parser.add_argument("--top-k", type=int, default=max(KS))
    parser.add_argument("--answers", action="store_true", help="also call /search and check the LLM answers")
    parser.add_argument(
        "--mode", choices=["hybrid", "vector", "bm25"], help="retriever to use (default: server setting)"
    )
    parser.add_argument(
        "--rerank", action="store_true", help="re-rank with the cross-encoder (first call downloads the model)"
    )
    parser.add_argument("--tag", help="name of this run (default: the mode)")
    args = parser.parse_args()

    args.tag = args.tag or ((args.mode or "default") + ("-rerank" if args.rerank else ""))
    qa = load_qa(Path(args.qa))
    if not qa:
        sys.exit(f"No questions found in {args.qa} (only comments?)")
    print(f"{len(qa)} questions, tag={args.tag}\n")

    rows = []
    for i, item in enumerate(qa, start=1):
        t0 = time.time()
        chunks = post(
            f"{args.api}/retrieve",
            {"query": item["question"], "top_k": args.top_k, "mode": args.mode, "rerank": args.rerank or None},
            60,
        )
        rank = first_relevant_rank(chunks, item["keywords"])
        row = {
            "id": item.get("id", i),
            "question": item["question"],
            "rank": rank,
            "top1_score": chunks[0]["similarity_score"] if chunks else None,
            "retrieve_ms": round((time.time() - t0) * 1000),
        }

        if args.answers:
            t0 = time.time()
            res = post(
                f"{args.api}/search",
                {"query": item["question"], "top_k": 5, "mode": args.mode, "rerank": args.rerank or None},
                300,
            )
            answer = res["answer"]
            expected = item.get("answer_keywords") or item["keywords"]
            row["answer"] = answer
            row["answer_hit"] = all(normalize(k) in normalize(answer) for k in expected)
            row["answer_s"] = round(time.time() - t0, 1)

        rows.append(row)
        mark = f"rank {rank}" if rank else "MISS"
        extra = f"  answer {'OK' if row.get('answer_hit') else 'WRONG'}" if args.answers else ""
        print(f"[{i:>2}/{len(qa)}] {mark:<7}{extra}  {item['question']}")

    n = len(rows)
    summary = summarize(rows, [k for k in KS if k <= args.top_k], args.answers)
    print("\n" + "  ".join(f"{k}={v}" for k, v in summary.items()))

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = out_dir / f"{stamp}_{args.tag}.json"
    out.write_text(
        json.dumps(
            {"tag": args.tag, "time": stamp, "n": n, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(f"saved {out}")

    misses = [r for r in rows if not r["rank"]]
    if misses:
        print("\nMissed (relevant chunk not in top-k):")
        for r in misses:
            print(f"  - {r['question']}")


if __name__ == "__main__":
    main()
