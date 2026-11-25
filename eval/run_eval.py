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
    python eval/run_eval.py --qa eval/agent_qa.jsonl --answers   # multi-step questions, plain RAG
    python eval/run_eval.py --qa eval/agent_qa.jsonl --agent     # same questions, tool-calling agent
"""

import argparse
import json
import re
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


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers_in(text: str) -> list:
    return [float(m.replace(",", "")) for m in _NUMBER.findall(text)]


def _contains(text: str, expected: str) -> bool:
    """Text match, or for a bare number, numeric match.

    A number matches when the answer contains the same value at the precision written in the
    expected answer: "4" matches "4.0元" but not "113年"; "31.2" matches "31.16%".
    """
    expected = normalize(expected).rstrip("%")
    if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", expected):
        value = float(expected.replace(",", ""))
        decimals = len(expected.split(".")[1]) if "." in expected else 0
        tolerance = 0.5 * 10**-decimals if decimals else 1e-9
        return any(abs(n - value) <= tolerance + 1e-9 for n in _numbers_in(text))
    return expected in text


def answer_matches(answer: str, expected: list) -> bool:
    """Every expected string must appear in the answer; "a|b" accepts either a or b."""
    text = normalize(answer)
    return all(any(_contains(text, alt) for alt in kw.split("|")) for kw in expected)


def summarize(rows: list, ks, with_answers: bool = False) -> dict:
    """Hit@k and MRR from per-question ranks (0 = not found, None = no retrieval check)."""
    summary = {}
    ranked = [r for r in rows if r.get("rank") is not None]
    if ranked:
        n = len(ranked)
        summary.update({f"hit@{k}": round(sum(1 for r in ranked if 0 < r["rank"] <= k) / n, 3) for k in ks})
        summary["mrr"] = round(sum(1 / r["rank"] for r in ranked if r["rank"]) / n, 3)
    if with_answers:
        summary["answer_hit"] = round(sum(r["answer_hit"] for r in rows) / len(rows), 3)
        steps = [r["steps"] for r in rows if "steps" in r]
        if steps:
            summary["avg_tool_calls"] = round(sum(steps) / len(steps), 2)
        summary["avg_answer_s"] = round(sum(r["answer_s"] for r in rows) / len(rows), 1)
    return summary


def load_qa(path: Path) -> list:
    items = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            item = json.loads(line)
            if not item.get("keywords") and not item.get("answer_keywords"):
                sys.exit(f"{path}:{line_no}: 'keywords' or 'answer_keywords' is required")
            items.append(item)
    return items


def rescore(path: Path, qa: list) -> None:
    """Apply the current answer checker to saved answers, e.g. after fixing the checker."""
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {item.get("id", i): item for i, item in enumerate(qa, start=1)}
    for row in data["rows"]:
        item = expected[row["id"]]
        row["answer_hit"] = answer_matches(row["answer"], item.get("answer_keywords") or item["keywords"])
    data["summary"] = summarize(data["rows"], KS, with_answers=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{path.name}: " + "  ".join(f"{k}={v}" for k, v in data["summary"].items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--qa", default=str(HERE / "qa.jsonl"))
    parser.add_argument("--top-k", type=int, default=max(KS))
    parser.add_argument("--answers", action="store_true", help="also call /search and check the LLM answers")
    parser.add_argument("--agent", action="store_true", help="check answers from /agent instead of /search")
    parser.add_argument(
        "--mode", choices=["hybrid", "vector", "bm25"], help="retriever to use (default: server setting)"
    )
    parser.add_argument(
        "--rerank", action="store_true", help="re-rank with the cross-encoder (first call downloads the model)"
    )
    parser.add_argument("--timeout", type=float, default=600, help="seconds to wait for each request")
    parser.add_argument("--tag", help="name of this run (default: the mode)")
    parser.add_argument("--model", help="Ollama model for /agent, e.g. qwen3:8b (default: server setting)")
    parser.add_argument("--rescore", help="re-check the answers saved in a results JSON (no API calls)")
    args = parser.parse_args()

    if args.rescore:
        rescore(Path(args.rescore), load_qa(Path(args.qa)))
        return

    check_answers = args.answers or args.agent
    default_tag = "agent" if args.agent else (args.mode or "default") + ("-rerank" if args.rerank else "")
    args.tag = args.tag or default_tag
    qa = load_qa(Path(args.qa))
    if not qa:
        sys.exit(f"No questions found in {args.qa} (only comments?)")
    print(f"{len(qa)} questions, tag={args.tag}\n")

    rows = []
    for i, item in enumerate(qa, start=1):
        row = {"id": item.get("id", i), "question": item["question"], "rank": None}

        if item.get("keywords") and not args.agent:
            t0 = time.time()
            chunks = post(
                f"{args.api}/retrieve",
                {"query": item["question"], "top_k": args.top_k, "mode": args.mode, "rerank": args.rerank or None},
                args.timeout,
            )
            row["rank"] = first_relevant_rank(chunks, item["keywords"])
            row["top1_score"] = chunks[0]["similarity_score"] if chunks else None
            row["retrieve_ms"] = round((time.time() - t0) * 1000)

        if check_answers:
            t0 = time.time()
            if args.agent:
                res = post(f"{args.api}/agent", {"question": item["question"], "model": args.model}, args.timeout)
                row["steps"] = len(res["steps"])
                row["tools"] = [s["tool"] for s in res["steps"]]
                row["stopped_reason"] = res["stopped_reason"]
            else:
                res = post(
                    f"{args.api}/search",
                    {"query": item["question"], "top_k": 5, "mode": args.mode, "rerank": args.rerank or None},
                    args.timeout,
                )
            row["answer"] = res["answer"]
            row["answer_hit"] = answer_matches(res["answer"], item.get("answer_keywords") or item["keywords"])
            row["answer_s"] = round(time.time() - t0, 1)

        rows.append(row)
        marks = []
        if row["rank"] is not None:
            marks.append(f"rank {row['rank']}" if row["rank"] else "MISS")
        if check_answers:
            marks.append("answer " + ("OK" if row["answer_hit"] else "WRONG"))
        if "steps" in row:
            marks.append(f"{row['steps']} tools")
        print(f"[{i:>2}/{len(qa)}] {'  '.join(marks):<28} {item['question']}")

    n = len(rows)
    summary = summarize(rows, [k for k in KS if k <= args.top_k], check_answers)
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

    misses = [r for r in rows if r["rank"] == 0]
    if misses:
        print("\nMissed (relevant chunk not in top-k):")
        for r in misses:
            print(f"  - {r['question']}")
    wrong = [r for r in rows if check_answers and not r["answer_hit"]]
    if wrong:
        print("\nWrong answers:")
        for r in wrong:
            print(f"  - {r['question']}\n    {r['answer'][:150]!r}")


if __name__ == "__main__":
    main()
