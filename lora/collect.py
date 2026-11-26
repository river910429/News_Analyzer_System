"""Run the teacher model through the agent harness and keep the conversations worth imitating.

Each training question goes to /agent with the teacher (qwen2.5:14b) and include_messages, so
the trajectory is produced by exactly the harness, tools and system prompt the student will
run under. A trajectory is kept only if it is clean:

- the answer matches the expected answer (numeric match, as in eval/run_eval.py), or, for an
  unanswerable question, it abstains;
- the harness did not have to repair the answer itself: no arithmetic corrections, no tool
  calls recovered from text, no guardrail, and it ended normally;
- it searched at least once (no answers from memory).

This is rejection sampling: the student learns from the teacher's successes only.

Critiques are allowed, but the trajectory is repaired before training: each rejected draft
answer and the critique that followed it are removed. "search → draft '無法得知' → critique →
search again → answer" becomes "search → search again → answer", which is what a good agent
does unprompted. In the first run, keeping only critique-free trajectories kept 20 of 148,
and most rejections were 14B refusing after one search, then succeeding when told to retry.

    python lora/collect.py --api http://localhost:30080 --model qwen2.5:14b
"""

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "services" / "api-gateway"))
from run_eval import answer_matches  # noqa: E402
from src.agent import TOOL_SCHEMAS  # noqa: E402

ABSTAIN = "無法得知|找不到|並未|未提及|沒有提到|未提供|未明確"


def ask(api: str, question: str, model: str, timeout: float) -> dict:
    payload = {"question": question, "model": model, "include_messages": True}
    req = urllib.request.Request(
        f"{api}/agent", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def repair_messages(messages: list, critiques: list) -> list:
    """Drop each (draft answer, critique) pair, so the trajectory reads as if done right first time."""
    critique_set = set(critiques)
    out, i = [], 0
    while i < len(messages):
        m = messages[i]
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        is_draft = m["role"] == "assistant" and not m.get("tool_calls")
        if is_draft and nxt and nxt["role"] == "user" and nxt.get("content") in critique_set:
            i += 2
            continue
        out.append(m)
        i += 1
    return out


def rejection_reason(item: dict, res: dict) -> str:
    """Why a trajectory is not used for training, or "" to keep it."""
    if res.get("stopped_reason") != "answered":
        return f"stopped:{res.get('stopped_reason')}"
    if res.get("corrections"):
        return "arithmetic_fixed"
    steps = res.get("steps") or []
    if any(s.get("recovered") for s in steps):
        return "text_tool_call"
    if not any(s["tool"] in ("search_report", "get_page") for s in steps):
        return "no_search"
    expected = [ABSTAIN] if item.get("type") == "unanswerable" else [item["answer"]]
    if not answer_matches(res["answer"], expected):
        return "wrong_answer"
    return ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default="http://localhost:30080")
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--questions", default=str(HERE / "data" / "questions.jsonl"))
    p.add_argument("--out-dir", default=str(HERE / "data"))
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    items = [json.loads(line) for line in Path(args.questions).read_text(encoding="utf-8").splitlines() if line]
    kept, reasons, log = [], {}, []
    started = time.time()
    for i, item in enumerate(items, start=1):
        try:
            res = ask(args.api, item["question"], args.model, args.timeout)
            reason = rejection_reason(item, res)
        except Exception as e:  # timeout / HTTP error: skip the question
            res, reason = {}, f"error:{e.__class__.__name__}"
        reasons[reason or "kept"] = reasons.get(reason or "kept", 0) + 1
        log.append({"id": item["id"], "reason": reason or "kept", "answer": res.get("answer", "")})
        if not reason:
            critiques = res.get("critiques") or []
            messages = repair_messages(res["messages"], critiques)
            kept.append(
                {"id": item["id"], "company": item["company"], "messages": messages, "repaired": len(critiques)}
            )
            if critiques:
                reason_note = "KEPT (repaired " + ",".join(res.get("checks") or []) + ")"
            else:
                reason_note = "KEPT"
        else:
            reason_note = reason
        print(f"[{i:>3}/{len(items)}] {reason_note:<32} {item['question']}")

    random.seed(args.seed)
    random.shuffle(kept)
    # Split by company-free random order; all training companies differ from the eval companies.
    n_val = max(1, int(len(kept) * args.val_fraction)) if len(kept) > 10 else 0
    out = Path(args.out_dir)
    for name, rows in (("val", kept[:n_val]), ("train", kept[n_val:])):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row | {"tools": TOOL_SCHEMAS}, ensure_ascii=False) + "\n")
    (out / "collect_log.json").write_text(
        json.dumps({"model": args.model, "reasons": reasons, "rows": log}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    minutes = (time.time() - started) / 60
    repaired = sum(1 for k in kept if k["repaired"])
    print(f"\nkept {len(kept)}/{len(items)} ({repaired} repaired) in {minutes:.0f} min: {reasons}")
    print(f"train {len(kept) - n_val}, val {n_val} -> {out}")


if __name__ == "__main__":
    main()
