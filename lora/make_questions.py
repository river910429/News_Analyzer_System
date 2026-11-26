"""Generate training questions from the annual reports of the *training* companies.

The evaluation sets cover TSMC, MediaTek and UMC; training questions come only from other
companies' reports (ASE, Novatek, Realtek by default), so the fine-tuned model is measured on
companies it never saw in training.

For each sampled passage that contains numbers, a teacher LLM writes questions of the kinds
the 7B agent got wrong — annual totals vs quarterly values, two-year changes, ratios of two
figures — each with the expected answer. A few unanswerable questions (periods the reports
cannot cover) teach the model to say so.

    python lora/make_questions.py --api http://localhost:30080 --companies ASE Novatek Realtek
"""

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "services" / "api-gateway" / "src"))
from companies import DISPLAY_NAMES  # noqa: E402

PROMPT = """以下是{company}年報第 {page} 頁的一段內容：

{text}

請根據這段內容出最多 {n} 個繁體中文問題，給財報分析助理練習。規則：
1. 每題都要寫出公司名稱「{company}」、年度（例如 114年）與指標的具體名稱
   （例如「營業收入淨額」「毛利率」「研發費用」「員工人數」），只能用這段內容回答。
2. 題型盡量不同：(a) 直接查一個數字；(b) 兩個年度的變化（增加/減少多少）；
   (c) 兩個數字的比例或百分比；(d) 段落列出季度或半年度數字時，問全年合計。
3. answer 只寫最後的數字（不要單位、不要逗號以外的符號），例如 "48.2" 或 "1,234,567"。
4. 不要出這些題目：「某數字」「哪一項數值是…」這類沒有指明指標的題目；問頁碼、電話、
   日期、地址、項目編號或「有幾項」的題目；段落內容看不懂（亂碼）的題目。
5. 段落沒有適合的數字時，回傳空的 questions。

只輸出 JSON：{{"questions": [{{"question": "...", "answer": "...", "type": "lookup|change|ratio|total"}}]}}"""

UNANSWERABLE = [
    "{company}117年第二季的營業收入是多少？",
    "{company}116年第一季的毛利率是多少？",
    "{company}118年的每股盈餘預估是多少？",
    "{company}114年在南美洲市場的營收是多少？",
    "{company}114年單日股價最高是多少元？",
]


def get(url: str):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_json(ollama: str, model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        f"{ollama}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        content = json.loads(resp.read().decode("utf-8"))["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


# Characters a Traditional Chinese report is made of; anything else is extraction noise.
_NORMAL_CHAR = re.compile(
    r"[\u0020-\u007e\u00a0-\u00ff\u2000-\u206f\u2100-\u22ff\u3000-\u303f\u3400-\u4dbf"
    r"\u4e00-\u9fff\uff00-\uffef\s]"
)
# Questions that cannot teach report reading: no named metric, page lookups, trivia.
_BAD_QUESTION = re.compile(
    r"某|哪一項數|數字是多少|財務數字|第\s*\d+\s*頁|頁中|電話|地址|符號|標記為|項目編號|有幾項|幾種"
)


def garbled_ratio(text: str) -> float:
    """Share of characters outside the scripts of a Chinese report.

    Some PDFs (日月光's) embed fonts without a Unicode map, so pypdf returns noise such as
    "115ϋ Jamaican dollar 4˜26˚"; the teacher then writes questions about the noise.
    """
    if not text:
        return 1.0
    return sum(1 for ch in text if not _NORMAL_CHAR.match(ch)) / len(text)


def numeric_passages(chunks: list, min_numbers: int = 4, min_len: int = 150, max_garbled: float = 0.01) -> list:
    """Chunks worth asking about: readable, enough text and several multi-digit numbers."""
    keep = []
    for c in chunks:
        text = c["text"]
        if (
            len(text) >= min_len
            and garbled_ratio(text) <= max_garbled
            and len(re.findall(r"\d[\d,]*\.?\d*", text)) >= min_numbers
        ):
            keep.append(c)
    return keep


def number_in(answer: str, text: str) -> bool:
    """True if the answer's number appears in the passage (commas and spaces ignored)."""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", answer)
    if not m:
        return False
    value = m.group(0).replace(",", "")
    flat = re.sub(r"(?<=\d)[,\s]+(?=\d)", "", text)
    return re.search(rf"(?<![\d.]){re.escape(value)}(?!\d)", flat) is not None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default="http://localhost:30080")
    p.add_argument("--ollama", default="http://localhost:11434")
    p.add_argument("--model", default="qwen2.5:14b", help="teacher model that writes the questions")
    p.add_argument("--companies", nargs="+", default=["ASE", "Novatek", "Realtek"])
    p.add_argument("--passages", type=int, default=90, help="passages sampled per company")
    p.add_argument("--per-passage", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "data" / "questions.jsonl"))
    args = p.parse_args()

    random.seed(args.seed)
    docs = [d for d in get(f"{args.api}/documents") if d.get("company") in args.companies]
    if not docs:
        raise SystemExit(f"no indexed documents for {args.companies}; upload the reports first")
    eval_companies = {"TSMC", "MediaTek", "UMC"}
    assert not eval_companies & set(args.companies), "training on evaluation companies would leak"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, n = set(), 0
    with out.open("w", encoding="utf-8") as f:
        for doc in docs:
            company = doc["company"]
            name = DISPLAY_NAMES[company]
            passages = numeric_passages(get(f"{args.api}/documents/{doc['id']}/chunks"))
            sample = random.sample(passages, min(args.passages, len(passages)))
            print(f"{name}: {len(passages)} readable numeric passages, sampling {len(sample)}")
            for c in sample:
                prompt = PROMPT.format(company=name, page=c["page"], text=c["text"], n=args.per_passage)
                for q in ollama_json(args.ollama, args.model, prompt).get("questions", [])[: args.per_passage]:
                    question, answer = str(q.get("question", "")).strip(), str(q.get("answer", "")).strip()
                    if not question or not re.search(r"\d", answer) or name not in question or question in seen:
                        continue
                    if _BAD_QUESTION.search(question) or garbled_ratio(question) > 0:
                        continue
                    # A lookup answer must be copied from the passage; the teacher can misread too.
                    if q.get("type") == "lookup" and not number_in(answer, c["text"]):
                        continue
                    seen.add(question)
                    row = {
                        "id": f"{company}-{n}",
                        "question": question,
                        "answer": answer,
                        "type": q.get("type", ""),
                        "company": company,
                        "page": c["page"],
                        "passage": c["text"],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
            for t in UNANSWERABLE:
                row = {"id": f"{company}-u{n}", "question": t.format(company=name), "type": "unanswerable"}
                f.write(json.dumps(row | {"company": company}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} questions -> {out}")


if __name__ == "__main__":
    main()
