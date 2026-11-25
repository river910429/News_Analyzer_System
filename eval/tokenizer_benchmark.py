"""Compare tokenizers for BM25 retrieval on the indexed chunks.

Downloads the chunks of every indexed document from the API, builds an in-memory BM25
index per tokenizer, and reports Hit@k / MRR on the QA sets. Tokenizers whose library is
not installed are skipped.

    pip install jieba                 # dictionary word segmentation (Simplified-Chinese dictionary)
    pip install hanlp                 # HanLP 2.x neural tokenizer (downloads a model on first use)
    pip install ckip-transformers     # CKIP (Academia Sinica), trained on Traditional Chinese

    python eval/tokenizer_benchmark.py
    python eval/tokenizer_benchmark.py --only bigram,hanlp
"""

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
KS = (1, 3, 5, 10)

_CJK = "㐀-鿿豈-﫿"
_SPACE_BETWEEN_CJK = re.compile("(?<=[" + _CJK + "])[ \t]+(?=[" + _CJK + "])")
_ASCII = re.compile(r"[0-9A-Za-z][0-9A-Za-z.,%]*")
_CJK_RUN = re.compile("[" + _CJK + "]+")

# Finance terms added to dictionary-based segmenters, so 毛利率 is not split into 毛利 / 率.
FINANCE_TERMS = [
    "毛利率",
    "營業利益率",
    "稅後純益率",
    "每股盈餘",
    "營業收入淨額",
    "稅後淨利",
    "現金股利",
    "資本支出",
    "負債總額",
    "資產總額",
    "權益總額",
    "前十大客戶",
    "最大客戶",
    "晶圓出貨量",
    "營業活動",
    "再生能源",
    "淨零排放",
    "研發預算",
    "核心價值",
    "股票代碼",
    "員工總數",
    "年產能",
    "合併報表",
    "個體報表",
]


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize(text):
    return "".join(text.split())


def _prepare(text):
    return _SPACE_BETWEEN_CJK.sub("", text).lower()


def _ascii_tokens(text):
    tokens = [t.rstrip(".,") for t in _ASCII.findall(text)]
    return tokens + [t.replace(",", "") for t in tokens if "," in t]


def unigram(text):
    text = _prepare(text)
    return _ascii_tokens(text) + [ch for run in _CJK_RUN.findall(text) for ch in run]


def bigram(text):
    text = _prepare(text)
    tokens = _ascii_tokens(text)
    for run in _CJK_RUN.findall(text):
        tokens += [run] if len(run) == 1 else [run[i : i + 2] for i in range(len(run) - 1)]
    return tokens


def unigram_bigram(text):
    return unigram(text) + [t for t in bigram(text) if _CJK_RUN.fullmatch(t) and len(t) == 2]


def make_segmenter_tokenizer(segment):
    """Wrap a word segmenter (list of CJK runs -> list of word lists) into a tokenizer."""

    def tokenize_many(texts):
        prepared = [_prepare(t) for t in texts]
        runs = [_CJK_RUN.findall(t) for t in prepared]
        flat = [r for rs in runs for r in rs]
        words = segment(flat) if flat else []
        out, i = [], 0
        for text, rs in zip(prepared, runs):
            toks = _ascii_tokens(text)
            for _ in rs:
                toks += [w for w in words[i] if w.strip()]
                i += 1
            out.append(toks)
        return out

    return tokenize_many


def load_tokenizers(only):
    """name -> function(list of texts) -> list of token lists."""
    simple = {"unigram": unigram, "bigram": bigram, "unigram+bigram": unigram_bigram}
    tokenizers = {name: (lambda f: lambda texts: [f(t) for t in texts])(fn) for name, fn in simple.items()}

    try:
        import logging

        import jieba

        jieba.setLogLevel(logging.ERROR)
        tokenizers["jieba"] = make_segmenter_tokenizer(lambda runs: [jieba.lcut(r) for r in runs])

        def jieba_dict(runs):
            for term in FINANCE_TERMS:
                jieba.add_word(term, freq=100000)
            return [jieba.lcut(r) for r in runs]

        tokenizers["jieba+finance dict"] = make_segmenter_tokenizer(jieba_dict)
    except ImportError:
        print("skip jieba (pip install jieba)")

    try:
        import hanlp

        tok = None

        def hanlp_segment(runs):
            nonlocal tok
            if tok is None:
                tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
                tok.dict_force = set(FINANCE_TERMS)
            return tok(runs)

        tokenizers["hanlp (coarse)"] = make_segmenter_tokenizer(hanlp_segment)
    except ImportError:
        print("skip hanlp (pip install hanlp)")

    try:
        from ckip_transformers.nlp import CkipWordSegmenter

        ws = None

        def ckip_segment(runs):
            nonlocal ws
            if ws is None:
                ws = CkipWordSegmenter(model="bert-base")
            return ws(runs, batch_size=64)

        tokenizers["ckip"] = make_segmenter_tokenizer(ckip_segment)
    except ImportError:
        print("skip ckip (pip install ckip-transformers)")

    if only:
        wanted = {w.strip() for w in only.split(",")}
        tokenizers = {k: v for k, v in tokenizers.items() if k.split(" ")[0] in wanted or k in wanted}
    return tokenizers


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(t) for t in docs_tokens]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(self.tfs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def top(self, query_tokens, k):
        terms = set(query_tokens)
        scores = []
        for i, (tf, length) in enumerate(zip(self.tfs, self.lengths)):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg)
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            scores.append((s, i))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [i for s, i in scores[:k] if s > 0]


def load_qa(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("//")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--qa", default=f"{HERE / 'qa.jsonl'},{HERE / 'qa_paraphrase.jsonl'}")
    parser.add_argument("--only", help="comma-separated tokenizer names, e.g. bigram,hanlp")
    args = parser.parse_args()

    docs = [d for d in get_json(f"{args.api}/documents") if d["status"] == "completed"]
    if not docs:
        sys.exit("No indexed documents.")
    texts = [c["text"] for d in docs for c in get_json(f"{args.api}/documents/{d['id']}/chunks")]
    print(f"{len(texts)} chunks from {len(docs)} document(s)\n")

    qa_sets = {Path(p).name: load_qa(p) for p in args.qa.split(",")}
    relevant = {
        name: [{i for i, t in enumerate(texts) if all(normalize(k) in normalize(t) for k in q["keywords"])} for q in qa]
        for name, qa in qa_sets.items()
    }

    lines = ["| Tokenizer | QA set | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | index time (s) |", "|" + "---|" * 8]
    for name, tokenize_many in load_tokenizers(args.only).items():
        t0 = time.time()
        bm25 = BM25(tokenize_many(texts))
        index_s = time.time() - t0
        for qa_name, qa in qa_sets.items():
            query_tokens = tokenize_many([q["question"] for q in qa])
            ranks = []
            for qt, rel in zip(query_tokens, relevant[qa_name]):
                top = bm25.top(qt, max(KS))
                ranks.append(next((r for r, i in enumerate(top, start=1) if i in rel), 0))
            n = len(qa)
            hits = [sum(1 for r in ranks if 0 < r <= k) / n for k in KS]
            mrr = sum(1 / r for r in ranks if r) / n
            row = f"| {name} | {qa_name} | " + " | ".join(f"{h:.3f}" for h in hits) + f" | {mrr:.3f} | {index_s:.1f} |"
            lines.append(row)
            print(row)

    out = HERE / "results" / "tokenizer_benchmark.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
