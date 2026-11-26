"""In-memory BM25 and rank fusion.

Production BM25 runs on the inverted index in PostgreSQL (see bm25_search in main.py).
The in-memory BM25 here is the reference implementation, used by tests and the
offline tokenizer benchmark.
"""

import math
import threading
from collections import Counter
from typing import Callable, Dict, List, Sequence, Tuple

from .tokenizer import tokenize


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> List[float]:
        terms = set(tokenize(query))
        out = []
        for tf, length in zip(self.tfs, self.lengths, strict=True):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            for t in terms:
                f = tf.get(t)
                if f:
                    score += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(score)
        return out


class LexicalIndex:
    """BM25 over all chunks, rebuilt lazily when the chunk table changes."""

    def __init__(self, load_signature: Callable[[], Tuple], load_chunks: Callable[[], List[Tuple[int, str]]]):
        self._load_signature = load_signature
        self._load_chunks = load_chunks
        self._lock = threading.Lock()
        self._signature: Tuple = ()
        self._ids: List[int] = []
        self._bm25: BM25 = BM25([])

    def _refresh(self) -> None:
        signature = self._load_signature()
        with self._lock:
            if signature == self._signature:
                return
            rows = self._load_chunks()
            self._ids = [r[0] for r in rows]
            self._bm25 = BM25([r[1] for r in rows])
            self._signature = signature

    def search(self, query: str, limit: int) -> List[int]:
        """Chunk ids ordered by BM25 score (only chunks sharing at least one term)."""
        self._refresh()
        with self._lock:
            scores = self._bm25.scores(query)
            ids = self._ids
        ranked = sorted(range(len(ids)), key=lambda i: -scores[i])
        return [ids[i] for i in ranked[:limit] if scores[i] > 0]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> Dict[int, float]:
    """Combine several ranked id lists: score = sum of 1 / (k + rank)."""
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused
