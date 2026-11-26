"""Tokenizer for Traditional Chinese financial text (BM25 / inverted index).

Chinese has no spaces, so CJK text becomes overlapping character bigrams; numbers and
ASCII words stay whole, with a comma-free copy so 3,809 also matches 3809.
On the 43-question benchmark bigrams beat unigrams and dictionary word segmentation
(jieba), see eval/README.md.

This file is duplicated in services/api-gateway/src and services/nlp-worker/src because
each service has its own Docker build context; tests/test_tokenizer.py fails if the copies differ.
"""

import re
from typing import List

_CJK = r"\u3400-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef"
_SPACE_BETWEEN_CJK = re.compile(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK}])")
_ASCII_TOKEN = re.compile(r"[0-9A-Za-z][0-9A-Za-z.,%]*")
_CJK_RUN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")


def tokenize(text: str) -> List[str]:
    text = _SPACE_BETWEEN_CJK.sub("", text).lower()
    tokens = [t.rstrip(".,") for t in _ASCII_TOKEN.findall(text)]
    tokens += [t.replace(",", "") for t in tokens if "," in t]
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        tokens += [run[i:i + 2] for i in range(len(run) - 1)]
    return tokens
