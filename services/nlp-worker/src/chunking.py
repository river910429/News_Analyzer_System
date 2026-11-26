"""Text cleaning and sentence/line-aware chunking for PDF pages."""

import re
from typing import List, Optional, Sequence, Tuple

from numerals import annotate_numbers

CJK = r"㐀-鿿豈-﫿　-〿＀-￯"
_SPACE_BETWEEN_CJK = re.compile(rf"(?<=[{CJK}])[ \t]+(?=[{CJK}])")
_SPACE_CJK_ASCII = re.compile(
    rf"(?<=[{CJK}])[ \t]+(?=[0-9A-Za-z%])|(?<=[0-9A-Za-z%,.])[ \t]+(?=[{CJK}])"
)
# Split after Chinese sentence punctuation, and at line breaks (table rows are lines).
_UNIT_BOUNDARY = re.compile(r"(?<=[。！？；])|\n")


def clean_text(text: str) -> str:
    """pypdf often inserts spaces between CJK characters ('進 入 民 國'); remove them.

    Chinese years and mixed numbers are then annotated (民國一一四年〔114年〕, 9萬557〔90,557〕)
    so BM25 matches query terms like "114年" and the LLM reads the right value.
    """
    text = text.replace("　", " ")
    text = _SPACE_BETWEEN_CJK.sub("", text)
    text = _SPACE_CJK_ASCII.sub("", text)
    return annotate_numbers(re.sub(r"[ \t]{2,}", " ", text))


def _units(page_text: str) -> List[str]:
    return [u.strip() for u in _UNIT_BOUNDARY.split(page_text) if u and u.strip()]


def chunk_pages(
    pages: Sequence[Optional[str]], size: int = 500, overlap: int = 100, min_len: int = 30
) -> List[Tuple[int, str]]:
    """Build chunks of up to `size` chars from whole sentences/lines.

    Chunks never cross a page boundary, so every chunk has a page number for citations.
    Consecutive chunks share up to `overlap` chars of trailing sentences.
    Returns a list of (page_number, chunk_text), page numbers starting at 1.
    """
    chunks: List[Tuple[int, str]] = []
    for page_no, raw in enumerate(pages, start=1):
        buf: List[str] = []
        length = 0
        for unit in _units(clean_text(raw or "")):
            if buf and length + len(unit) > size:
                chunks.append((page_no, "\n".join(buf)))
                tail: List[str] = []
                tail_len = 0
                for prev in reversed(buf):
                    if tail_len + len(prev) > overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev)
                buf, length = tail, tail_len
            while len(unit) > size:  # a single unit longer than a chunk: hard split
                chunks.append((page_no, unit[:size]))
                unit = unit[size - overlap:]
            buf.append(unit)
            length += len(unit)
        if buf and length >= min_len:
            chunks.append((page_no, "\n".join(buf)))
    return chunks
