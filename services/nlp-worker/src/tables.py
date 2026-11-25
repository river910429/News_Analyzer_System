"""Extra chunks for number tables, with their title and column header attached.

pypdf reads annual-report spreads (two printed pages side by side) across both halves
and garbles table headers, so a row like "負債總額 2,472,228,595 ..." loses which table
and which year column it belongs to. pdfplumber, run on each half of a spread, keeps
rows and headers in order. These chunks are added next to the normal prose chunks.
"""

import io
import logging
import re
from typing import List, Tuple

logger = logging.getLogger("nlp-worker.tables")

_BIG_NUMBER = re.compile(r"\(?\d{1,3}(?:,\d{3})+\)?")
_YEAR_HEADER = re.compile(r"(民國)?\d{3}年")


def _label(title: str) -> str:
    # Taiwanese reports name consolidated statements "<公司>及子公司財務報表".
    if "財務報表" in title or "財務報告" in title:
        return f"{title}（合併報表）" if "子公司" in title else f"{title}（個體報表）"
    return title


def table_chunks(half_text: str, size: int = 500, min_rows: int = 4) -> List[str]:
    """Group the number rows of one (half) page into chunks prefixed with title + column header."""
    lines = [line.strip() for line in half_text.split("\n") if line.strip()]
    rows = [line for line in lines if len(_BIG_NUMBER.findall(line)) >= 2]
    if len(rows) < min_rows:
        return []

    title = next((line for line in lines[:4] if "表" in line), lines[0])
    header = next((line for line in lines if _YEAR_HEADER.search(line) and not _BIG_NUMBER.search(line)), "")
    prefix = "\n".join(part for part in (f"【{_label(title)}】", header) if part)

    chunks: List[str] = []
    buf: List[str] = []
    length = len(prefix)
    for row in rows:
        if buf and length + len(row) > size:
            chunks.append(prefix + "\n" + "\n".join(buf))
            buf, length = [], len(prefix)
        buf.append(row)
        length += len(row) + 1
    if buf:
        chunks.append(prefix + "\n" + "\n".join(buf))
    return chunks


def extract_table_chunks(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """Table chunks for the whole PDF as (page_number, text). Empty if pdfplumber is missing."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed, skipping table chunks")
        return []

    out: List[Tuple[int, str]] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            width, height = page.width, page.height
            # A landscape page is a two-page spread: read each half on its own.
            parts = (
                [page.crop((0, 0, width / 2, height)), page.crop((width / 2, 0, width, height))]
                if width > height * 1.2
                else [page]
            )
            for part in parts:
                try:
                    text = part.extract_text() or ""
                except Exception:
                    logger.exception("pdfplumber failed on page %d", page_no)
                    continue
                out.extend((page_no, chunk) for chunk in table_chunks(text))
    return out
