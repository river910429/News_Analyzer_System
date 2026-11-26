"""Company entity recognition and linking (dictionary-based NER).

Finds which companies a question or a document is about and maps every alias
(台積電, 台積公司, TSMC, 2330 ...) to one canonical id. The retriever uses it to filter
chunks to the companies a question asks about; the worker tags each document.

A dictionary beats a statistical NER model here: the set of companies is known and small,
aliases include stock codes and English names a general NER model would miss, and linking
(台積電 -> TSMC) is needed anyway. Add a company by adding a line.

This file is duplicated in services/api-gateway/src and services/nlp-worker/src;
tests/test_companies.py fails if the copies differ.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

COMPANIES: Dict[str, Tuple[str, ...]] = {
    "TSMC": ("台積電", "台積公司", "台灣積體電路", "TSMC", "2330"),
    "MediaTek": ("聯發科", "聯發科技", "MediaTek", "2454"),
    "UMC": ("聯電", "聯華電子", "UMC", "2303"),
    # Training companies for the LoRA stage (lora/): the evaluation stays on the three above.
    "ASE": ("日月光投控", "日月光", "ASE Technology", "3711"),
    "Novatek": ("聯詠", "聯詠科技", "Novatek", "3034"),
    "Realtek": ("瑞昱", "瑞昱半導體", "Realtek", "2379"),
}

DISPLAY_NAMES = {
    "TSMC": "台積電",
    "MediaTek": "聯發科",
    "UMC": "聯電",
    "ASE": "日月光投控",
    "Novatek": "聯詠",
    "Realtek": "瑞昱",
}


def _matches(text: str, alias: str) -> int:
    if alias.isascii():
        return text.lower().count(alias.lower())
    return text.count(alias)


def companies_in(text: str) -> List[str]:
    """Canonical ids of the companies mentioned in a question, in order of first mention."""
    found = []
    for company, aliases in COMPANIES.items():
        positions = [
            text.lower().find(a.lower()) if a.isascii() else text.find(a) for a in aliases if _matches(text, a)
        ]
        if positions:
            found.append((min(positions), company))
    return [company for _, company in sorted(found)]


def link(name: str) -> Optional[str]:
    """Canonical id for a company name or alias (e.g. the agent's tool argument), or None."""
    if not name:
        return None
    for company, aliases in COMPANIES.items():
        if name.strip().lower() == company.lower() or any(_matches(name, a) for a in aliases):
            return company
    return None


def document_company(filename: str, text: str = "") -> Optional[str]:
    """Which company a document belongs to: from its file name, else the most mentioned in its text."""
    from_name = companies_in(filename)
    if from_name:
        return from_name[0]
    counts = Counter({c: sum(_matches(text, a) for a in aliases) for c, aliases in COMPANIES.items()})
    company, n = counts.most_common(1)[0] if counts else (None, 0)
    return company if n else None
