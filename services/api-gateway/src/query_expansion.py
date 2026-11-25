"""Query expansion with a financial synonym dictionary (BM25 side only).

Users say "EPS", "股息", "總負債"; annual reports say 每股盈餘, 現金股利, 負債總額.
BM25 only matches shared terms, so the query is extended with the report's wording.
Vector search handles paraphrase on its own and gets the original query.

Keep entries generic (finance vocabulary), not tuned to individual test questions,
or the paraphrase benchmark stops measuring anything.
"""

from typing import Dict, Tuple

SYNONYMS: Dict[str, Tuple[str, ...]] = {
    # income statement
    "EPS": ("每股盈餘",),
    "營收": ("營業收入", "營業收入淨額"),
    "淨利": ("稅後淨利", "本期淨利"),
    "淨賺": ("稅後淨利",),
    "賺": ("淨利",),
    "毛利": ("毛利率", "營業毛利"),
    "獲利率": ("營業利益率", "純益率"),
    "營益率": ("營業利益率",),
    "淨利率": ("稅後純益率",),
    # balance sheet / cash flow
    "總資產": ("資產總額",),
    "總負債": ("負債總額",),
    "負債比": ("負債總額", "資產總額"),
    "股東權益": ("權益總額",),
    "現金流": ("現金流量", "營業活動之現金流入"),
    "資本支出": ("資本預算",),
    # shareholders
    "股息": ("現金股利",),
    "配息": ("現金股利",),
    "股利": ("現金股利",),
    "股票代號": ("股票代碼",),
    # company / operations
    "台積電": ("台積公司",),
    "TSMC": ("台積公司",),
    "員工": ("員工總數", "員工人數"),
    "人數": ("員工總數",),
    "研發": ("研究發展",),
    "企業文化": ("核心價值",),
    "價值觀": ("核心價值",),
    "市佔": ("市場占有率",),
    "客戶占比": ("營業收入淨額",),
    "貢獻": ("占",),
    "比重": ("比例", "占"),
    "產量": ("出貨量",),
    # technology naming
    "nm": ("奈米",),
    "N2": ("2奈米",),
    "N3": ("3奈米",),
    "A16": ("A16",),
    # ESG / FX
    "碳中和": ("淨零排放",),
    "綠電": ("再生能源",),
    "台幣升值": ("美元對新台幣貶值",),
    "匯率": ("匯率", "新台幣"),
}


def expand_query(query: str) -> str:
    """Append the report-style terms for every synonym key found in the query."""
    lowered = query.lower()
    extra = []
    for key, terms in SYNONYMS.items():
        if key.lower() in lowered:
            extra.extend(t for t in terms if t not in query and t not in extra)
    return f"{query} {' '.join(extra)}" if extra else query
