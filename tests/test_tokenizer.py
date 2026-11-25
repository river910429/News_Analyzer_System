from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_tokenizer_copies_are_identical():
    gateway = (ROOT / "services/api-gateway/src/tokenizer.py").read_text(encoding="utf-8")
    worker = (ROOT / "services/nlp-worker/src/tokenizer.py").read_text(encoding="utf-8")
    assert gateway == worker, "update both copies of tokenizer.py"


def test_query_expansion_adds_report_terms():
    from src.query_expansion import expand_query

    assert "每股盈餘" in expand_query("台積電EPS多少？")
    assert "負債總額" in expand_query("總負債是多少")
    assert "台積公司" in expand_query("tsmc 股息") and "現金股利" in expand_query("tsmc 股息")


def test_query_expansion_leaves_report_wording_alone():
    from src.query_expansion import expand_query

    q = "台積公司的核心價值是什麼？"
    assert expand_query(q) == q


def test_postings_rows_counts_terms_per_chunk():
    from indexing import postings_rows

    rows, lengths = postings_rows([7, 8], ["毛利率毛利", "EPS 66.25"])
    as_dict = {(term, chunk): tf for term, chunk, tf in rows}
    assert as_dict[("毛利", 7)] == 2  # 毛利 appears twice in 毛利率毛利
    assert as_dict[("eps", 8)] == 1 and as_dict[("66.25", 8)] == 1
    assert lengths == [4, 2]
