from pathlib import Path

from companies import companies_in, document_company, link

ROOT = Path(__file__).resolve().parent.parent


def test_company_module_copies_are_identical():
    a = (ROOT / "services/api-gateway/src/companies.py").read_text(encoding="utf-8")
    b = (ROOT / "services/nlp-worker/src/companies.py").read_text(encoding="utf-8")
    assert a == b, "update both copies of companies.py"


def test_companies_in_finds_aliases_in_order():
    assert companies_in("台積電和聯發科114年的每股盈餘誰高？") == ["TSMC", "MediaTek"]
    assert companies_in("聯華電子 vs tsmc") == ["UMC", "TSMC"]
    assert companies_in("毛利率是多少？") == []


def test_alias_overlap_does_not_confuse_companies():
    # 聯發科技 contains 聯發科; 聯電 must not be found inside 聯發科
    assert companies_in("聯發科技的研發費用") == ["MediaTek"]


def test_link_maps_tool_arguments():
    assert link("台積公司") == "TSMC"
    assert link("mediatek") == "MediaTek"
    assert link("三星") is None


def test_document_company_from_name_then_text():
    assert document_company("MediaTek_2025_AR.pdf") == "MediaTek"
    assert document_company("114年報.pdf", "聯華電子股份有限公司 民國一一四年度年報 聯電") == "UMC"
    assert document_company("report.pdf", "no company here") is None


def test_interleave_keeps_every_company_near_the_top():
    from src.main import _interleave

    assert _interleave([[1, 2, 3], [7, 8]]) == [1, 7, 2, 8, 3]
    assert _interleave([[1, 2], [2, 3]]) == [1, 2, 3]
