from pathlib import Path

from numerals import annotate_numbers, cn_to_int, strip_annotations

ROOT = Path(__file__).resolve().parent.parent


def test_numerals_copies_are_identical():
    gateway = (ROOT / "services/api-gateway/src/numerals.py").read_text(encoding="utf-8")
    worker = (ROOT / "services/nlp-worker/src/numerals.py").read_text(encoding="utf-8")
    assert gateway == worker, "update both copies of numerals.py"


def test_cn_to_int():
    assert cn_to_int("一一四") == 114  # 聯電 / 聯發科 style
    assert cn_to_int("一百一十四") == 114  # 台積電 style
    assert cn_to_int("十四") == 14
    assert cn_to_int("一百零五") == 105


def test_roc_years_are_annotated():
    # x-10: 聯電's 113 EPS sentence, which a "113年每股盈餘" query never matched
    text = "本公司民國一一三年營業收入新臺幣232,302,584千元，每股普通股獲利為新臺幣3.80元"
    assert "民國一一三年〔113年〕營業收入" in annotate_numbers(text)
    assert "一百一十四年〔114年〕" in annotate_numbers("台積公司民國一百一十四年毛利率為59.9%")
    # durations and small numbers are left alone
    assert annotate_numbers("過去十年，未來三年") == "過去十年，未來三年"


def test_mixed_numbers_are_annotated():
    # multi-10: qwen2.5 read 9萬557 as 95570
    assert "9萬557〔90,557〕人" in annotate_numbers("全球員工總數為9萬557人")
    assert "4萬4,690〔44,690〕" in annotate_numbers("4萬4,690位專業人員")
    # multi-12: 1,224億2,000萬 - 900億8,000萬 = 323.4億
    out = annotate_numbers("1,224億2,000萬美元")
    assert "〔1,224.2億 = 122,420,000,000〕" in out
    assert "〔38,090.5億" in annotate_numbers("3兆8,090億5,000萬元")
    # single-unit amounts and plain numbers are clear already
    for plain in ("10億元", "387萬片", "2,472,228,595仟元", "59.9%", "5.5萬人"):
        assert annotate_numbers(plain) == plain


def test_annotation_is_idempotent_and_strippable():
    text = "民國一百一十四年全年合併營收為1,224億2,000萬美元，員工9萬557人"
    once = annotate_numbers(text)
    assert annotate_numbers(once) == once
    assert strip_annotations(once) == text


def test_chunking_annotates():
    from chunking import clean_text

    assert clean_text("民 國 一 一 四 年 營 收") == "民國一一四年〔114年〕營收"
