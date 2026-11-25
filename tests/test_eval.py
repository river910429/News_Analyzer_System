from run_eval import first_relevant_rank, summarize


def test_first_relevant_rank_ignores_pdf_line_breaks():
    chunks = [{"content": "其他"}, {"content": "營業收入淨額為3,809,\n054,272仟元"}]
    assert first_relevant_rank(chunks, ["營業收入淨額", "3,809,054,272"]) == 2


def test_first_relevant_rank_requires_all_keywords():
    chunks = [{"content": "營業收入淨額"}, {"content": "3,809,054,272"}]
    assert first_relevant_rank(chunks, ["營業收入淨額", "3,809,054,272"]) == 0


def test_summarize_hit_at_k_and_mrr():
    rows = [{"rank": 1}, {"rank": 2}, {"rank": 0}, {"rank": 4}]
    s = summarize(rows, [1, 3, 5])
    assert s["hit@1"] == 0.25
    assert s["hit@3"] == 0.5
    assert s["hit@5"] == 0.75
    assert s["mrr"] == round((1 + 0.5 + 0.25) / 4, 3)


def test_answer_matches_supports_alternatives():
    from run_eval import answer_matches

    assert answer_matches("負債約占資產的 31.16%", ["31.2|31.16"])
    assert answer_matches("毛利率增加 3.8 個百分點", ["3.8"])
    assert not answer_matches("約 30%", ["31.2|31.16"])


def test_summarize_answer_only_rows():
    rows = [
        {"rank": None, "answer_hit": True, "answer_s": 2.0, "steps": 2},
        {"rank": None, "answer_hit": False, "answer_s": 4.0, "steps": 4},
    ]
    s = summarize(rows, [1, 3], with_answers=True)
    assert "hit@1" not in s
    assert s == {"answer_hit": 0.5, "avg_tool_calls": 3.0, "avg_answer_s": 3.0}


def test_answer_matches_numbers_on_boundaries():
    from run_eval import answer_matches

    assert not answer_matches("民國113年到114年", ["3"])
    assert answer_matches("相差 3 年（來源：第 6 頁）", ["3"])
    assert answer_matches("增加了3年。", ["3"])
    assert not answer_matches("增加 3.8 個百分點", ["3"])
    assert answer_matches("每股 18.0 元", ["18.0"])
    assert answer_matches("合計 5,460,795,283 仟元", ["5,460,795,283"])
    assert answer_matches("約為 59%。", ["59"])
