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
