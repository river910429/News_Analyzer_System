from src.lexical import BM25, LexicalIndex, reciprocal_rank_fusion, tokenize


def test_tokenize_cjk_bigrams_and_whole_numbers():
    tokens = tokenize("毛利率為59.9%")
    assert "毛利" in tokens and "利率" in tokens
    assert "59.9%" in tokens


def test_tokenize_adds_comma_free_numbers():
    tokens = tokenize("營收3,809,054,272")
    assert "3,809,054,272" in tokens and "3809054272" in tokens


def test_tokenize_ignores_pdf_spaces_between_cjk():
    assert tokenize("營 業 收 入") == tokenize("營業收入")


def test_bm25_prefers_document_with_query_terms():
    docs = ["今年毛利率為59.9%", "員工總數為9萬557人", "股票代碼為2330"]
    scores = BM25(docs).scores("毛利率是多少")
    assert scores.index(max(scores)) == 0


def test_bm25_exact_number_match():
    docs = ["營業收入淨額3,809,054,272", "營業收入淨額2,894,307,699"]
    scores = BM25(docs).scores("2894307699")
    assert scores[1] > scores[0]


def test_rrf_rewards_items_ranked_high_by_both():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    ranked = sorted(fused, key=lambda i: -fused[i])
    assert ranked[0] == 1  # 1st and 2nd beats 3rd and 1st
    assert set(ranked) == {1, 2, 3, 4}


def test_lexical_index_rebuilds_only_when_chunks_change():
    calls = {"load": 0}
    data = [(10, "台積公司成立於民國七十六年"), (11, "股票代碼為2330")]
    signature = [(2, 11)]

    def load():
        calls["load"] += 1
        return list(data)

    index = LexicalIndex(lambda: signature[0], load)
    assert index.search("股票代碼", 5)[0] == 11
    index.search("成立", 5)
    assert calls["load"] == 1  # cached

    data.append((12, "新的股票資料"))
    signature[0] = (3, 12)
    index.search("股票", 5)
    assert calls["load"] == 2


def test_search_drops_chunks_without_any_shared_term():
    index = LexicalIndex(lambda: (1,), lambda: [(1, "毛利率"), (2, "員工人數")])
    assert index.search("毛利", 10) == [1]
