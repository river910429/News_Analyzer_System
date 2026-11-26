from chunking import chunk_pages, clean_text


def test_clean_text_removes_spaces_between_cjk():
    assert clean_text("進 入 民 國 一 百") == "進入民國一百"
    assert clean_text("營 收 3,809 億 元") == "營收3,809億元"


def test_clean_text_keeps_spaces_between_english_words():
    assert clean_text("Annual Report 2025") == "Annual Report 2025"


def test_chunks_never_cross_pages_and_keep_page_numbers():
    pages = ["第一頁的內容。" * 5, "", "第三頁的內容。" * 5]
    chunks = chunk_pages(pages, size=500, overlap=50, min_len=10)
    assert [p for p, _ in chunks] == [1, 3]
    assert "第一頁" in chunks[0][1] and "第三頁" not in chunks[0][1]


def test_chunks_respect_size_and_split_on_sentences():
    page = "".join(f"這是第{i}句話，用來測試切塊。" for i in range(100))
    chunks = chunk_pages([page], size=200, overlap=40)
    assert len(chunks) > 1
    for _, text in chunks:
        assert len(text.replace("\n", "")) <= 200
        assert text.endswith("。")  # no sentence is cut in half


def test_consecutive_chunks_overlap():
    page = "".join(f"句子{i:03d}的內容在這裡。" for i in range(60))
    chunks = chunk_pages([page], size=150, overlap=40)
    first_tail = chunks[0][1].split("\n")[-1]
    assert first_tail in chunks[1][1]


def test_table_rows_stay_whole():
    rows = "\n".join(f"營業收入淨額 {i},809,054,272 2,894,307,699 31%" for i in range(3))
    chunks = chunk_pages([rows], size=500, overlap=0, min_len=10)
    assert len(chunks) == 1
    assert chunks[0][1].count("\n") == 2


def test_very_long_unit_is_hard_split():
    chunks = chunk_pages(["字" * 1200], size=500, overlap=100)
    assert all(len(t) <= 500 for _, t in chunks)
    assert len(chunks) == 3
