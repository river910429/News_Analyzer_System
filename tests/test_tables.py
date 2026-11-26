from tables import table_chunks

CONSOLIDATED = """台積公司及子公司財務報表
單位：新台幣仟元
項目 民國114年 民國113年 增減金額 增減比率（%）
流動資產 3,812,456,021 3,184,575,836 627,880,185 20%
資產總額 7,933,023,878 6,691,938,000 1,241,085,878 19%
流動負債 1,605,467,170 1,383,452,151 222,015,019 16%
負債總額 2,472,228,595 2,368,362,135 103,866,460 4%
資產總額增加：主要係長期投資增加所致。"""


def test_table_chunks_keep_title_header_and_rows():
    [chunk] = table_chunks(CONSOLIDATED)
    lines = chunk.split("\n")
    assert lines[0] == "【台積公司及子公司財務報表（合併報表）】"
    assert lines[1].startswith("項目 民國114年 民國113年")
    assert "負債總額 2,472,228,595 2,368,362,135 103,866,460 4%" in lines
    assert not any("主要係" in line for line in lines)  # prose is left to the normal chunks


def test_parent_only_statement_is_labelled():
    text = CONSOLIDATED.replace("台積公司及子公司財務報表", "台積公司財務報表")
    assert table_chunks(text)[0].startswith("【台積公司財務報表（個體報表）】")


def test_prose_page_gives_no_table_chunks():
    assert table_chunks("台積公司成立於民國七十六年。\n營收為3,809億元。") == []


def test_long_tables_are_split_and_every_chunk_has_the_title():
    rows = "\n".join(f"項目{i} 1,{i:03d},000 2,{i:03d},000" for i in range(60))
    chunks = table_chunks("某公司及子公司財務報表\n項目 民國114年 民國113年\n" + rows, size=300)
    assert len(chunks) > 1
    assert all(c.startswith("【某公司及子公司財務報表（合併報表）】\n項目 民國114年") for c in chunks)
