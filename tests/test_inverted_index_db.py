"""BM25 over the PostgreSQL inverted index must rank like the reference implementation.

Needs a PostgreSQL server: set TEST_DATABASE_URL (CI starts one). Skipped otherwise.
"""

import os

import pytest

psycopg2 = pytest.importorskip("psycopg2")
DSN = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")

CHUNKS = [
    "台積公司民國一百一十四年毛利率為59.9%，前一年為56.1%。",
    "營業利益率為50.8%，前一年則為45.7%。",
    "每股盈餘為新台幣66.25元，前一年為45.25元。",
    "【台積公司及子公司財務報表（合併報表）】\n負債總額 2,472,228,595 2,368,362,135",
    "【台積公司財務報表（個體報表）】\n負債總額 1,856,618,339 1,748,704,133",
    "台積公司成立於民國七十六年，股票代碼為2330。",
]


@pytest.fixture()
def db():
    from psycopg2.extras import execute_values

    from indexing import SCHEMA_SQL, write_postings

    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS postings, document_chunks, documents CASCADE")
    cur.execute("CREATE TABLE documents (id SERIAL PRIMARY KEY, filename TEXT, s3_key TEXT, status TEXT)")
    cur.execute(
        "CREATE TABLE document_chunks (id SERIAL PRIMARY KEY,"
        " document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE, chunk_text TEXT NOT NULL)"
    )
    cur.execute(SCHEMA_SQL)
    cur.execute("INSERT INTO documents (filename) VALUES ('t.pdf') RETURNING id")
    doc_id = cur.fetchone()[0]
    ids = [
        r[0]
        for r in execute_values(
            cur,
            "INSERT INTO document_chunks (document_id, chunk_text) VALUES %s RETURNING id",
            [(doc_id, c) for c in CHUNKS],
            fetch=True,
        )
    ]
    write_postings(cur, ids, CHUNKS)
    conn.commit()
    yield cur, ids, doc_id
    conn.rollback()
    cur.execute("DROP TABLE IF EXISTS postings, document_chunks, documents CASCADE")
    conn.commit()
    conn.close()


@pytest.mark.parametrize("query", ["毛利率是多少", "EPS 66.25", "合併報表 負債總額", "股票代碼", "前一年"])
def test_sql_bm25_matches_reference(db, query):
    from src.lexical import BM25
    from src.main import bm25_search

    cur, ids, _ = db
    scores = BM25(CHUNKS).scores(query)
    expected = [ids[i] for i in sorted(range(len(CHUNKS)), key=lambda i: (-scores[i], i)) if scores[i] > 0]
    assert bm25_search(cur, query, 10) == expected


def test_deleting_a_document_removes_its_postings(db):
    cur, _, doc_id = db
    cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    cur.execute("SELECT count(*) FROM postings")
    assert cur.fetchone()[0] == 0
