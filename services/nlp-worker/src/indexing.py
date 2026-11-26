"""Inverted index for BM25, stored in PostgreSQL.

postings(term, chunk_id, tf): one row per distinct term in a chunk.
document_chunks.token_count: chunk length in tokens, for BM25 length normalisation.
Document frequency and average length are computed at query time, so adding or
re-indexing a document needs no global rebuild.
"""

from collections import Counter
from typing import Iterable, List, Sequence, Tuple

from psycopg2.extras import execute_values

from tokenizer import tokenize

SCHEMA_SQL = """
ALTER TABLE documents ADD COLUMN IF NOT EXISTS company TEXT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS token_count INTEGER;
CREATE TABLE IF NOT EXISTS postings (
    term TEXT NOT NULL,
    chunk_id INTEGER NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    tf INTEGER NOT NULL,
    PRIMARY KEY (term, chunk_id)
);
CREATE INDEX IF NOT EXISTS postings_chunk_id_idx ON postings (chunk_id);
"""


def term_frequencies(text: str) -> Counter:
    return Counter(tokenize(text))


def postings_rows(chunk_ids: Sequence[int], texts: Sequence[str]) -> Tuple[List[Tuple[str, int, int]], List[int]]:
    """(term, chunk_id, tf) rows for every chunk, plus each chunk's token count."""
    rows: List[Tuple[str, int, int]] = []
    lengths: List[int] = []
    for chunk_id, text in zip(chunk_ids, texts, strict=True):
        tf = term_frequencies(text)
        rows.extend((term, chunk_id, count) for term, count in tf.items())
        lengths.append(sum(tf.values()))
    return rows, lengths


def write_postings(cursor, chunk_ids: Sequence[int], texts: Iterable[str]) -> int:
    """Insert postings and token counts for newly inserted chunks. Returns the number of postings."""
    rows, lengths = postings_rows(chunk_ids, list(texts))
    execute_values(
        cursor,
        "UPDATE document_chunks AS c SET token_count = v.n FROM (VALUES %s) AS v(id, n) WHERE c.id = v.id",
        list(zip(chunk_ids, lengths, strict=True)),
        page_size=1000,
    )
    execute_values(cursor, "INSERT INTO postings (term, chunk_id, tf) VALUES %s", rows, page_size=1000)
    return len(rows)
