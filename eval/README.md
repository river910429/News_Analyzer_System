# RAG Evaluation

Measures how well retrieval finds the right passage, so changes (chunking, hybrid search, reranker, model) can be compared with numbers instead of impressions.

## Metrics

| Metric | Meaning |
|---|---|
| Hit@k | Share of questions whose relevant chunk is in the top k results |
| MRR | Mean of 1 / rank of the first relevant chunk (higher = found earlier) |
| answer_hit | Share of LLM answers that contain the expected value (`--answers`) |

A chunk is **relevant** when it contains every string in the question's `keywords` (whitespace ignored, because PDF extraction often breaks numbers across lines).

## Run

```bash
docker compose up -d                 # system running, PDF uploaded and "completed"
python eval/run_eval.py --tag baseline
python eval/run_eval.py --tag baseline --answers   # slower: calls the LLM per question
```

Results are saved to `eval/results/<time>_<tag>.json`. Re-run with a new `--tag` after each change and compare the summaries.

## Results

QA set: 43 questions on *2025 TSMC Annual Report* (`eval/qa.jsonl`), top-10 retrieval.

| Run | Chunking | Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|---|---|
| baseline | fixed 500 chars | vector (bge-m3) | 0.581 | 0.814 | 0.884 | 0.977 | 0.719 |
| page + sentence chunks | 500 chars, per page | BM25 (offline check) | 0.791 | 0.953 | 0.953 | 1.000 | 0.872 |
| page + sentence chunks | 500 chars, per page | vector | | | | | |
| page + sentence chunks | 500 chars, per page | hybrid (RRF) | | | | | |

Compare retrievers on the running system:

```bash
python eval/run_eval.py --mode vector
python eval/run_eval.py --mode bm25
python eval/run_eval.py --mode hybrid
```
