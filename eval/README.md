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
| page + sentence chunks | 500 chars, per page | vector | 0.581 | 0.860 | 0.930 | 0.953 | 0.718 |
| page + sentence chunks | 500 chars, per page | BM25 (CJK bigrams) | 0.744 | 0.907 | 0.930 | 0.977 | 0.831 |
| page + sentence chunks | 500 chars, per page | **hybrid (RRF)** | 0.698 | 0.884 | **0.977** | 0.977 | 0.799 |
| page + sentence chunks | 500 chars, per page | hybrid + reranker | | | | | |

Hybrid is the default: the LLM receives the top 5 chunks, and hybrid has the best Hit@5
(42/43 questions have the answer in the context, vs 37/43 for the baseline).

Compare retrievers on the running system:

```bash
python eval/run_eval.py --mode vector
python eval/run_eval.py --mode bm25
python eval/run_eval.py --mode hybrid
```

## Multi-step questions: RAG vs agent

`eval/agent_qa.jsonl` has 12 questions that need two facts and a calculation
(e.g. "毛利率比前一年增加幾個百分點？"). Answers are checked against the expected value;
a bare number must match on its own (so "3" does not match "113").

```bash
python eval/run_eval.py --qa eval/agent_qa.jsonl --answers   # plain RAG (/search)
python eval/run_eval.py --qa eval/agent_qa.jsonl --agent     # tool-calling agent (/agent)
```

| Run | answer_hit | avg tool calls | avg time (s) |
|---|---|---|---|
| RAG (/search), qwen2.5 7B | 0.583 | – | 1.3 |
| Agent v1 (/agent), qwen2.5 7B | 0.417 | 1.08 | 1.7 |
| Agent v2: new prompt + critique checks | | | |

Scores were re-checked with `--rescore` after fixing the checker ("4.0元" now matches "4").

**Agent v1 failure analysis** (7 wrong of 12):

| Failure | Count | Example |
|---|---|---|
| Gave up although the numbers were in the retrieved text | 4 | 前十大客戶 78% and 最大客戶 19% retrieved, answered "無法得知" |
| Mental arithmetic instead of the calculator | 1 | 900.8 × 35.9% instead of 1,224.2 − 900.8 |
| Wrong table column / wrong passage, no second search | 2 | quarterly dividend (p39) instead of the annual one (p7) |

Avg 1.08 tool calls: the model almost never searched twice or used the calculator.
v2 changes: a step-by-step system prompt with a worked example, and harness-side
**critique checks** that send a draft back once if it gives up after a single search or
answers a calculation question without calling the calculator.
