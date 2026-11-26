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
| page + sentence chunks | 500 chars, per page | hybrid + reranker (bge-reranker-base) | 0.698 | 0.930 | 0.930 | 0.953 | 0.801 |
| + labelled table chunks | as above + pdfplumber tables | **hybrid** | 0.698 | **0.930** | **0.977** | **1.000** | **0.807** |

The reranker was not adopted: Hit@5 dropped from 0.977 to 0.930 and each query got slower.
With table chunks every question's answer is in the top 10, including the one no setup found before.

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
| Agent v2: new prompt + critique checks | **0.750** | 1.50 | 4.2 |
| Agent v2 prompt, **qwen3 8B** | 0.583 | 3.83 | 25.0 |
| **Agent v3 + labelled table chunks (pdfplumber)** | **0.833** | 1.50 | 4.4 |

qwen3 8B searched more (3.8 tool calls) but was 6× slower and twice copied the prompt's
worked example as its answer to an unrelated question, so qwen2.5 7B stays the default.

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

**Agent v2 remaining failures** (3 of 12):

| Failure | Cause |
|---|---|
| Dividend: used quarterly dividends (p39) instead of the annual 18.0 vs 14.0 (p7) | retrieval picked the wrong passage; the model wrote `calculator("6.0 - 5.0")` as text instead of calling it, copying the call syntax from the prompt's example |
| Debt ratio 23.4%: consolidated assets ÷ **parent-only** liabilities | page 64 holds both balance sheets and pypdf loses the table headers, so the two tables cannot be told apart |
| Professional staff 46.8% | a number read from a garbled table on p59 |

The table problems point at ingestion, not the agent: a layout-aware PDF parser that keeps
table headers would fix them at the source.

**Fix at the source: labelled table chunks.** The worker now also reads every page with
pdfplumber, splits two-page spreads into halves, and stores each number table as extra chunks
that start with the table title and column header, e.g.
`【台積公司及子公司財務報表（合併報表）】 / 項目 民國114年 民國113年 增減金額 …`.
Offline check: retrieval metrics are unchanged (BM25 Hit@1 0.791 → 0.791), and consolidated vs
parent-only rows are now distinguishable.

**Remaining agent failures** (2 of 12) are the model reading the wrong cell even from a
labelled table: parent-only liabilities divided by consolidated assets, and the 113 column of
the staff table instead of 114. Next options: a stronger model for comparison, or returning
tables to the agent as structured rows (year → value) instead of text.

**Summary on the 12 multi-step questions:** plain RAG 0.583 → agent v1 0.417 → v2 0.750 →
v3 + table chunks **0.833**, at about 3 s extra per question.

## Tokenization for BM25

`python eval/tokenizer_benchmark.py` re-indexes the chunks with each tokenizer and runs both
QA sets. Results on the server's 544 chunks of the TSMC report:

| Tokenizer | qa.jsonl Hit@1 | Hit@5 | MRR | paraphrase Hit@5 | MRR | index time |
|---|---|---|---|---|---|---|
| unigram | 0.558 | 0.930 | 0.701 | 0.444 | 0.256 | 0.0 s |
| **bigram (production)** | **0.767** | **0.953** | **0.853** | **0.444** | **0.354** | 0.0 s |
| unigram + bigram | 0.744 | 0.953 | 0.829 | 0.389 | 0.346 | 0.1 s |
| jieba | 0.581 | 0.907 | 0.718 | 0.278 | 0.178 | 1.0 s |
| jieba + finance dictionary | 0.628 | 0.907 | 0.736 | 0.111 | 0.114 | 0.4 s |
| HanLP 2.x (coarse ELECTRA small) + finance dictionary | 0.605 | 0.930 | 0.744 | 0.278 | 0.172 | 27 s |
| CKIP (BERT base, Traditional Chinese) | 0.651 | 0.907 | 0.762 | 0.444 | 0.315 | 429 s (CPU) |

Every word segmenter, including the two neural ones and CKIP trained on Traditional Chinese,
lost to plain character bigrams, and they were 100-10,000x slower to index. Segmentation errors
and mismatched granularity between query and document (毛利 vs 毛利率, 營收 vs 營業收入淨額)
cost more than whole words gain. Bigrams never mis-segment and overlap partially matching words.

## Paraphrased questions: the benchmark was too easy

`eval/qa_paraphrase.jsonl` asks 18 of the questions the way a user would ("台積電EPS多少？",
"總負債", "N2 什麼時候量產", "碳中和"). Same relevant chunks, different words.
BM25 Hit@5 drops from 0.953 to **0.389**: the original questions reuse the report's wording.

Synonym expansion for BM25 (`query_expansion.py`, generic finance vocabulary):

| BM25 (inverted index in PostgreSQL) | qa.jsonl Hit@5 / MRR | paraphrase Hit@5 / MRR |
|---|---|---|
| no expansion | 0.953 / 0.871 | 0.389 / 0.334 |
| synonym expansion | 0.930 / 0.835 | **0.722 / 0.577** |

On the server, hybrid retrieval (vector + BM25):

| Hybrid | qa.jsonl Hit@1 / Hit@5 / MRR | paraphrase Hit@1 / Hit@5 / MRR |
|---|---|---|
| no expansion | 0.698 / 0.977 / 0.807 | 0.444 / 0.556 / 0.505 |
| **synonym expansion (default)** | 0.674 / 0.953 / 0.788 | **0.611 / 0.833 / 0.681** |

Expansion costs one question on the original set and gains five on the paraphrased one;
real users ask in their own words, so it is on by default.

Caveat: the dictionary was written after reading these paraphrases, so the paraphrase gain is
optimistic. A fair test needs a new held-out set of paraphrases written without looking at the
dictionary. Run on the server (hybrid, with and without expansion):

```bash
python eval/run_eval.py --mode hybrid --qa eval/qa_paraphrase.jsonl
python eval/run_eval.py --mode hybrid --qa eval/qa_paraphrase.jsonl --no-expand
```

## Harder benchmark: three companies

With one document, "毛利率是多少" has one right chunk anywhere in the index. With the TSMC,
MediaTek (聯發科) and UMC (聯電) 2025 annual reports indexed together, retrieval must also pick
the right company, and the year columns are ordered differently (TSMC and UMC list 114 first,
MediaTek lists 113 first).

| File | Questions | Checks |
|---|---|---|
| `qa.jsonl` / `qa_paraphrase.jsonl` | 43 / 18 TSMC | `doc: TSMC` — hits must come from the TSMC file |
| `qa_multidoc.jsonl` | 24 MediaTek + 18 UMC | `doc` restricts relevant chunks to that company's file |
| `agent_qa_multidoc.jsonl` | 10 cross-company calculations + 4 unanswerable | answers, and refusing when the report has no answer |

The unanswerable questions mention things that are in the reports (印度 appears only as a patent
region, the Kumamoto fab exists but its headcount is not given), so a system that answers from
loosely related passages will make something up.

Upload the reports with the company in the file name, e.g. `MediaTek_2025_AR.pdf` and
`UMC_2025_AR.pdf`, then:

```bash
python eval/run_eval.py --mode hybrid --tag multidoc-tsmc
python eval/run_eval.py --mode hybrid --qa eval/qa_multidoc.jsonl --tag multidoc-mtk-umc
python eval/run_eval.py --qa eval/agent_qa_multidoc.jsonl --answers --tag multidoc-rag
python eval/run_eval.py --qa eval/agent_qa_multidoc.jsonl --agent --tag multidoc-agent
```

### Results with three reports indexed (Kubernetes deployment)

| Set | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| TSMC 43 questions, TSMC only indexed | 0.674 | 0.907 | 0.953 | 1.000 | 0.788 |
| TSMC 43 questions, three reports indexed | 0.628 | 0.791 | 0.884 | 0.977 | 0.731 |
| MediaTek + UMC 42 questions | 0.643 | 0.833 | 0.929 | 1.000 | 0.762 |
| + company filter (NER) — TSMC | 0.628 | 0.814 | 0.884 | 1.000 | 0.736 |
| + company filter (NER) — MediaTek + UMC | **0.667** | **0.881** | **0.976** | 1.000 | **0.799** |

The filter helped MediaTek/UMC (Hit@5 0.929 → 0.976) but not TSMC. The reason was in the
benchmark: 11 TSMC questions were written for a one-document index and do not name the
company ("合併資產負債表中114年負債總額…"), so there is nothing to filter on — and with three
reports the question really is ambiguous. Those 11 questions now start with 台積公司
(original wording kept in `single_doc_question`).

Adding two more reports pushed TSMC answers down: other companies' balance sheets and
financial ratios compete for the same terms (負債總額 fell to rank 10). The agent got 4 of the
first 11 multi-company questions right, usually after a single search that returned only one
company's numbers.

**Company filter.** `companies.py` recognises companies in the question (台積電 / 台積公司 / TSMC /
2330 → TSMC) and links them to documents tagged at indexing time. Retrieval then searches only
those documents; for comparisons each company is searched separately and the results interleaved.
The agent's `search_report` tool takes a `company` argument and is told to search each company
once. Compare with `--no-company-filter`.

### Company filter, TSMC questions naming the company

| TSMC 43 questions | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| one report indexed | 0.674 | 0.907 | 0.953 | 1.000 | 0.788 |
| three reports, no filter | 0.628 | 0.791 | 0.884 | 0.977 | 0.731 |
| **three reports, company filter** | **0.674** | **0.884** | **0.930** | 0.977 | **0.784** |

With the company named and filtered on, TSMC retrieval is back to its single-document level.

### Agent collapse: the context window

After telling the agent to search each company separately, its accuracy on the multi-company set
fell to 1/14, with answers about unrelated passages, tool calls printed as text and stray tokens.
Cause: Ollama's default context is 4,096 tokens, and two searches of five ~500-character passages
already exceed it. Ollama then silently drops the oldest messages — the system prompt and tool
definitions first. Fix: `num_ctx` 16,384, and a warning when a prompt nears the limit.

Correction: raising `num_ctx` did not change a single answer (temperature 0.1 reproduced them
almost word for word), so the context window was not the cause. The answers themselves showed it:

- **Schema-shaped arguments.** The model sent `{"company": {"type": "string", "value": "聯電"}}`
  and `{"query": {"text": "…"}}` — the tool raised, and the model looped on search_report
  (up to 10 calls) or gave up. The harness now unwraps such values before calling the tool.
- **The worked example was read as data.** Answers reused its numbers and wording ("第一步：呼叫
  search_report…", TSMC's EPS 66.25/45.25 reported as UMC's, "根據提供的範文推測"). The example
  is removed; the rules stay.
- The eval now stores each step's arguments and result, so the next failure can be read from the
  trace instead of inferred from the answer.

### Argument fix, then harness checks on the answer

With arguments unwrapped, the agent answered 4/14 (2 tool calls per question, 4.2 s). The
remaining errors fell into groups that the harness can check without knowing the answer:

| Error in the run | Example | Harness check |
|---|---|---|
| Mental arithmetic, wrong result | `199,140,569 / 578,996,009 * 100 = 34.57%` (true: 34.39) | Recompute every `expression = result` in the answer; if wrong, send back the true value |
| Number from another period | "115年第一季" answered with the full-year 114 revenue | Years and quarters in the question (114年 ↔ 2025, 第一季 ↔ Q1) must appear in the retrieved text |
| A stated figure derived from other numbers | Kumamoto headcount = total staff − technicians − assistants | A lookup question (no 增加/相差/占/比…) answered with arithmetic is sent back: copy the figure or say it is unavailable |
| Correct abstention scored as wrong | "財報中並未明確提到聯發科在印度市場的營收" | Metric fix: unanswerable questions accept 並未 / 未提及 / 未提供 as well as 無法得知 |

The checks run on the full tool results (steps keep only the first 2,000 characters), and the
critique budget is 3. Still open, waiting on the traces: picking the wrong number from a correct
passage — MediaTek's dividend (29.00) taken as its EPS, the first-half dividend instead of the
annual total, the wrong gross-margin line.

| Multi-company agent (14) | answer_hit |
|---|---|
| before the company filter (first 11 questions) | 4/11 |
| `company` argument added | 1/14 |
| `num_ctx` 16k (not the cause) | 0/14 |
| arguments unwrapped, example removed | 4/14 |
| harness checks on the answer | 6/14 |
| text tool calls recovered, number grounding, guardrail | 6/14 |
| grounding bug fixed, arithmetic corrected, Chinese numerals annotated | **9/14** |

**Second round.** The checks fixed x-02 (arithmetic), u-02 (period) and, with the metric fix,
u-01. What was left:

- **Tool calls printed as text** (x-08 with 0 tools, u-04): `search_report({...})` repeated three
  times, and qwen2.5's native `{"name": …, "arguments": …}</tool_call>` leaking into the content.
  The "you only mentioned the tool" critique did not help — the model repeated the text. The
  harness now parses such calls and executes them (marked `recovered` in the trace), the usual
  fallback in agent frameworks for models with unreliable tool-call formatting.
- **Feedback ignored** (u-03): sent back three times, the model kept subtracting staff
  categories. When the budget runs out on a *period* or *derived-number* failure, the harness
  now replaces the answer with an abstention (`stopped_reason = guardrail:…`). For a
  financial assistant, "not in the document" is a better failure than a made-up figure.
- **Numbers that are not in the passages**: TSMC multi-10 divided by 95,570 (the passage says
  90,557); multi-12 wrote 1,224.2億 as `1,22420000000`. Every operand of a calculation must
  now occur, as a whole number, in the retrieved text or in an earlier calculation result.

### The single-company baseline was inflated

On the 12 TSMC questions the agent now scores 7/12, against 10/12 earlier. The earlier prompt's
worked example used TSMC's real gross margins — 59.9% vs 56.1%, giving 3.8 — which is exactly
question multi-01. The example leaked a test answer into the prompt, so 10/12 overstated the agent; 7/12 is the first
clean number. Remaining TSMC errors are reasoning, not mechanics: quarterly instead of annual
dividends (multi-04), subtracting the second-largest customer as well as the largest
(multi-07), and abstaining on the 112 customer share (multi-05).

### Third round: reading the traces

The r2 traces (arguments and results stored per step) overturned two assumptions.

**The grounding check was wrong, not the model.** It fired on correct answers — x-08's
`199,140,569 / 578,996,009` was flagged although both numbers are in the passage. The check
removed all whitespace before matching, which glues table cells together
(`199,140,569 192,015,673` → `199140569192015673`), so no table number ever matched. Three
false critiques per question then pushed the model into worse answers (x-06 44.98 → 44.92).
Numbers are now matched with thousands separators removed and spaces kept; a test uses the
real table row.

**The model ignores arithmetic feedback, so the harness corrects it.** "34.58 is wrong, the
result is 34.39" was ignored repeatedly. Wrong `expression = result` values are now rewritten
in the answer (and wherever the answer repeats them, re-checking chained expressions), with
each fix stored in `corrections`. Replaying r2's answers: x-06, x-08 and multi-08 are fixed
without another LLM call.

**Chinese numerals were the retrieval root cause.** The passages the agent never found:

| Question | Sentence in the report | Why it was missed |
|---|---|---|
| x-01 MediaTek EPS | 民國一一四年…每股盈餘為新台幣66.16元 | query "114年" ≠ "一一四年" |
| x-05 MediaTek dividend | 民國一一四年總計配發每股新台幣54元 | same |
| x-10 UMC 113 EPS | 民國一一三年…每股普通股獲利為新臺幣3.80元 | year form, and 每股普通股獲利 ≠ 每股盈餘 |
| multi-10 TSMC staff | 全球員工總數為9萬557人 | the model read 95570 |
| multi-12 TSMC USD revenue | 1,224億2,000萬美元 | the model wrote 1,22420000000, then 32.34億 |

Search instead returned tables headed "114年度", which hold other figures (MediaTek's 29.00
first-half dividend was taken as its EPS). `numerals.py` now annotates these forms at index
time and in tool output — 民國一一四年〔114年〕, 9萬557〔90,557〕, 1,224億2,000萬〔1,224.2億 =
122,420,000,000〕 — keeping the original text for citations; evaluation strips the 〔〕 before
keyword matching. 每股盈餘 → 每股普通股獲利 was added to the synonym dictionary. Requires a
reindex, after which retrieval metrics are re-measured too.

### Results after the third round

| Agent | before | after |
|---|---|---|
| multi-company (14) | 6/14 | **9/14** |
| TSMC (12, clean baseline without the leaked example) | 7/12 | **11/12** |
| avg answer time, multi-company | 4.6 s | 3.7 s |

Arithmetic correction fixed 5 answers outright (x-06, x-07, x-08, multi-08, multi-10); the
numeral annotations brought in the passages for x-01 (MediaTek EPS 66.16), multi-05 (the 112 customer share) and multi-12
(1,224.2億 − 900.8億). multi-07 also flipped to correct, which is run-to-run variance, not a fix.

Retrieval after the reindex (annotated chunks):

| Hit@5 / MRR | before | after |
|---|---|---|
| TSMC 43, three reports, company filter | 0.930 / 0.784 | **0.953 / 0.818** |
| MediaTek + UMC 42 | 0.976 / 0.799 | 0.976 / 0.801 |

What is left is the model's reading, not the plumbing: quarterly instead of annual dividends
(x-05, multi-04: 6.0 per quarter vs 18.0 per year), the dividend per share taken as EPS
(x-10: 2.60 vs 3.34), a wrong UMC gross margin (x-02, x-03), and the global headcount given as
the Kumamoto plant's (u-03). These are what the model-size comparison and LoRA stage target.

### Model size: qwen2.5:14b under the same harness

| Same harness | multi-company 14 | TSMC 12 | s / answer |
|---|---|---|---|
| qwen2.5:7b | 9/14 | 11/12 | 3.7 |
| qwen2.5:14b | **12/14** | 11/12 | 6.2 |

14B fixed exactly the reading errors (annual vs quarterly dividends, UMC gross margin, the
Kumamoto headcount) and needed far fewer critiques. Its own misses: the dividend per share
taken for UMC's EPS (both models — the EPS sentence has no year in it), a hallucinated India
revenue figure, and multi-11 answered without reading the passage. This gap is the target of
the QLoRA stage (`lora/`): distil 14B's trajectories into 7B.
