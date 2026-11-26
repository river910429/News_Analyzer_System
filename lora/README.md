# Stage C: distilling the 14B agent into 7B with QLoRA

## Why

With the same harness, qwen2.5:14b answers 12/14 multi-company questions and qwen2.5:7b 9/14
(TSMC set: 11/12 both). The 7B errors left after three rounds of harness fixes are reading
errors — a quarterly dividend taken for the annual one, the dividend per share taken for EPS,
a global headcount given for one plant. Rules cannot catch those generally; the 14B model
mostly does not make them. The goal: 14B behaviour at 7B speed (3.7 s vs 6.2 s per answer).

## Method

```
annual reports of OTHER companies (ASE, Novatek, Realtek)
   │  make_questions.py   teacher writes questions + expected answers from numeric passages
   ▼
questions.jsonl  (~250 answerable + 15 unanswerable)
   │  collect.py          teacher (14B) answers through the real /agent harness
   │                      keep only clean, correct trajectories (rejection sampling)
   ▼
train.jsonl / val.jsonl   full conversations: system prompt, tool calls, tool results, answer
   │  train.py            QLoRA on Qwen2.5-7B-Instruct, loss on assistant turns only
   ▼
adapter ─ merge.py ─► merged fp16 ─ export.ps1 (llama.cpp) ─► GGUF Q4_K_M ─► Ollama qwen2.5-fin:7b
   │
   ▼
eval/run_eval.py --model qwen2.5-fin:7b   on TSMC / MediaTek / UMC (never seen in training)
```

Design choices:

- **Held-out companies.** Training data comes only from companies outside the evaluation
  sets, so a gain means the model learned how to read reports, not these reports' answers.
  `make_questions.py` refuses to run on TSMC, MediaTek or UMC.
- **Trajectories, not Q&A pairs.** The student learns the whole loop — which searches to
  issue, with which `company` argument, when to compute, when to stop — under the exact
  system prompt, tool schemas and tool outputs it sees at inference.
- **Rejection sampling.** A trajectory is kept only if the answer is right and the harness
  fixed nothing (no critique, no arithmetic correction, no tool call recovered from text).
  The student never imitates an error the harness had to repair.
- **Loss on assistant turns only.** Tool results and the prompt are context, not targets.
- **Two independent paths must agree.** The teacher writes the question and its answer while
  looking at the passage; later the teacher must find the answer by itself through search. A
  trajectory is kept only when both agree. A lookup answer must also appear in the passage.
- **Same quantization as the baseline.** The merged model is imported as q4_K_M, like the
  stock `qwen2.5:7b`, so 7B vs 7B-LoRA compares training, not precision.

## Run

Upload three training reports first (frontend or `/upload`), e.g. 日月光投控 (3711), 聯詠 (3034),
瑞昱 (2379) 114年年報. File names containing the company name are tagged automatically.

```powershell
# 1. questions (teacher: qwen2.5:14b), ~15 min
python lora/make_questions.py --api http://localhost:30080

# 2. teacher trajectories through the harness, ~30 min
python lora/collect.py --api http://localhost:30080 --model qwen2.5:14b

# 3. free the GPU, build the training image, train (~1-2 h on a 16 GB GPU)
ollama stop qwen2.5:14b; ollama stop qwen2.5:7b
docker build -t rag-trainer lora
docker run --rm --gpus all -v ${PWD}/lora:/work/lora -v hf-cache:/root/.cache/huggingface `
    rag-trainer python lora/train.py

# 4. merge on the CPU and import into Ollama
docker run --rm -v ${PWD}/lora:/work/lora -v hf-cache:/root/.cache/huggingface `
    rag-trainer python lora/merge.py
powershell -ExecutionPolicy Bypass -File lora\export.ps1

# 5. evaluate on the held-out companies
# re-run the stock 7B too: the harness changed since its last run (no_search check)
python eval/run_eval.py --api http://localhost:30080 --qa eval/agent_qa_multidoc.jsonl --agent --model qwen2.5 --tag final-7b
python eval/run_eval.py --api http://localhost:30080 --qa eval/agent_qa_multidoc.jsonl --agent --model qwen2.5-fin:7b --tag final-lora
python eval/run_eval.py --api http://localhost:30080 --qa eval/agent_qa.jsonl --agent --model qwen2.5 --tag final-7b-tsmc
python eval/run_eval.py --api http://localhost:30080 --qa eval/agent_qa.jsonl --agent --model qwen2.5-fin:7b --tag final-lora-tsmc
```

Merging needs ~16 GB of RAM inside Docker: set `memory=24GB` in `%UserProfile%\.wslconfig`
(the default is half the machine) and scale the K8s deployments to 0 while it runs.
Training peaked at 21 GiB at `--max-len 6144`, spilling past 16 GB of VRAM into shared memory
(130 s per step); `--max-len 4096` stays on the GPU.

Out of memory in step 3: `--max-len 4096` (drops the longest conversations) or
`--base Qwen/Qwen2.5-3B-Instruct` (then export with `-Base qwen2.5:3b -Name qwen2.5-fin:3b`).

## Data collection log

**Run 1: 20 of 148 kept.** Reading the rejections:

| Rejected | Count | Cause | Fix |
|---|---|---|---|
| no_search | 27 | 14B answered from memory; the harness had no check for this | new harness check `no_search` (also protects production) |
| critiqued | ~75 | mostly `early_refusal`: 14B said 無法得知 after one search, then succeeded when told to retry | keep and **repair**: drop the draft + critique pair |
| wrong_answer | 21 | partly bad questions | question filters below |
| garbage questions | — | 日月光's PDF has fonts without a Unicode map ("115ϋ Jamaican dollar"); vague "某數字", page/phone questions | skip passages with >1% non-report characters; reject vague or trivia questions in the prompt and by regex |

Sampling raised to 90 passages per company. The garbled pages also hurt 日月光 retrieval in
production; OCR for such pages is a separate fix.

## Training run

191 trajectories (6 longer than 6,144 tokens dropped), 2 epochs, 48 steps, 1 h 45 min.
Train loss 0.41 → 0.20; validation loss 0.177 → 0.170 (21 held-out trajectories, no overfitting).
40.4 M trainable parameters (0.53 % of 7.6 B).

## First evaluation and what was wrong with it

Same harness, stock `qwen2.5` vs `qwen2.5-fin:7b`: TSMC 9/12 → 10/12 with **0 critiques and 0
arithmetic fixes** (stock: 4 and 4) and 2.7 s vs 3.4 s per answer — but multi-company 8/14 → 6/14.
Two problems, both outside the weights:

- **Prompt template with CRLF.** `export.ps1` wrote the Modelfile with Windows line endings, so
  Ollama's ChatML template put `\r\n` into every turn — a format the model never saw in
  training. The tell: every fine-tuned answer contains `\r\n`, none of the stock model's do.
- **LaTeX arithmetic.** The student learned the teacher's `\[ \frac{a}{b} \times 100 = c \]`.
  The harness could not parse it, so wrong results went uncorrected (x-06 44.92 vs 44.98,
  x-08 34.52 vs 34.39; the stock model wrote plain `a / b * 100` and got fixed). The harness now
  converts LaTeX to plain arithmetic before checking.

Both models are re-evaluated after the fixes.

## Results

| Model (same harness) | multi-company 14 | TSMC 12 | s / answer |
|---|---|---|---|
| qwen2.5:7b | 9/14 | 11/12 | 3.7 |
| qwen2.5:14b (teacher) | 12/14 | 11/12 | 6.2 |
| qwen2.5-fin:7b (QLoRA), first run — invalid, see below | 6/14 | 10/12 | 3.5 |
| qwen2.5-fin:7b (QLoRA), fixed template + LaTeX parsing | *pending* | *pending* | |

Also compared: `harness_critiques` and `harness_fixes` in the eval summary — how often the
harness had to step in. Training should lower them: the harness catches errors, the
fine-tuned model makes fewer.
