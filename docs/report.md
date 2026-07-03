# Text-to-SQL with QLoRA (Qwen2.5-1.5B) — Project Report

**Author:** Akshu Grewal
**Assessment:** IR InFotech Round 4 (ML / Model Training / Model Fine-Tuning)

## Problem Statement

Text-to-SQL is the task of turning a natural-language question into a SQL query against a known database schema. Given a `CREATE TABLE` statement and a question like *"How many heads of the departments are older than 56?"*, the model should output `SELECT COUNT(*) FROM head WHERE age > 56`.

I picked it because it's a real, useful problem with an objective target for every example (there's a reference query to compare against), and because it's a natural fit for parameter-efficient fine-tuning: a small instruction-tuned model can already write SQL, so fine-tuning is about steering *how* it answers — bare SQL in a specific style — rather than teaching SQL from scratch.

## Solution Approach

Take a small instruction model, `Qwen/Qwen2.5-1.5B-Instruct`, and fine-tune it with **QLoRA**: load the base model in 4-bit, freeze it, and train only a small LoRA adapter on top. Then check what the fine-tuning did by comparing two models on the same held-out eval set:

1. **Qwen2.5-1.5B-Instruct, before fine-tuning** — the honest starting point.
2. **Qwen2.5-1.5B-Instruct + QLoRA adapter** — the same model with the trained adapter loaded on top.

QLoRA is what makes this fit on a free Colab T4 (16GB): 4-bit quantization shrinks the frozen base, and only the ~70MB adapter is trained and saved.

## Dataset & Preprocessing

**Dataset:** [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) — 78,577 `(question, context, answer)` triples, where `context` is a `CREATE TABLE` schema and `answer` is the reference SQL.

**Preprocessing** (`notebooks/01_data_prep.ipynb`, runs on CPU):
- Deduplicated on the `question` field (78,577 → 78,311). The dataset pairs some identical questions with near-identical schemas; leaving them in risks leaking the same example across the train/eval boundary.
- Length-filtered on character counts (question < 300, context < 500, answer < 300) so full prompts fit comfortably in the token budget — dropped a further ~70 rows, leaving 78,243.
- Sampled 8,500 rows with a fixed seed (42) and split into **8,000 train / 500 eval**. Both splits are written to `data/train.jsonl` and `data/eval.jsonl` and committed, so the training and eval notebooks always see the same data.

## Model Selection

**Qwen2.5-1.5B-Instruct.** Reasons:
- It's small (1.5B params) and, quantized to 4-bit, trains on a single free T4 — no paid hardware needed.
- It's already instruction-tuned and writes reasonable SQL out of the box, so the fine-tune only has to adjust output style/format, which is exactly what LoRA is good at.
- Its chat template gives a clean, consistent prompt format to train and evaluate against.

**Prompt format:** each example is a short chat — a fixed system prompt (*"reply with only the SQL query, nothing else"*), a user turn with the schema and question, and the assistant turn as the target SQL. Using the model's own chat template means eval-time inference looks exactly like training.

## Training / Fine-Tuning Process

QLoRA fine-tuning with `trl`'s `SFTTrainer` (`notebooks/02_train_qlora.ipynb`, Colab T4):
- **Quantization:** 4-bit NF4 with double quantization, fp16 compute dtype (the T4 has no bf16).
- **LoRA:** `r=16`, `alpha=32`, dropout `0.05`, `bias="none"`, applied to all attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`).
- **Optimization:** 1 epoch over 8,000 examples, per-device batch 4 × gradient accumulation 4 (effective batch 16), learning rate `2e-4`, cosine schedule with 3% warmup, gradient checkpointing, seed 42.
- One epoch takes roughly 2–3 hours on the T4. Only the LoRA adapter (~70MB) is saved (to Google Drive, so it survives a runtime reset); the eval notebook loads it back on top of a fresh 4-bit base.

Two guardrails are baked into the notebooks, both from problems I actually hit (see Challenges):
- The trainable LoRA params are upcast to fp32 before training, because the newer `trl`/`peft` default of bf16 adapters crashes the fp16 gradient scaler on a T4.
- Before saving *and* before evaluating, an assertion checks that the adapter's `lora_B` matrices aren't all zero — a fresh, untrained adapter has them at exactly zero, and that would silently produce base-identical predictions.

## Evaluation Results

Both models were scored on the same 200-example slice of the held-out eval set (`notebooks/03_evaluate.ipynb`), generating greedily one example at a time. Two metrics:

- **Exact match** — generated SQL equals the reference, character for character.
- **Normalized match** — same, but case-insensitive, whitespace collapsed, and a trailing `;` stripped. This is the fairer number, since `SELECT Name` vs `select name` is really the same query.

| Model | Exact match | Normalized match |
|---|---|---|
| Qwen2.5-1.5B (before fine-tuning) | 0.04 | 0.05 |
| Qwen2.5-1.5B + QLoRA (fine-tuned) | 0.04 | 0.05 |

**The fine-tuning did not change the results.** On this eval run the tuned model produced output *identical to the base model on all 200 examples*, so the two rows above are the same by construction. `results/predictions.json` holds the per-example base / tuned / reference outputs and confirms this directly.

Two things are going on, and both are worth being honest about:

1. **The adapter had no measurable effect on generation.** Identical outputs on every single example means the trained adapter effectively didn't steer the model at eval time — the tuned model still wraps answers in ` ```sql … ``` ` fences and still uses the base model's quoting style, exactly the behaviours the fine-tune was meant to remove. The most likely causes are the adapter not being applied to the generating model, or a training run whose signal was too weak (1 epoch, and an eval on the saved artifact rather than a re-verified in-session adapter) to shift greedy decoding. Resolving this would mean re-running training and eval end-to-end in one session and confirming the adapter changes at least some predictions before trusting the score.

2. **The string-match metric is harsh, which is why *both* scores are so low (~4–5%).** Many base predictions are semantically correct but fail the match: the model emits ` ```sql ` code fences, writes `WHERE grid = '9'` where the reference has `WHERE grid = "9"`, or picks a defensible-but-different aggregate (`SELECT silver` vs the reference's `SELECT AVG(silver)`). Under exact/normalized string comparison all of these count as wrong even when the SQL is reasonable. This is precisely the gap fine-tuning was supposed to close — align the output format and quoting with the dataset's conventions — and the failure above is that it didn't.

So the honest headline is: the pipeline runs end-to-end and produces a clean before/after comparison, but this fine-tuning run delivered no improvement, and the report reflects that rather than dressing it up.

## Challenges Faced

- **bf16 adapters on a T4.** The current `trl`/`peft` defaults create LoRA weights in bf16, which crashes the fp16 gradient scaler on the first optimizer step (the T4 has no bf16 support). Fix: upcast the trainable params to fp32 — the standard QLoRA setup of fp32 adapters over an fp16-compute frozen base.
- **Silently untrained adapters.** A freshly initialized LoRA adapter has all-zero `lora_B` matrices and produces predictions identical to the base model. If training and saving happen in different Colab sessions, it's easy to save a dead adapter without noticing. I added an assertion (`max |lora_B| > 0`) before both saving and evaluating to catch this. Notably, this same failure mode — base-identical predictions — is what the eval run above exhibits, which points at the adapter not actually being in effect during generation.
- **A metric that punishes correct-but-differently-styled SQL.** Exact/normalized string match treats `'9'` vs `"9"`, code fences, and equivalent aggregates as wrong. It's simple and objective, but it undersells any model that's semantically right in the wrong format — and it makes format-alignment the whole game for the fine-tune.
- **Free-tier constraints.** Everything GPU runs on a Colab T4, which shaped most of the choices above: a 1.5B model, 4-bit loading, fp16 not bf16, one epoch, and saving the adapter to Drive so it survives runtime resets.

## Possible Improvements

- **Fix and re-verify the adapter application.** Run training → save → eval in one session, assert the tuned predictions differ from base on a handful of examples before scoring, and only then report numbers. This is the first thing to do — the current result is inconclusive, not a genuine "fine-tuning doesn't help".
- **A fairer metric.** Compare on *execution* (run both queries against a small SQLite instance built from the schema and check the result sets match), or at least strip code fences and normalize quoting before matching. This decouples "is the SQL correct" from "does the string match".
- **Stronger training signal.** More epochs, more data, or masking the loss to the assistant (SQL) tokens only so the model isn't spending capacity re-predicting the schema and question.
- **Error analysis on real differences.** Once the adapter demonstrably changes outputs, categorize where it helps and where it still fails (joins, aggregates, multi-condition `WHERE`) to guide the next iteration.
