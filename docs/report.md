# Text-to-SQL with QLoRA — Project Report

**Author:** Akshu Grewal
**Assessment:** IR InFotech Round 4 (ML / Model Training / Model Fine-Tuning)

## Problem Statement

Small open LLMs are handy for cheap, local inference, but out of the box they are unreliable at producing exact SQL for a given schema. They paraphrase, add markdown fences, or invent column names. The goal of this project is to take a small instruct model (Qwen2.5-1.5B-Instruct) and fine-tune it with QLoRA so it reliably converts a plain English question plus a table schema into the correct SQL query, and to measure that improvement properly against the base model.

I picked text-to-SQL because it has an objective right answer per example. That makes the before/after comparison honest, no vibes-based evaluation.

## Solution Approach

1. Clean and split a public text-to-SQL dataset into fixed train/eval sets (committed to the repo so results are reproducible).
2. Fine-tune the base model with QLoRA on a free Colab T4.
3. Score base vs fine-tuned on the same held-out examples with exact match and normalized match.

## Dataset & Preprocessing

**Dataset:** [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) (~78.6k examples). Each row: natural language question, schema as a CREATE TABLE statement, target SQL.

**Preprocessing** (`notebooks/01_data_prep.ipynb`):
- Dropped duplicate questions to avoid train/eval leakage (the same question shows up with near-identical schemas).
- Filtered out overly long rows (question > 300 chars, schema > 500, answer > 300) so full prompts fit a 512 token budget on the T4.
- Sampled 8,000 training and 500 eval examples with a fixed seed (42). Both files are committed as JSONL.

**Prompt format:** the model's own chat template. System prompt pins the task ("reply with only the SQL query"), user turn carries schema + question, assistant turn is the target SQL. Inference uses the identical format, so there's no train/inference mismatch.

## Model Selection

**Qwen2.5-1.5B-Instruct.** Reasons:
- Small enough to fine-tune and run in 4-bit on a free T4 (16GB) with headroom.
- Instruct-tuned already, so it follows the output format quickly and the fine-tune only has to teach SQL precision, not instruction following from scratch.
- Strong base quality for its size compared to alternatives I considered (TinyLlama-1.1B, Phi-2).

## Training / Fine-Tuning Process

QLoRA (`notebooks/02_train_qlora.ipynb`):
- Base model loaded in 4-bit NF4 with double quantization, fp16 compute (T4 has no bf16).
- LoRA: r=16, alpha=32, dropout 0.05, on all attention and MLP projections (q/k/v/o/gate/up/down). Attention-only adapters underperformed in a quick smoke test.
- 1 epoch over 8k examples, effective batch 16 (4 per device x 4 accumulation), lr 2e-4 cosine with 3% warmup, max sequence 512, gradient checkpointing on.
- Trainable params: <!-- TODO: from print_trainable_parameters -->
- Training time on T4: <!-- TODO -->
- Final training loss: <!-- TODO: from trainer logs -->

Only the LoRA adapter (~70MB) is saved and pushed to the Hub, not the full model.

## Evaluation Results

200 held-out examples, greedy decoding, same prompts for both models (`notebooks/03_evaluate.ipynb`).

| Model | Exact match | Normalized match |
|---|---|---|
| Base Qwen2.5-1.5B-Instruct | <!-- TODO --> | <!-- TODO --> |
| + QLoRA fine-tune | <!-- TODO --> | <!-- TODO --> |

Normalized match ignores case, extra whitespace and a trailing semicolon, since those don't change the query. Exact match is stricter than the task really requires (a correct query written differently counts as wrong), so normalized match is the headline number; note that both metrics still under-count semantically-equivalent-but-differently-written queries.

<!-- TODO: 2-3 sentences interpreting the numbers + reference the fixed/still-wrong examples from the notebook -->

## Challenges Faced

- **T4 memory budget.** Full fine-tuning a 1.5B model doesn't fit; 4-bit NF4 + LoRA + gradient checkpointing does, with the sequence length capped at 512. The length filtering in data prep follows directly from this.
- **Base model output noise.** The base model loves wrapping SQL in markdown fences or explaining itself, which tanks exact match for formatting rather than correctness reasons. The system prompt plus fine-tuning fixed the format; the normalized metric keeps the base comparison fair.
- **Metric honesty.** String match under-counts correct answers (e.g. `WHERE a=1 AND b=2` vs the flipped order). Proper execution accuracy would need to actually build and run each schema; noted under improvements.
- <!-- TODO: add anything real that comes up during the run -->

## Possible Improvements

- **Execution accuracy:** spin up each CREATE TABLE in SQLite, run predicted and reference queries, compare result sets. Much fairer metric than string match.
- **Harder benchmark:** evaluate on Spider for multi-table joins; sql-create-context is mostly single-table.
- **Hyperparameter sweep:** r, learning rate, and 2-3 epochs were not swept due to the 48h window; the current config is a sensible first pick, not a tuned one.
- **DPO pass:** after SFT, a small preference dataset of (correct, plausible-but-wrong) SQL pairs could sharpen precision further.
