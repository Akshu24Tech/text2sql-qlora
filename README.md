# Text-to-SQL — QLoRA fine-tuning of Qwen2.5-1.5B

Fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` with **QLoRA** (4-bit base + LoRA adapters) to turn a table schema plus a natural-language question into a SQL query.

Built for the IR InFotech Round 4 practical assessment (ML / Model Training / Model Fine-Tuning).

## The idea

Text-to-SQL is a concrete, useful task: given a `CREATE TABLE` schema and a question in plain English, produce the SQL that answers it. It's easy to check — you can compare the generated query against a reference — and it's a good fit for parameter-efficient fine-tuning, because a small instruction-tuned model already writes SQL but doesn't reliably match a target output style.

I take a small instruction model (Qwen2.5-1.5B-Instruct), fine-tune it with QLoRA on a labelled text-to-SQL dataset, and measure how much the fine-tuning changes things by comparing it against the **same model before fine-tuning** on a held-out eval set.

The GPU work runs on a free Colab **T4** (16GB): the base model loads in 4-bit, and only the LoRA adapter (~70MB) trains on top.

## Dataset

[b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) from Hugging Face — 78,577 examples, each a `(question, context, answer)` triple where `context` is a `CREATE TABLE` statement and `answer` is the target SQL. Deduplication + length filtering + a fixed 8,000/500 train/eval split are in `notebooks/01_data_prep.ipynb`, which writes `data/train.jsonl` and `data/eval.jsonl` so training and eval never drift apart.

## Project structure

```
notebooks/
  01_data_prep.ipynb     # load dataset, dedupe, length-filter, fixed train/eval split (CPU)
  02_train_qlora.ipynb   # QLoRA fine-tune Qwen2.5-1.5B on Colab T4
  03_evaluate.ipynb      # base vs fine-tuned, exact & normalized match on held-out eval
data/
  train.jsonl            # 8,000 train examples
  eval.jsonl             # 500 held-out examples
results/
  metrics.json           # base vs fine-tuned scores
  predictions.json       # per-example base + tuned + reference SQL on the eval subset
  adapter_model.safetensors  # the trained LoRA adapter (gitignored — regenerate via notebook 2)
docs/
  report.md              # problem statement, approach, results, challenges
```

## Model & method

- **Base:** `Qwen/Qwen2.5-1.5B-Instruct` (1.5B params), loaded in 4-bit NF4 with double quantization, fp16 compute (the T4 has no bf16).
- **Adapter:** LoRA, `r=16`, `alpha=32`, dropout `0.05`, on all attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`).
- **Training:** 1 epoch over the 8,000 examples with `trl`'s `SFTTrainer`, effective batch size 16 (batch 4 × grad-accum 4), learning rate `2e-4`, cosine schedule, gradient checkpointing. Only the adapter is saved (~70MB).
- **Prompt format:** the model's own chat template — a fixed system prompt, a user turn carrying the schema + question, and the assistant turn as the target SQL — so inference at eval time matches training exactly.

## Results

See `docs/report.md` for the full write-up. Short version: on the held-out eval subset the fine-tuned adapter did **not** move the numbers relative to the base model — both score the same, and the discussion in the report explains what happened and why the string-match metric is harsh here.

## Live demo

A Gradio demo of the fine-tuned model is deployable to a free Hugging Face Space (paste a schema + question, get SQL back). App, config, and step-by-step deploy instructions are in [`deploy/hf-space/`](deploy/hf-space/DEPLOY.md).

## Running it

Notebook 1 runs locally (CPU) for data prep:

```
uv sync
```

Then open `notebooks/02_train_qlora.ipynb` and `notebooks/03_evaluate.ipynb` in **Google Colab** with a **T4 GPU** runtime and run them in order — they install the GPU stack (bitsandbytes / peft / trl / accelerate) themselves and read the data straight from the repo over HTTPS. The adapter is saved to Google Drive between the two notebooks.
