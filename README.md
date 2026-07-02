# Text-to-SQL with QLoRA

Fine-tuning Qwen2.5-1.5B-Instruct to convert plain English questions into SQL queries, using QLoRA on a free Colab T4.

Built for the IR InFotech Round 4 practical assessment (ML / Model Training / Model Fine-Tuning).

## The idea

Small instruct models are decent at general chat but unreliable at producing exact SQL for a given schema. The goal here is to show a measurable jump in SQL generation accuracy after a lightweight QLoRA fine-tune, comparing the base model vs the tuned one on a held-out set.

## Dataset

[b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) from Hugging Face. Each row has a natural language question, the table schema (as a CREATE TABLE statement), and the target SQL query. I sample 8k examples for training and 500 for evaluation after cleaning (dedup + length filtering). Prep details in `notebooks/01_data_prep.ipynb`.

## Project structure

```
notebooks/
  01_data_prep.ipynb    # dataset exploration, cleaning, train/eval split
  02_train_qlora.ipynb  # QLoRA fine-tuning (run this on Colab T4)
  03_evaluate.ipynb     # base vs fine-tuned comparison with metrics
data/
  train.jsonl           # 8k processed training examples
  eval.jsonl            # 500 held-out eval examples
docs/
  report.md             # problem statement, approach, results, challenges
```

## Stack

- Qwen2.5-1.5B-Instruct, loaded in 4-bit (bitsandbytes NF4)
- LoRA via peft (r=16, attention + MLP projections)
- trl SFTTrainer
- Free Colab T4 (16GB), fits comfortably in 4-bit

## Results

See `docs/report.md` for the full evaluation. Metrics used: exact match and normalized SQL match (case/whitespace insensitive) on 200 held-out examples, base vs fine-tuned.

## Running it

Local setup uses [uv](https://docs.astral.sh/uv/):

```
uv sync              # just datasets/pandas, enough for data prep
uv sync --extra train  # full training stack (only needed on a GPU box)
```

1. `01_data_prep.ipynb` runs anywhere (CPU is fine), regenerates `data/*.jsonl`
2. `02_train_qlora.ipynb` needs a GPU, made for Colab T4 (installs its own pins). Saves the LoRA adapter.
3. `03_evaluate.ipynb` also needs a GPU, loads base + adapter and scores both.
