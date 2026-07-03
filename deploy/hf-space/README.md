---
title: Text2SQL QLoRA
emoji: 🗄️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Text-to-SQL — Qwen2.5-1.5B QLoRA

Live demo for the text-to-SQL fine-tuning project. Paste a `CREATE TABLE` schema
and a natural-language question; the model returns a SQL query.

- **Base model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Method:** QLoRA (4-bit training on Colab T4), LoRA adapter loaded on top of the base at inference
- **Dataset:** [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)
- **Code:** https://github.com/Akshu24Tech/text2sql-qlora

Running on the free CPU tier, so the base model is loaded in full precision (4-bit
needs a GPU) and generation takes a few seconds per query.
