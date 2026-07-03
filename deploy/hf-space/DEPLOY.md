# Deploying the demo to a Hugging Face Space

This gives you a permanent public **live link** (e.g.
`https://huggingface.co/spaces/Akshu24Tech/text2sql-qlora`) to put in your submission.

## What you're uploading

Four things go into the Space:

```
app.py            # the Gradio demo (already written)
requirements.txt  # dependencies (already written)
README.md         # Space config + description (already written)
adapter/          # YOUR trained LoRA adapter — you add this (see step 1)
```

## Step 1 — get the adapter folder

The trained adapter is in your Google Drive at **`MyDrive/qwen25-sql-lora`**
(saved by the training notebook). Download that folder. You need at least these
two files out of it:

- `adapter_config.json`
- `adapter_model.safetensors`

Put them in a folder named exactly **`adapter`** next to `app.py`. So the final
layout on your machine is:

```
deploy/hf-space/
  app.py
  requirements.txt
  README.md
  adapter/
    adapter_config.json
    adapter_model.safetensors
```

> The adapter isn't in the GitHub repo (weights are gitignored), which is why you
> pull it from Drive. `adapter_config.json` is required — it tells PEFT the base
> model and LoRA settings. If your Drive folder is missing it, tell me and I'll
> reconstruct it from the training notebook's settings.

## Step 2 — create the Space

1. Go to https://huggingface.co/new-space
2. **Owner:** your account · **Space name:** `text2sql-qlora`
3. **SDK:** Gradio · **Hardware:** CPU basic (free) · **Visibility:** Public
4. Click **Create Space**.

## Step 3 — upload the files

**Easiest (web UI):** on the new Space page → **Files** tab → **Add file → Upload
files** → drag in `app.py`, `requirements.txt`, `README.md`, and the whole
`adapter/` folder. Commit.

**Or via git** (from `deploy/hf-space/`, with the `adapter/` folder in place):

```bash
git clone https://huggingface.co/spaces/<your-username>/text2sql-qlora
# copy app.py, requirements.txt, README.md, adapter/ into the cloned folder
cd text2sql-qlora
git add .
git commit -m "text-to-SQL QLoRA demo"
git push
```

(The `adapter_model.safetensors` is ~36MB — under the 10GB Space limit, no Git LFS
setup needed, but HF may prompt you to track it with LFS, which is fine.)

## Step 4 — wait for the build, grab the link

The Space will show **Building** then **Running** (first build takes a few minutes
— it installs torch and downloads the ~3GB base model on boot). Once it says
**Running**, the URL at the top of the page is your live link:

```
https://huggingface.co/spaces/<your-username>/text2sql-qlora
```

That's what you submit.

## Note for your recording / submission

The demo loads the base Qwen2.5-1.5B and applies your LoRA adapter on top, so it
faithfully shows *your* fine-tuned model generating SQL from a schema + question.
Keep in mind the eval finding (the adapter matched base output on the 200 eval
examples) — the demo will behave like the base model until the fine-tune is
re-run and verified. It still fully satisfies a "live link" requirement.
