"""
Text-to-SQL demo — Qwen2.5-1.5B-Instruct + QLoRA adapter.

Runs on a free Hugging Face Space (CPU). bitsandbytes 4-bit needs a GPU, so on
CPU we load the base model in full precision and apply the LoRA adapter on top —
same weights as training, just not quantized. First generation is slow (the model
loads on boot); after that expect ~10-40s per query on the free CPU tier.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "adapter"  # the LoRA adapter folder, uploaded alongside this file
SYSTEM = (
    "You are a text-to-SQL assistant. Given a table schema and a question, "
    "reply with only the SQL query, nothing else."
)

print("loading tokenizer + base model (this takes a minute on first boot)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()
print("model ready.")


@torch.no_grad()
def generate_sql(schema: str, question: str) -> str:
    if not schema.strip() or not question.strip():
        return "Please fill in both the schema and the question."
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


EXAMPLES = [
    [
        "CREATE TABLE employees (name VARCHAR, department VARCHAR, salary INTEGER)",
        "What is the average salary in the engineering department?",
    ],
    [
        "CREATE TABLE head (age INTEGER)",
        "How many heads of the departments are older than 56?",
    ],
    [
        "CREATE TABLE table_name_33 (mascot VARCHAR, school VARCHAR)",
        "Name the mascot for shelbyville",
    ],
]

with gr.Blocks(title="Text-to-SQL — Qwen2.5-1.5B QLoRA") as demo:
    gr.Markdown(
        "# Text-to-SQL — Qwen2.5-1.5B QLoRA\n"
        "Paste a `CREATE TABLE` schema and ask a question in plain English; "
        "the model returns a SQL query. Fine-tuned with QLoRA on "
        "[b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context). "
        "Running on a free CPU Space, so generation takes a few seconds."
    )
    schema = gr.Textbox(
        label="Table schema (CREATE TABLE ...)",
        lines=3,
        placeholder="CREATE TABLE employees (name VARCHAR, department VARCHAR, salary INTEGER)",
    )
    question = gr.Textbox(
        label="Question",
        lines=2,
        placeholder="What is the average salary in the engineering department?",
    )
    btn = gr.Button("Generate SQL", variant="primary")
    output = gr.Code(label="Generated SQL", language="sql")
    gr.Examples(examples=EXAMPLES, inputs=[schema, question])
    btn.click(generate_sql, inputs=[schema, question], outputs=output)

if __name__ == "__main__":
    demo.queue().launch()
