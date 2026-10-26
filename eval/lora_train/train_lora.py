#!/usr/bin/env python3
"""LoRA SFT: AMALIA-9B-0626-SFT + dialogos SAUDADE (clean, multi-turn).

Assistant-only loss: o loss cobre SO os turnos do assistente (gpt);
os turnos do utilizador entram como contexto. Exige chat template com
marcadores {% generation %} — o template original do AMALIA nao os tem,
por isso usamos uma copia corrigida (identica em texto, so acrescenta os
marcadores de mascara).
"""
import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))
TRAIN_DIR = DATA / "08_sft" / "ready_to_train"

AMALIA = "amalia-llm/AMALIA-9B-0626-SFT"

# Template do AMALIA (em fallback para system por omissao do model card)
# com marcadores {% generation %} delimitando o conteudo do assistente.
FIXED_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{% set messages = [{'role': 'system', 'content': 'O teu nome é AMALIA, e és um modelo avançado de linguagem útil. Responde sempre na língua do utilizador, a menos que sejas instruído em contrário, e lembra-te que a tua língua principal é o português europeu.'}] + messages %}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|im_end|>\n' }}"
    "{% else %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

p = argparse.ArgumentParser()
p.add_argument("--rank", type=int, required=True)
p.add_argument("--train-file", default=str(TRAIN_DIR / "train.jsonl"))
p.add_argument("--val-file", default=str(TRAIN_DIR / "val.jsonl"))
p.add_argument("--output-dir", default=None)
p.add_argument("--epochs", type=float, default=2.0)
p.add_argument("--max-steps", type=int, default=-1)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--max-grad-norm", type=float, default=1.0)
p.add_argument("--max-length", type=int, default=2048)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--accum", type=int, default=1)
p.add_argument("--eval-batch", type=int, default=8)
p.add_argument("--eval-steps", type=int, default=1000)
p.add_argument("--save-steps", type=int, default=9000)
p.add_argument("--logging-steps", type=int, default=20)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

out = a.output_dir or str(ROOT / "outputs" / f"r{a.rank}")
tag = os.path.basename(out.rstrip("/")) or ("clean-r%d" % a.rank)
run_name = "amalia-saudade-" + tag

tok = AutoTokenizer.from_pretrained(AMALIA)
tok.chat_template = FIXED_TEMPLATE

# --- verificacao pre-voo: a mascara assistant-only funciona ---
_probe = [{"role": "user", "content": "Olá, tudo bem?"},
          {"role": "assistant", "content": "Tudo! Em que ajudo?"}]
try:
    enc = tok.apply_chat_template(_probe, tokenize=True,
                                  return_assistant_tokens_mask=True, return_dict=True)
    m = enc["assistant_masks"]
    n_tok, n_asst = len(m), sum(m)
    assert n_asst > 0 and n_asst < n_tok, f"mascara estranha: {n_asst}/{n_tok}"
    print(f"[pre-flight] template OK: {n_tok} tokens, {n_asst} de assistente "
          f"(o loss so ve a resposta)")
except Exception as e:
    raise RuntimeError(f"FALHA na mascara assistant-only: {e}")

ds = load_dataset("json", data_files={"train": a.train_file, "val": a.val_file})
print(f"train: {len(ds['train'])} | val: {len(ds['val'])}")

lora = LoraConfig(
    r=a.rank,
    lora_alpha=a.rank,          # guia Unsloth: alpha = r (standard)
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

cfg = SFTConfig(
    output_dir=out,
    num_train_epochs=a.epochs,
    max_steps=a.max_steps,
    learning_rate=a.lr,
    max_grad_norm=a.max_grad_norm,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    per_device_train_batch_size=a.batch,
    per_device_eval_batch_size=a.eval_batch,
    gradient_accumulation_steps=a.accum,
    gradient_checkpointing=True,
    max_length=a.max_length,
    packing=False,
    assistant_only_loss=True,
    bf16=True,
    logging_steps=a.logging_steps,
    eval_strategy="steps",
    eval_steps=a.eval_steps,
    save_strategy="steps",
    save_steps=a.save_steps,
    save_total_limit=4,
    seed=a.seed,
    report_to="wandb",
    run_name=run_name,
    model_init_kwargs={"dtype": torch.bfloat16},
    dataset_num_proc=8,
)

trainer = SFTTrainer(
    model=AMALIA,
    args=cfg,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    processing_class=tok,
    peft_config=lora,
)
trainer.model.print_trainable_parameters()
trainer.train()
metrics = trainer.evaluate()
print("EVAL FINAL:", metrics)
trainer.save_model(out)
tok.save_pretrained(out)
print("DONE", out)
