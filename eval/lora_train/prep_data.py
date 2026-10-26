#!/usr/bin/env python3
"""Prepara dados SAUDADE clean para SFT (TRL conversational).

Le dialogos sharegpt (human/gpt), converte para messages (user/assistant),
baralha com seed fixa, separa train/val, e reporta estatisticas de tokens
com o tokenizer do AMALIA-9B.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))
OUT_DIR = DATA / "08_sft" / "ready_to_train"

AMALIA = "amalia-llm/AMALIA-9B-0626-SFT"

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--out-dir", default=str(OUT_DIR))
p.add_argument("--val-frac", type=float, default=0.005)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(AMALIA)

rows, anomalies = [], 0
with open(a.input, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        msgs = [{"role": ("user" if m["role"] == "human" else "assistant"),
                 "content": m["content"]} for m in d["conversations"]]
        # sanidade: comeca em user, alterna, sem vazios
        ok = (msgs[0]["role"] == "user"
              and all(m["content"].strip() for m in msgs)
              and all(msgs[i]["role"] != msgs[i + 1]["role"] for i in range(len(msgs) - 1)))
        if not ok:
            anomalies += 1
            continue
        rows.append({"messages": msgs})

print(f"lidos: {len(rows)} | descartados (estrutura invalida): {anomalies}")

random.seed(a.seed)
random.shuffle(rows)
nv = max(1, int(len(rows) * a.val_frac))
val, train = rows[:nv], rows[nv:]

for name, split in [("train", train), ("val", val)]:
    with open(Path(a.out_dir) / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# smoke (50 train / 10 val) para testes rapidos em GPU
with open(Path(a.out_dir) / "smoke_train.jsonl", "w", encoding="utf-8") as f:
    for r in train[:50]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(Path(a.out_dir) / "smoke_val.jsonl", "w", encoding="utf-8") as f:
    for r in val[:10]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

sample = random.sample(rows, min(2000, len(rows)))
lens = np.array([len(tok.apply_chat_template(r["messages"], tokenize=True)) for r in sample])
print(f"train: {len(train)} | val: {len(val)}")
print(f"tokens/dialogo (n={len(lens)}): p50={np.percentile(lens, 50):.0f} "
      f"p95={np.percentile(lens, 95):.0f} p99={np.percentile(lens, 99):.0f} max={lens.max()}")
print(f"total estimado: {lens.mean() * len(rows) / 1e6:.0f}M tokens")
