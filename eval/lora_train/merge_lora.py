"""Funde o adapter LoRA treinado com a base AMALIA -> modelo standalone.

Uso: python merge_lora.py --adapter outputs/r16 --out outputs/r16-merged
"""
import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

AMALIA = "amalia-llm/AMALIA-9B-0626-SFT"

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

print("1/4 base...")
base = AutoModelForCausalLM.from_pretrained(AMALIA, dtype=torch.bfloat16)
print("2/4 adapter:", a.adapter)
model = PeftModel.from_pretrained(base, a.adapter)
print("3/4 merge...")
model = model.merge_and_unload()
print("4/4 save ->", a.out)
model.save_pretrained(a.out, safe_serialization=True)

# tokenizer do adapter (ja vem com o chat_template certo do treino)
tok = AutoTokenizer.from_pretrained(a.adapter)
tok.save_pretrained(a.out)

# sanity: config final deve ser LlamaForCausalLM sem vestigios de peft
with open(Path(a.out) / "config.json", encoding="utf-8") as f:
    cfg = json.load(f)
assert "peft" not in json.dumps(cfg).lower(), "config ainda tem vestigios peft!"
assert cfg["architectures"] == ["LlamaForCausalLM"]
print("OK — modelo standalone em", a.out)
print("dtype:", cfg.get("dtype") or cfg.get("torch_dtype"))
