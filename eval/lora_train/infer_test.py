"""Smoke test for a trained LoRA adapter.

Loads the adapter on top of AMALIA-9B-SFT, generates answers for five
validation prompts (entities unseen during training) and writes a JSON
with prompt / reference / generation for inspection.
"""
import os
from pathlib import Path

import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

AMALIA = "amalia-llm/AMALIA-9B-0626-SFT"
ADAPTER = str(ROOT / "outputs" / "r16")
VAL = str(DATA / "val.jsonl")
OUT = str(ROOT / "outputs" / "r16-infer-test.json")

tok = AutoTokenizer.from_pretrained(ADAPTER)  # traz o chat template do treino
base = AutoModelForCausalLM.from_pretrained(AMALIA, dtype=torch.bfloat16,
                                            device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

prompts = []
with open(VAL, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        d = json.loads(line)
        user_msgs = [m for m in d["messages"] if m["role"] == "user"]
        asst_msgs = [m for m in d["messages"] if m["role"] == "assistant"]
        prompts.append((user_msgs[0]["content"], asst_msgs[0]["content"]))

results = []
for q, ref in prompts:
    inp = tok.apply_chat_template([{"role": "user", "content": q}],
                                  add_generation_prompt=True,
                                  return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=256, do_sample=False,
                             eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
    results.append({"prompt": q, "referencia": ref, "gerado_r16": gen})
    print("Q:", q[:120])
    print("REF:", ref[:200])
    print("R16:", gen[:400])
    print("-" * 60)

json.dump(results, open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("guardado em", OUT)
