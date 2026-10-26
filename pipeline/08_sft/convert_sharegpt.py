import os

import json
from pathlib import Path

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

OUT_DIR = DATA / "08_sft" / "output"

files = [
    "awareness_dialog_sft_all_clean.jsonl",
    "cognition_dialog_sft_all_clean.jsonl",
    "reasoning_dialog_sft_all_clean.jsonl",
    "awareness_dialog_sft_all_tiser.jsonl",
    "cognition_dialog_sft_all_tiser.jsonl",
    "reasoning_dialog_sft_all_tiser.jsonl"
]

print("Converting to ShareGPT format (conversations/from/value)...")

for fname in files:
    src = OUT_DIR / fname
    dst = OUT_DIR / fname.replace(".jsonl", "_sharegpt.jsonl")
    if not src.exists():
        continue
        
    kept = 0
    with open(src, "r", encoding="utf-8") as f_in, open(dst, "w", encoding="utf-8") as f_out:
        for line in f_in:
            d = json.loads(line)
            turns = d.pop("turns", [])
            
            conversations = []
            for t in turns:
                conversations.append({"role": "human", "content": t.get("human", "")})
                conversations.append({"role": "gpt", "content": t.get("gpt", "")})
                
            d["conversations"] = conversations
            
            f_out.write(json.dumps(d, ensure_ascii=False) + "\n")
            kept += 1
            
    print(f"{fname} -> {dst.name} ({kept} dialogues converted)")
