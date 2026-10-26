import os
from pathlib import Path

import json, random

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

N = 1000
SRC = str(DATA / "val.jsonl")
OUT = str(DATA / "val_heldout.jsonl")
lines = [l for l in open(SRC, encoding="utf-8") if l.strip()]
print("linhas val =", len(lines))
items = []
for i, tl in enumerate(lines):
    d = json.loads(tl)
    msgs = d.get("messages", [])
    if len(msgs) < 2:
        continue
    q = next((x["content"] for x in reversed(msgs) if x.get("role") == "user"), None)
    r = next((x["content"] for x in reversed(msgs) if x.get("role") in ("assistant", "gpt")), None)
    if not q or not r:
        continue
    ctx = []
    for m in msgs[: max(0, len(msgs) - 2)]:
        role = "assistant" if m.get("role") == "gpt" else m.get("role")
        ctx.append({"role": role, "content": m.get("content", "")})
    items.append({"id": "heldout_%05d" % i, "entity": "", "category": "awcog",
                  "persona": "", "question": q, "gold_answer": r, "context": ctx})
print("validos =", len(items))
random.seed(42)
sel = random.sample(items, min(N, len(items)))
random.shuffle(sel)
with open(OUT, "w", encoding="utf-8") as f:
    for x in sel:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("amostra=%d -> %s" % (len(sel), OUT))
print("HELDOUT-OK")
