import os

import json
from pathlib import Path

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

SPLIT_DIR = DATA / "06_split"
SFT_DIR   = DATA / "08_sft" / "output"
OUT_DIR   = DATA / "08_sft" / "output"
OE_DIR    = DATA / "07_openended"

SKIP_CATS = {"Freguesia_Portugal"}

# ── 1. Collect entities from the categories to exclude ──
skip_entities = set()
for fname in SPLIT_DIR.glob("*.json"):
    data = json.loads(fname.read_text(encoding="utf-8"))
    for cat, entities in data.items():
        if cat in SKIP_CATS and isinstance(entities, dict):
            for entity in entities:
                skip_entities.add(entity)

print(f"Entities to exclude ({', '.join(SKIP_CATS)}): {len(skip_entities)}")

# ── 2. Filter the 6 SFT files (Cognition, Awareness, Reasoning; clean & tiser) ──
files = [
    ("cognition_dialog_sft_all_clean.jsonl", "cognition_dialog_sft_all_clean.jsonl"),
    ("awareness_dialog_sft_all_clean.jsonl", "awareness_dialog_sft_all_clean.jsonl"),
    ("reasoning_dialog_sft_all_clean.jsonl", "reasoning_dialog_sft_all_clean.jsonl"),
    ("cognition_dialog_sft_all_tiser.jsonl", "cognition_dialog_sft_all_tiser.jsonl"),
    ("awareness_dialog_sft_all_tiser.jsonl", "awareness_dialog_sft_all_tiser.jsonl"),
    ("reasoning_dialog_sft_all_tiser.jsonl", "reasoning_dialog_sft_all_tiser.jsonl"),
]

print("\n--- Filtering SFT Dialogues ---")
for src_name, dst_name in files:
    src = SFT_DIR / src_name
    dst = OUT_DIR / dst_name
    if not src.exists():
        print(f"[SKIP] {src_name} — not found")
        continue

    kept = skipped = 0
    with open(dst, "w", encoding="utf-8") as out_f:
        with open(src, encoding="utf-8") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                entity = r.get("entity", "")
                if entity in skip_entities:
                    skipped += 1
                else:
                    out_f.write(line + "\n")
                    kept += 1

    total = kept + skipped
    pct = 100 * skipped / total if total > 0 else 0
    print(f"{src_name} -> {dst_name}")
    print(f"  kept={kept}  skipped={skipped}  ({pct:.1f}% removed)")

# ── 3. Filter open-ended test files ──
OE_DIR    = DATA / "07_openended"

oe_files = [
    ("saudade_cognition_openended_test_clean.jsonl", "saudade_cognition_openended_test_clean.jsonl"),
    ("saudade_awareness_openended_test_clean.jsonl", "saudade_awareness_openended_test_clean.jsonl"),
]

print("\n--- Filtering Open-Ended Test Sets ---")
for src_name, dst_name in oe_files:
    src = OE_DIR / src_name
    dst = OE_DIR / dst_name
    if not src.exists():
        print(f"[SKIP] {src_name} — not found")
        continue
    kept = skipped = 0
    with open(dst, "w", encoding="utf-8") as out_f:
        with open(src, encoding="utf-8") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                cat = r.get("category", "")
                if cat in SKIP_CATS:
                    skipped += 1
                else:
                    out_f.write(line + "\n")
                    kept += 1
    total = kept + skipped
    pct = 100 * skipped / total if total > 0 else 0
    print(f"{src_name} -> {dst_name}")
    print(f"  kept={kept}  skipped={skipped}  ({pct:.1f}% removed)")
