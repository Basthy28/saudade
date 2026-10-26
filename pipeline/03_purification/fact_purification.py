
import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

def purify_facts(cats):
    report_dir = DATA / "03_purification" / "output"
    source_dir = DATA / "02_distillation" / "temporal_cleaned"
    dest_dir = DATA / "03_purified"

    dest_dir.mkdir(parents=True, exist_ok=True)

    print("Loading hallucination reports...")
    bad_facts_map = defaultdict(set)
    total_checked = 0
    total_bad = 0

    # Load from all node sharded reports
    report_files = list(report_dir.glob("distilled_hallucination_report_ALL*.jsonl"))
    if not report_files:
        print("No hallucination reports found! Run check_distilled_hallucinations.py first.")
        return

    for report_file in report_files:
        with open(report_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    total_checked += 1
                    if not record.get("evaluation", {}).get("supported", False):
                        entity = record["entity"]
                        fact_text = record["facto"]
                        bad_facts_map[entity].add(fact_text)
                        total_bad += 1
                except:
                    pass

    print(f"Loaded {total_checked} total fact evaluations from {len(report_files)} files.")
    print(f"Found {total_bad} facts flagged as HALLUCINATED that will be purged.")

    print("Purifying dataset...")
    
    files = []
    if cats:
        for cat in cats:
            cat_dir = source_dir / cat
            if cat_dir.exists():
                files.extend(list(cat_dir.glob("*.json")))
    else:
        files = list(source_dir.rglob("*.json"))
        
    purged_count = 0
    kept_count = 0

    for filepath in tqdm(files):
        entity = filepath.stem
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "events" not in data:
            continue

        CleanFacts = []
        for fact in data["events"]:
            fact_text = fact.get("facto", fact.get("event", ""))
            if fact_text in bad_facts_map[entity]:
                purged_count += 1
            else:
                CleanFacts.append(fact)
                kept_count += 1

        data["events"] = CleanFacts

        # Only save entities that still have at least 1 valid fact after purification
        if len(CleanFacts) > 0:
            rel_path = filepath.relative_to(source_dir)
            dest_file = dest_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n--- PURIFICATION COMPLETE ---")
    print(f"Original Facts Checked: {total_checked}")
    print(f"Facts Purged: {purged_count}")
    print(f"Clean Facts Saved: {kept_count}")
    print(f"Purified dataset is ready at: {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cats", nargs='*', help="Specific categories to purify")
    args = parser.parse_args()
    purify_facts(args.cats)
