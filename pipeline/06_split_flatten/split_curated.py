import os

import json
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

DATA_DIR = DATA / "06_split"
CURATION_DIR = DATA / "05_curation" / "output"
TEST_RATIO = 0.20
RANDOM_SEED = 42

def load_and_filter_audit(dataset_name: str) -> list:
    files = list(CURATION_DIR.glob(f"{dataset_name}_audit_node_*.jsonl"))
    valid_items = []
    total_loaded = 0
    total_removed = 0

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                total_loaded += 1
                verdict = item.get("verdict", "KEEP")
                
                if verdict == "REMOVE":
                    total_removed += 1
                    continue
                
                if verdict == "MODIFY" and item.get("suggested_fix"):
                    item["question"] = item["suggested_fix"]
                    
                valid_items.append(item)

    print(f"[{dataset_name}] Loaded {total_loaded} QA items. Removed {total_removed}. Kept {len(valid_items)}.")
    return valid_items

def group_and_split_cognition(items: list, rng: random.Random):
    # Group by category -> entity -> event -> question_type
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for it in items:
        cat = it["category"]
        ent = it["entity"]
        ev = it["event"]
        qt = it["question_type"]
        qtext = it["question"]
        ans = it["expected_answers"]
        
        if "questions" not in grouped[cat][ent][ev]:
            grouped[cat][ent][ev]["questions"] = {}
        grouped[cat][ent][ev]["questions"][qt] = qtext
        grouped[cat][ent][ev]["answers"] = ans

    train_data, test_data = {}, {}
    total_q = 0
    
    for cat, entities in grouped.items():
        train_data[cat] = {}
        test_data[cat] = {}
        for ent, events in entities.items():
            ev_keys = list(events.keys())
            rng.shuffle(ev_keys)
            
            n_test = max(1, round(len(ev_keys) * TEST_RATIO))
            test_evs = ev_keys[:n_test]
            train_evs = ev_keys[n_test:]
            total_q += len(ev_keys)
            
            if train_evs: train_data[cat][ent] = {k: events[k] for k in train_evs}
            if test_evs:  test_data[cat][ent]  = {k: events[k] for k in test_evs}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "saudade_cognition_pt_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(DATA_DIR / "saudade_cognition_pt_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"Cognition split complete ({total_q} unique events split).")

def group_and_split_awareness(items: list, rng: random.Random):
    # For awareness we must group past_ and future_ parts back into pairs
    pairs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for it in items:
        cat = it["category"]
        ent = it["entity"]
        qt = it["question_type"]  # past_generic, future_rephrased_1, etc
        pair_id = f"{it['fact_target']['date']}_{it['fact_context']['date']}" # Unique ID for the pair
        
        if "past_fact" not in pairs[cat][ent][pair_id]:
            if qt.startswith("past_"):
                pairs[cat][ent][pair_id]["past_fact"] = it["fact_target"]["fact"]
                pairs[cat][ent][pair_id]["past_date"] = it["fact_target"]["date"]
                pairs[cat][ent][pair_id]["future_fact"] = it["fact_context"]["fact"]
                pairs[cat][ent][pair_id]["future_date"] = it["fact_context"]["date"]
                pairs[cat][ent][pair_id]["past_answers"] = it["expected_answers"]
            else:
                pairs[cat][ent][pair_id]["past_fact"] = it["fact_context"]["fact"]
                pairs[cat][ent][pair_id]["past_date"] = it["fact_context"]["date"]
                pairs[cat][ent][pair_id]["future_fact"] = it["fact_target"]["fact"]
                pairs[cat][ent][pair_id]["future_date"] = it["fact_target"]["date"]
                pairs[cat][ent][pair_id]["future_answers"] = it["expected_answers"]
        
        q_dict_name = "past_questions" if qt.startswith("past_") else "future_questions"
        q_key = qt.replace("past_", "").replace("future_", "")
        
        if q_dict_name not in pairs[cat][ent][pair_id]:
            pairs[cat][ent][pair_id][q_dict_name] = {}
            
        pairs[cat][ent][pair_id][q_dict_name][q_key] = it["question"]

    train_data, test_data = {}, {}
    for cat, entities in pairs.items():
        train_data[cat] = {}
        test_data[cat] = {}
        for ent, pair_dict in entities.items():
            pair_list = list(pair_dict.values())
            rng.shuffle(pair_list)
            
            n_test = max(1, round(len(pair_list) * TEST_RATIO))
            test_qa = pair_list[:n_test]
            train_qa = pair_list[n_test:]
            
            if train_qa: train_data[cat][ent] = {"awareness_qa": train_qa}
            if test_qa:  test_data[cat][ent]  = {"awareness_qa": test_qa}

    with open(DATA_DIR / "saudade_awareness_pt_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(DATA_DIR / "saudade_awareness_pt_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print("Awareness split complete.")

def group_and_split_reasoning(items: list, rng: random.Random):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for it in items:
        cat = it["category"]
        ent = it["entity"]
        ev = it["event"]
        qt = it["question_type"]
        
        if "ranking_qa" not in grouped[cat][ent][ev]:
            grouped[cat][ent][ev]["ranking_qa"] = {}
        
        grouped[cat][ent][ev]["ranking_qa"][qt] = it["question"]
        grouped[cat][ent][ev]["answers"] = it["expected_answers"]
        grouped[cat][ent][ev]["facts_pair"] = it["facts_pair"]

    train_data, test_data = {}, {}
    for cat, entities in grouped.items():
        train_data[cat] = {}
        test_data[cat] = {}
        for ent, events in entities.items():
            ev_list = [{"ranking_qa": v["ranking_qa"], "answers": v["answers"], "facts_pair": v.get("facts_pair",[])} for v in events.values()]
            rng.shuffle(ev_list)
            
            n_test = max(1, round(len(ev_list) * TEST_RATIO))
            test_evs = ev_list[:n_test]
            train_evs = ev_list[n_test:]
            
            if train_evs: train_data[cat][ent] = {"ranking_qa": train_evs}
            if test_evs:  test_data[cat][ent]  = {"ranking_qa": test_evs}

    with open(DATA_DIR / "saudade_reasoning_pt_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(DATA_DIR / "saudade_reasoning_pt_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print("Reasoning split complete.")

if __name__ == "__main__":
    print(f"Starting split logic with Seed {RANDOM_SEED} and Test Ratio {TEST_RATIO*100}%...")
    rng = random.Random(RANDOM_SEED)
    
    cog_items = load_and_filter_audit("cognition")
    if cog_items: group_and_split_cognition(cog_items, rng)
        
    awa_items = load_and_filter_audit("awareness")
    if awa_items: group_and_split_awareness(awa_items, rng)
        
    rea_items = load_and_filter_audit("reasoning")
    if rea_items: group_and_split_reasoning(rea_items, rng)
    
    print("All splits generated in:", DATA_DIR)
