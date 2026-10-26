import os

import json
from pathlib import Path

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

DATA_DIR = DATA / "06_split"
BLACKLIST = {"Festival RTP: A Melhor Canção de Sempre", "Toquinho"}

def extract_conversations_train(dataset_type: str):
    file_path = DATA_DIR / f"saudade_{dataset_type}_pt_train.json"
    if not file_path.exists():
        file_path = DATA_DIR / f"saudade_{dataset_type}_pt_train.json"
    
    if not file_path.exists():
        print(f"[ERRO] Ficheiro nao encontrado: {file_path}")
        return

    print(f"A converter {dataset_type} ({file_path.name}) para formato conversa...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conversations_records = []

    for category, entities in data.items():
        if not isinstance(entities, dict): continue
        for entity, content in entities.items():
            if not isinstance(content, dict) or entity in BLACKLIST: continue
            
            # ── AWARENESS
            if dataset_type == "awareness":
                qa_list = content.get("awareness_qa", [])
                if not qa_list:
                    for _, ev in content.items():
                        if isinstance(ev, dict) and "awareness_qa" in ev:
                            qa_list.extend(ev["awareness_qa"])
                for qa in qa_list:
                    future_f = qa.get("future_fact", "")
                    past_f   = qa.get("past_fact", "")
                    
                    # Past questions (distractor is future_fact)
                    for qtype, q in qa.get("past_questions", {}).items():
                        if q:
                            answers = qa.get("past_answers") or [qa.get("past_ground_truth", "")]
                            ans = answers[0] if answers else ""
                            rec = {
                                "domain": "awareness",
                                "category": category,
                                "entity": entity,
                                "question_type": f"past_{qtype}",
                                "conversations": [
                                    {"role": "human", "content": q},
                                    {"role": "gpt", "content": ans}
                                ],
                                "all_answers": answers,
                                "target_fact": past_f,
                                "distractor_fact": future_f,
                            }
                            conversations_records.append(rec)
                            
                    # Future questions (distractor is past_fact)
                    for qtype, q in qa.get("future_questions", {}).items():
                        if q:
                            answers = qa.get("future_answers") or [qa.get("future_ground_truth", "")]
                            ans = answers[0] if answers else ""
                            rec = {
                                "domain": "awareness",
                                "category": category,
                                "entity": entity,
                                "question_type": f"future_{qtype}",
                                "conversations": [
                                    {"role": "human", "content": q},
                                    {"role": "gpt", "content": ans}
                                ],
                                "all_answers": answers,
                                "target_fact": future_f,
                                "distractor_fact": past_f,
                            }
                            conversations_records.append(rec)

            # ── COGNITION
            elif dataset_type == "cognition":
                qa_dict = content.get("cognition_qa", content.get("temporal_qa", {}))
                for ev_name, ev in qa_dict.items():
                    if not isinstance(ev, dict): continue
                    questions = ev.get("questions") or ev.get("question")
                    answers   = ev.get("answers", [])
                    ans = answers[0] if answers else ""
                    if questions:
                        if isinstance(questions, str): questions = {"generic": questions}
                        for qtype, q in questions.items():
                            if q:
                                rec = {
                                    "domain": "cognition",
                                    "category": category,
                                    "entity": entity,
                                    "event": ev_name,
                                    "question_type": qtype,
                                    "conversations": [
                                        {"role": "human", "content": q},
                                        {"role": "gpt", "content": ans}
                                    ],
                                    "all_answers": answers,
                                }
                                conversations_records.append(rec)

            # ── REASONING
            elif dataset_type == "reasoning":
                outer_rqa = content.get("reasoning_qa", content.get("ranking_qa", []))
                for pair_item in outer_rqa:
                    inner_rqa = pair_item.get("ranking_qa", {})
                    answers = pair_item.get("expected_answers", pair_item.get("answers", []))
                    ans = answers[0] if answers else ""
                    facts_pair = pair_item.get("facts_pair", [])
                    for qtype, q in inner_rqa.items():
                        if q:
                            rec = {
                                "domain": "reasoning",
                                "category": category,
                                "entity": entity,
                                "question_type": qtype,
                                "conversations": [
                                    {"role": "human", "content": q},
                                    {"role": "gpt", "content": ans}
                                ],
                                "all_answers": answers,
                                "facts_pair": facts_pair,
                            }
                            conversations_records.append(rec)

    out_file = DATA_DIR / f"saudade_{dataset_type}_train_flat.jsonl"
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in conversations_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"  [OK {dataset_type}] {len(conversations_records)} registos gravados em {out_file.name}")

if __name__ == "__main__":
    for ds in ["awareness", "cognition", "reasoning"]:
        extract_conversations_train(ds)
