import os

import json
from pathlib import Path

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

SFT_DIR = DATA / "08_sft" / "output"

def convert_to_single_turn(filename: str):
    src_file = SFT_DIR / filename
    if not src_file.exists():
        print(f"[ERRO] Ficheiro nao encontrado: {src_file}")
        return
        
    out_name = filename.replace(".jsonl", "_single_turn.jsonl")
    out_file = SFT_DIR / out_name
    
    print(f"A desdobrar {filename} para single-turn...")
    count = 0
    with open(src_file, 'r', encoding='utf-8') as f_in, \
         open(out_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            data = json.loads(line)
            entity = data.get("entity", "")
            dataset = data.get("dataset", "")
            persona = data.get("persona", "")
            convs = data.get("conversations", [])
            
            # Pair human (i) and gpt (i+1)
            for i in range(0, len(convs) - 1, 2):
                if convs[i].get("role") == "human" and convs[i+1].get("role") == "gpt":
                    single_record = {
                        "entity": entity,
                        "dataset": dataset,
                        "persona": persona,
                        "n_tiser": 1,
                        "n_casual": 0,
                        "conversations": [
                            convs[i],
                            convs[i+1]
                        ]
                    }
                    f_out.write(json.dumps(single_record, ensure_ascii=False) + "\n")
                    count += 1
                    
    print(f"  [OK] {count} turnos individuais gravados em {out_file.name}")

if __name__ == "__main__":
    files_to_convert = [
        "awareness_dialog_sft_all_clean.jsonl",
        "cognition_dialog_sft_all_clean.jsonl",
        "reasoning_dialog_sft_all_clean.jsonl",
        "awareness_dialog_sft_all_tiser.jsonl",
        "cognition_dialog_sft_all_tiser.jsonl",
        "reasoning_dialog_sft_all_tiser.jsonl",
    ]
    for f in files_to_convert:
        convert_to_single_turn(f)
