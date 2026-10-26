import json
import os
import re
import asyncio
import random
import argparse
from pathlib import Path
from typing import Dict, List
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import itertools

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / "data" / "03_purified"

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8088/v1")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
MAX_CONCURRENT_LLM = 50

client = AsyncOpenAI(api_key="EMPTY", base_url=LOCAL_LLM_URL)

def clean_facto(text: str) -> str:
    return re.sub(r'\s*\(Data:[^)]*\)', '', text).strip()

async def generate_awareness_questions(title: str, fact_pairs: List[List[Dict]]) -> List[Dict]:
    """Gera perguntas de Awareness em batches de 5 pares com retry em caso de JSON error."""
    all_results = []
    BATCH_SIZE = 5
    batches = [fact_pairs[i:i+BATCH_SIZE] for i in range(0, len(fact_pairs), BATCH_SIZE)]

    for batch in batches:
        results = await _generate_batch(title, batch)
        all_results.extend(results)

    return all_results

async def _generate_batch(title: str, fact_pairs: List[List[Dict]], max_retries: int = 3) -> List[Dict]:
    results = []
    clean_title = title.strip().replace('"', '').replace("'", "")

    prompt = f"Aqui tens pares de factos reais sobre a entidade \"{clean_title}\".\n\n"
    for idx, pair in enumerate(fact_pairs):
        f_past, f_future = pair
        prompt += f"Par {idx+1}:\n"
        prompt += f" - [Facto Passado] (Data: {f_past['data_inicio']}): {f_past['facto_clean']}\n"
        prompt += f" - [Facto Futuro – Contexto Enganador] (Data: {f_future['data_inicio']}): {f_future['facto_clean']}\n\n"

    prompt += f"""Instruções:
1. Para CADA Par de factos, tens de construir DUAS perguntas de Conhecimento Factual em Português de Portugal: uma sobre o Facto Passado e outra sobre o Facto Futuro.
2. Cada pergunta deve ser direta e pedir uma RESPOSTA CURTA (ex: "Quem foi o campeão da Primeira Liga na temporada 2015-16?").
3. ⚠️ OBRIGATÓRIO: TODAS as perguntas devem SEMPRE mencionar EXPLICITAMENTE o nome da entidade "{clean_title}" (ou forma reconhecível). NUNCA uses pronomes ou referências vagas como "o canal", "a série", "o clube", "o partido", "o jogo", "o programa", etc.
4. As perguntas NUNCA devem incluir a resposta (`ground_truth`) (zero data leakage).
5. As perguntas devem ser completamente isoladas e limpas (sem menções cruzadas ou contexto enganador do outro facto).
6. A `past_ground_truth` DEVE SER UMA RESPOSTA CURTA E EXATA extraída do Facto Passado (ex: 'SL Benfica').
7. A `future_ground_truth` DEVE SER UMA RESPOSTA CURTA E EXATA extraída do Facto Futuro (ex: 'FC Porto').
8. O campo `past_fact` e `future_fact` deve conter o texto limpo sem data.
9. Gera 3 variações simples (rephrased_1 a rephrased_3) para a pergunta do Facto Passado e para a do Facto Futuro.

Responde APENAS com um Objeto JSON com este formato exato:
{{
    "perguntas": [
        {{
            "past_fact": "Texto limpo do Facto Passado do Par 1.",
            "past_date": "{fact_pairs[0][0]['data_inicio'] if fact_pairs else ''}",
            "future_fact": "Texto limpo do Facto Futuro do Par 1.",
            "future_date": "{fact_pairs[0][1]['data_inicio'] if fact_pairs else ''}",
            
            "past_ground_truth": "RESPOSTA CURTA do Facto Passado.",
            "past_answers": ["RESPOSTA CURTA", "sinónimo se existir"],
            "past_questions": {{
                "generic": "[Pergunta direta SEMPRE mencionando '{clean_title}' e APENAS sobre o Facto Passado]?",
                "rephrased_1": "[Variação 1 — sempre menciona '{clean_title}']?",
                "rephrased_2": "[Variação 2 — sempre menciona '{clean_title}']?",
                "rephrased_3": "[Variação 3 — sempre menciona '{clean_title}']?"
            }},
            
            "future_ground_truth": "RESPOSTA CURTA do Facto Futuro.",
            "future_answers": ["RESPOSTA CURTA", "sinónimo se existir"],
            "future_questions": {{
                "generic": "[Pergunta direta SEMPRE mencionando '{clean_title}' e APENAS sobre o Facto Futuro]?",
                "rephrased_1": "[Variação 1 — sempre menciona '{clean_title}']?",
                "rephrased_2": "[Variação 2 — sempre menciona '{clean_title}']?",
                "rephrased_3": "[Variação 3 — sempre menciona '{clean_title}']?"
            }}
        }}
    ]
}}"""

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert dataset builder generating Awareness (Context Misalignment) QA pairs for European Portuguese. Output pure JSON without markdown codeblocks or thinking tags."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                timeout=600.0
            )
            out_text = response.choices[0].message.content or ""
            out_text = out_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            data_obj = json.loads(out_text)
            data_list = data_obj.get("perguntas", []) if isinstance(data_obj, dict) else data_obj

            for data in data_list:
                if not isinstance(data, dict):
                    continue
                past_fact   = clean_facto(data.get("past_fact", ""))
                future_fact = clean_facto(data.get("future_fact", ""))
                gt          = data.get("past_ground_truth", "").strip()
                past_questions = data.get("past_questions", {})
                future_questions = data.get("future_questions", {})
                if not all([past_fact, future_fact, gt, past_questions.get("generic")]):
                    continue
                results.append({
                    "past_fact": past_fact,
                    "past_date": data.get("past_date", ""),
                    "future_fact": future_fact,
                    "future_date": data.get("future_date", ""),
                    "past_ground_truth": gt,
                    "past_answers": data.get("past_answers", [gt]),
                    "past_questions": past_questions,
                    "future_ground_truth": data.get("future_ground_truth", ""),
                    "future_answers": data.get("future_answers", []),
                    "future_questions": future_questions,
                    "source_url": f"https://pt.wikipedia.org/wiki/{clean_title.replace(' ', '_')}"
                })
            break

        except Exception:
            pass
    return results

async def process_category_file(cat: str, file_path: Path, sem: asyncio.Semaphore, out_file: str):
    try:
        data   = json.loads(file_path.read_text(encoding="utf-8"))
        title  = data.get("title", "")
        events = data.get("events", [])

        if not title or len(events) < 2:
            return

        valid_events = []
        for e in events:
            di = e.get("data_inicio")
            if di and len(str(di)) >= 4:
                try:
                    int(str(di)[:4])
                except ValueError:
                    continue
                valid_events.append({
                    "facto_clean": clean_facto(str(e.get("facto", ""))),
                    "data_inicio": str(di)
                })

        if len(valid_events) < 2:
            return

        async with sem:
            valid_events.sort(key=lambda x: x["data_inicio"])
            pairs = []
            for f1, f2 in itertools.combinations(valid_events, 2):
                y1 = f1["data_inicio"][:4]
                y2 = f2["data_inicio"][:4]
                if y1 != y2 and f1["data_inicio"] < f2["data_inicio"]:
                    pairs.append([f1, f2])

            if not pairs:
                return

            random.shuffle(pairs)
            max_q = max(2, min(len(valid_events), 15))
            chosen_pairs = pairs[:max_q]

            awareness_qa = await generate_awareness_questions(title, chosen_pairs)

        if awareness_qa:
            result = {"category": cat, "entity": title, "awareness_qa": awareness_qa}
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Erro a processar {file_path}: {e}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cats", nargs='*', help="Specific categories to process")
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        print(f"Erro: Dir {INPUT_DIR} não encontrado.")
        return

    OUTPUT_DIR = BASE_DIR.parent / "data" / "04_generated"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = str(OUTPUT_DIR / "saudade_awareness_pt_extra.jsonl")

    # Multi-node Sharding
    node_id = int(os.getenv("SLURM_NODEID", "0"))
    num_nodes = int(os.getenv("SLURM_NNODES", "1"))
    
    if num_nodes > 1:
        out_path = Path(out_file)
        out_file = str(out_path.parent / f"{out_path.stem}_node_{node_id}{out_path.suffix}")

    # Checkpoints
    processed_entities = set()
    
    # Load previously generated entities from original run to skip them
    for p in OUTPUT_DIR.glob("saudade_awareness_pt_node_*.jsonl"):
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                try: processed_entities.add(json.loads(line)["entity"])
                except: pass
                
    # Load from the new extra file in case of restart
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                try: processed_entities.add(json.loads(line)["entity"])
                except: pass
                
    print(f"Loaded {len(processed_entities)} already processed entities to skip.")

    sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)
    all_files = []
    
    if args.cats:
        for cat in args.cats:
            cat_dir = INPUT_DIR / cat
            if cat_dir.exists():
                all_files.extend([(cat_dir.name, f) for f in cat_dir.glob("*.json")])
    else:
        for cat_dir in INPUT_DIR.iterdir():
            if cat_dir.is_dir():
                all_files.extend([(cat_dir.name, f) for f in cat_dir.glob("*.json")])

    all_files.sort()
    
    if num_nodes > 1:
        all_files = [f for i, f in enumerate(all_files) if i % num_nodes == node_id]
        print(f"Node {node_id}/{num_nodes} will process {len(all_files)} files.")

    tasks = []
    for cat_name, json_path in all_files:
        # Check entity title (it's the file stem)
        file_title = json_path.stem.replace('_', ' ')
        if file_title in processed_entities or json_path.stem in processed_entities:
            continue
        tasks.append(process_category_file(cat_name, json_path, sem, out_file))

    if not tasks:
        print("All valid tasks are already processed! Checkpoint complete.")
        return

    print(f"Generating awareness QA for {len(tasks)} new entities...")
    await tqdm_asyncio.gather(*tasks, desc="Generating Awareness QA")
    print(f"Processamento concluído. Dataset salvo em: {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
