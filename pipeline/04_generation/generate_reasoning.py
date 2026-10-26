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
    """Remove quaisquer anotações de data da string do facto (ex: ' (Ocorreu em: 2015)')."""
    return re.sub(r'\s*\(Ocorreu em:[^)]*\)', '', text).strip()

async def generate_rank_questions(title: str, fact_pairs: List[List[Dict]]) -> List[Dict]:
    all_results = []
    BATCH_SIZE = 5
    batches = [fact_pairs[i:i+BATCH_SIZE] for i in range(0, len(fact_pairs), BATCH_SIZE)]
    for batch in batches:
        all_results.extend(await _generate_rank_batch(title, batch))
    return all_results

async def _generate_rank_batch(title: str, fact_pairs: List[List[Dict]], max_retries: int = 3) -> List[Dict]:
    results = []
    clean_title = title.strip().replace('"', '').replace("'", "")

    prompt = f"Aqui tens pares de factos reais sobre a entidade \"{clean_title}\".\n\n"
    for idx, pair in enumerate(fact_pairs):
        f1, f2 = pair
        prompt += f"Par {idx+1}:\n"
        prompt += f" - [Facto A] (Data: {f1['data_inicio']}): {f1['facto_clean']}\n"
        prompt += f" - [Facto B] (Data: {f2['data_inicio']}): {f2['facto_clean']}\n\n"

    prompt += f"""Instruções:
1. Para CADA Par listado (Par 1, Par 2, ...) cria UMA pergunta de ordenação temporal.
2. A pergunta deve ser fluída e natural em Português de Portugal. Pergunta qual dos dois factos ocorreu PRIMEIRO.
3. ⚠️ OBRIGATÓRIO: TODAS as perguntas devem SEMPRE mencionar EXPLICITAMENTE o nome da entidade \"{clean_title}\" (ou forma reconhecível). NUNCA uses pronomes ou referências vagas como "o clube", "o partido", "a equipa", "o programa", etc. O leitor deve perceber de que entidade se fala sem contexto adicional.
4. Usa apenas os textos limpos dos factos (sem datas) no corpo da pergunta. As datas fornecidas são só para saber a ordem correta — NÃO as copies para a pergunta.
5. Não uses aspas a envolver os factos na pergunta. Não menciones "Facto A" ou "Facto B".
6. Gera também 3 variações (rephrased_1, rephrased_2, rephrased_3) que perguntem o mesmo de forma diferente e que também mencionem \"{clean_title}\".
7. No campo `occurred_first` coloca APENAS o texto limpo (sem datas) do facto que ocorreu cronologicamente primeiro com base nas datas fornecidas.
8. Os campos `facto_A` e `facto_B` devem conter os textos limpos dos Factos A e B respetivamente (sem datas).

Responde APENAS com um Objeto JSON com este formato exato:
{{
    "perguntas": [
        {{
            "facto_A": "Texto limpo do Facto A do Par 1.",
            "facto_B": "Texto limpo do Facto B do Par 1.",
            "occurred_first": "Texto limpo do facto que ocorreu PRIMEIRO (sem data).",
            "generic": "Pergunta fluída sobre qual dos dois eventos de {clean_title} ocorreu primeiro...",
            "rephrased_1": "[Variação 1 — sempre menciona {clean_title}]?",
            "rephrased_2": "[Variação 2 — sempre menciona {clean_title}]?",
            "rephrased_3": "[Variação 3 — sempre menciona {clean_title}]?"
        }}
    ]
}}"""

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert dataset builder generating chronological ordering QA pairs for European Portuguese. Output pure JSON without markdown codeblocks or thinking tags."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=8192,
                response_format={"type": "json_object"},
                timeout=600.0
            )
            out_text = response.choices[0].message.content or ""
            out_text = out_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            data_obj = json.loads(out_text)
            data_list = []
            if isinstance(data_obj, dict):
                if "perguntas" in data_obj and isinstance(data_obj["perguntas"], list):
                    data_list = data_obj["perguntas"]
                else:
                    for val in data_obj.values():
                        if isinstance(val, list):
                            data_list.extend(val)
                        elif isinstance(val, dict):
                            data_list.append(val)
            if not data_list and isinstance(data_obj, list):
                data_list = data_obj

            for data in data_list:
                if not isinstance(data, dict):
                    continue
                required = ["facto_A", "facto_B", "occurred_first", "generic", "rephrased_1", "rephrased_2", "rephrased_3"]
                if not all(k in data for k in required):
                    continue
                fact1 = clean_facto(data["facto_A"])
                fact2 = clean_facto(data["facto_B"])
                ans   = clean_facto(data["occurred_first"])
                results.append({
                    "facts_pair": [fact1, fact2],
                    "ranking_qa": {
                        "generic":     data["generic"],
                        "rephrased_1": data["rephrased_1"],
                        "rephrased_2": data["rephrased_2"],
                        "rephrased_3": data["rephrased_3"]
                    },
                    "answers":      [ans],
                    "ground_truth": ans,
                    "source_url":   f"https://pt.wikipedia.org/wiki/{title.replace(' ', '_')}"
                })
            break  # success

        except Exception as e:
            try:
                snippet = out_text[:200] + "..." + out_text[-200:] if len(out_text) > 400 else out_text
            except:
                snippet = "None"
            print(f"[REASONING_ERROR] Entity: {clean_title} | Attempt: {attempt+1}/{max_retries} | Error: {e} | Generated snippet: {snippet}")
            if attempt == max_retries - 1:
                pass
    return results

async def process_category_file(cat: str, file_path: Path, sem: asyncio.Semaphore, out_file: str):
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        title  = data.get("title", "")
        events = data.get("events", [])

        if not title or len(events) < 2:
            return

        valid_facts = []
        for e in events:
            di = e.get("data_inicio")
            if di and len(str(di)) >= 4:
                try:
                    int(str(di)[:4])
                except ValueError:
                    continue
                valid_facts.append({
                    "facto_clean": clean_facto(str(e.get("facto", ""))),
                    "data_inicio": str(di)
                })

        if len(valid_facts) < 2:
            return

        async with sem:
            pairs = []
            for f1, f2 in itertools.combinations(valid_facts, 2):
                y1 = f1["data_inicio"][:4]
                y2 = f2["data_inicio"][:4]
                if y1 != y2:
                    pairs.append([f1, f2])

            if not pairs:
                return

            random.shuffle(pairs)
            max_q = max(2, min(len(valid_facts), 15))
            chosen = pairs[:max_q]

            final_pairs = []
            for p in chosen:
                random.shuffle(p)
                final_pairs.append(p)

            ranking_qa = await generate_rank_questions(title, final_pairs)

        if ranking_qa:
            result = {"category": cat, "entity": title, "reasoning_qa": ranking_qa}
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
    out_file = str(OUTPUT_DIR / "saudade_reasoning_pt_extra.jsonl")

    # Multi-node Sharding
    node_id = int(os.getenv("SLURM_NODEID", "0"))
    num_nodes = int(os.getenv("SLURM_NNODES", "1"))
    
    if num_nodes > 1:
        out_path = Path(out_file)
        out_file = str(out_path.parent / f"{out_path.stem}_node_{node_id}{out_path.suffix}")

    # Checkpoints
    processed_entities = set()
    
    # Load previously generated entities from original run to skip them
    for p in OUTPUT_DIR.glob("saudade_reasoning_pt_node_*.jsonl"):
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
        file_title = json_path.stem.replace('_', ' ')
        if file_title in processed_entities or json_path.stem in processed_entities:
            continue
        tasks.append(process_category_file(cat_name, json_path, sem, out_file))

    if not tasks:
        print("All valid tasks are already processed! REASONING Checkpoint complete.")
        return

    print(f"Generating reasoning QA for {len(tasks)} new entities...")
    await tqdm_asyncio.gather(*tasks, desc="Generating Reasoning QA")
    print(f"Processamento concluído. Dataset salvo em: {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
