import json
import os
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / "data" / "03_purified"

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8088/v1")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
MAX_CONCURRENT_LLM = 50

client = AsyncOpenAI(api_key="EMPTY", base_url=LOCAL_LLM_URL)

async def generate_questions_for_entity(title: str, events: List[Dict]) -> Dict:
    valid_events = [e for e in events if e.get("data_inicio") is not None]
    if not valid_events:
        return {}

    clean_title = title.strip().replace('"', '').replace("'", "")
    
    prompt = f"""Gera perguntas de avaliação em Português de Portugal baseadas nos seguintes factos sobre a entidade {clean_title}.

Instruções OBRIGATÓRIAS (LÊ COM MÁXIMA ATENÇÃO):
1. **DIVERSIDADE ESTRUTURAL EXTREMA**: As frases TÊM OBRIGATORIAMENTE de ter estruturas diferentes. É TOTALMENTE PROIBIDO que comecem todas por "Em que ano", "Em que data" ou "Qual a data".
   - Exemplo VÁLIDO: "O Sporting contratou qual treinador na época 2019/20?"
   - Exemplo VÁLIDO: "Onde se realizou a final do torneio vencida por X?"
   - Exemplo VÁLIDO: "Segundo os registos, o que aconteceu a Y em 1945?"
   
2. **ESCOLHA DO ALVO DA PERGUNTA**: Não perguntes sempre a data! Podes e deves focar-te noutros elementos chave presentes no facto, como "Quem", "O quê", "Onde" ou "Como". A pergunta tem de ser natural e sumamente precisa para aquele evento específico.
   
3. **ZERO LEAKAGE (CRÍTICO)**: A tua resposta exata esperada (o `ground_truth`) NÃO PODE, SOB NENHUMA CIRCUNSTÂNCIA, aparecer no meio do texto da pergunta! Se a resposta for "2016", a palavra "2016" não pode estar na `generic` ou nas `rephrases`.

4. **FORMATO DE RESPOSTA**:
   - `ground_truth`: Uma string exata e curta que responde perfeitamente à pergunta. (Ex: "D. Afonso Henriques", ou "1139-07-25").
   - `answers`: Uma lista com a `ground_truth` e outras sintaxes super óbvias/sinónimos da mesma resposta.
   - A pergunta TEM de referir o nome EXATO da entidade ({clean_title}) sem usar aspas ou pelicas.
   
5. Gera a pergunta principal (`generic`) e 3 rephrases (`rephrased_1`, etc.) que procurem EXACTAMENTE a mesma `ground_truth` mas que mudem radicalmente a construção da frase ou os sinónimos usados do evento.

Factos Disponíveis:
"""
    for idx, f in enumerate(valid_events):
        prompt += f"[{idx}] Facto: {f['facto']} | Data de Início: {f['data_inicio']}\n"

    prompt += """
Responde APENAS com JSON válido neste formato exato (usa como chaves os IDs numéricos que precedem os factos abaixo):
{
  "0": {
    "generic": "Pergunta principal direta e com contexto?",
    "rephrased_1": "Variante 1?",
    "rephrased_2": "Variante 2?",
    "rephrased_3": "Variante 3?",
    "ground_truth": "Resposta Curta e Exata",
    "answers": ["Resposta Curta e Exata", "Sinónimo Válido se Existir"]
  },
  "1": { ... }
}
"""
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a precise dataset builder generating varied QA pairs. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                timeout=600.0
            )
            out_text = response.choices[0].message.content or ""
            out_text = out_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(out_text)
            
            entity_dict = {}
            for idx_str, q_dict in data.items():
                if not idx_str.isdigit():
                    continue
                idx = int(idx_str)
                if idx < 0 or idx >= len(valid_events):
                    continue
                
                fact = valid_events[idx]
                fact_key = fact["facto"]
                
                if fact_key in entity_dict:
                    fact_key = f"{fact_key}_{idx}"

                entity_dict[fact_key] = {
                    "questions": {
                        "generic": q_dict.get("generic", ""),
                        "rephrased_1": q_dict.get("rephrased_1", ""),
                        "rephrased_2": q_dict.get("rephrased_2", ""),
                        "rephrased_3": q_dict.get("rephrased_3", "")
                    },
                    "answers": q_dict.get("answers", [fact["data_inicio"]]),
                    "ground_truth": q_dict.get("ground_truth", fact["data_inicio"]),
                    "source_url": f"https://pt.wikipedia.org/wiki/{title.replace(' ', '_')}"
                }
            
            return entity_dict
        except Exception as e:
            if attempt == 2:
                print(f"Error generating for {title}: {e}")
            pass
    return {}


async def process_category_file(cat: str, file_path: Path, sem: asyncio.Semaphore, out_file: str):
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        title = data.get("title", "")
        events = data.get("events", [])
        
        if not title or not events:
            return

        async with sem:
            entity_qa = await generate_questions_for_entity(title, events)
        
        if entity_qa:
            result = {"category": cat, "entity": title, "cognition_qa": entity_qa}
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
    out_file = str(OUTPUT_DIR / "saudade_cognition_pt_extra.jsonl")

    # Multi-node Sharding
    node_id = int(os.getenv("SLURM_NODEID", "0"))
    num_nodes = int(os.getenv("SLURM_NNODES", "1"))
    
    if num_nodes > 1:
        out_path = Path(out_file)
        out_file = str(out_path.parent / f"{out_path.stem}_node_{node_id}{out_path.suffix}")

    # Checkpoints
    processed_entities = set()
    
    # Load previously generated entities from original run to skip them
    for p in OUTPUT_DIR.glob("saudade_cognition_pt_node_*.jsonl"):
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
        print("All valid tasks are already processed! Checkpoint complete.")
        return
        
    print(f"Generating cognition QA for {len(tasks)} new entities...")
    await tqdm_asyncio.gather(*tasks, desc="Generating Cognition QA")
    print(f"Processamento concluído. Dataset salvo em: {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
