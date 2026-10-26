import os
import json
import random
import asyncio
import argparse
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# Setup paths
ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))
DISTILLED_DIR = DATA / "02_distillation" / "temporal_cleaned"
RELEVANT_DIR = DATA / "01_extraction" / "newdata" / "temporal_relevant"

SYSTEM_PROMPT = """You are a strict, expert fact-checker evaluating Portuguese text.
Your job is to determine if a specific 'fact' is 100% supported by the provided Wikipedia text.
The fact MUST NOT contain any information, dates, or implications that are not explicitly present in or directly logically deducible from the text.
Answer exactly and only with a valid JSON in this format:
{"supported": true/false, "reason": "Short one sentence explanation."}"""

async def evaluate_fact(client, sem, model_name, wiki_text, facto, data_inicio, data_fim, meta, out_file):
    user_prompt = f"""WIKIPEDIA TEXT:
{wiki_text}

EXTRACTED FACT:
Fact: {facto}
Start Date: {data_inicio}
End Date: {data_fim}

Is this Fact and its Dates fully supported by the Wikipedia text?
Respond with JSON only."""

    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            eval_result = json.loads(content)
        except Exception as e:
            eval_result = {"supported": False, "reason": f"Error or invalid JSON: {e}"}

    meta["evaluation"] = eval_result
    
    # Append to file immediately (Checkpointing)
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        
    return eval_result.get("supported", False)

async def process_sample(sample_size, api_url, model_name, out_file, concurrent, cats):
    client = AsyncOpenAI(api_key="EMPTY", base_url=api_url, timeout=300.0)
    sem = asyncio.Semaphore(concurrent)
    
    # 0. Multi-node Sharding & Output File Routing
    node_id = int(os.getenv("SLURM_NODEID", "0"))
    num_nodes = int(os.getenv("SLURM_NNODES", "1"))
    
    if num_nodes > 1:
        # Append node id to out_file to prevent write collisions
        out_path = Path(out_file)
        out_file = str(out_path.parent / f"{out_path.stem}_node_{node_id}{out_path.suffix}")
    
    # 0.5 Load existing checkpoints
    processed_facts = set()
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_facts.add(f"{data['entity']}_{data['fact_index']}")
                except:
                    continue
        print(f"Loaded {len(processed_facts)} already processed facts from checkpoint {out_file}.")
    
    # 1. Gather all distilled files
    all_files = []
    if cats:
        for cat in cats:
            cat_dir = DISTILLED_DIR / cat
            if cat_dir.exists():
                for f in cat_dir.glob('*.json'):
                    all_files.append(f)
    else:
        for root, _, files in os.walk(DISTILLED_DIR):
            for f in files:
                if f.endswith('.json'):
                    all_files.append(Path(root) / f)
                
    if not all_files:
        print("No distilled files found!")
        return
        
    all_files.sort() # Ensure deterministic order for sharding
    print(f"Found {len(all_files)} total distilled files.")
    
    if num_nodes > 1:
        all_files = [f for i, f in enumerate(all_files) if i % num_nodes == node_id]
        print(f"Node {node_id}/{num_nodes} will process {len(all_files)} files.")
    
    # 2. Sample (only if sample_size > 0, otherwise process all)
    if sample_size > 0 and sample_size < len(all_files):
        random.seed(1337)
        sample_files = random.sample(all_files, sample_size)
    else:
        sample_files = all_files
    
    tasks = []
    
    for filepath in sample_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping corrupt distilled file {filepath}: {e}")
            continue
            
        category = filepath.parent.name
        entity = filepath.stem
        
        wiki_path = RELEVANT_DIR / category / entity / "entity_data.json"
        if not wiki_path.exists():
            continue
            
        try:
            with open(wiki_path, 'r', encoding='utf-8') as f:
                wiki_data = json.load(f)
        except Exception as e:
            print(f"Skipping corrupt wiki file {wiki_path}: {e}")
            continue
            
        # Extract full text
        full_text_blocks = []
        for sec in wiki_data.get('sections', []):
            if sec.get('title'):
                full_text_blocks.append(sec['title'])
            for content_block in sec.get('content', []):
                full_text_blocks.append(content_block.strip())
                
        wiki_text_full = "\n\n".join([b for b in full_text_blocks if b])
        
        # Hard truncate to roughly 12000 words
        words = wiki_text_full.split()
        if len(words) > 12000:
            wiki_text_full = " ".join(words[:12000])
                
        events = data.get('events', [])
        for i, ev in enumerate(events):
            if not isinstance(ev, dict): continue
            facto = ev.get('facto')
            if not facto: continue
            
            fact_key = f"{entity}_{i}"
            if fact_key in processed_facts:
                continue # Skip already processed!
            
            meta = {
                "entity": entity,
                "fact_index": i,
                "facto": facto,
                "data_inicio": ev.get('data_inicio'),
                "data_fim": ev.get('data_fim'),
                "file": str(filepath)
            }
            tasks.append(evaluate_fact(client, sem, model_name, wiki_text_full, facto, meta["data_inicio"], meta["data_fim"], meta, out_file))
            
    if not tasks:
        print("All valid tasks are already processed! Checkpoint complete.")
        return
        
    print(f"Evaluating {len(tasks)} new extracted facts from {len(sample_files)} entities with {concurrent} concurrency...")
    
    # 3. Run all tasks concurrently
    results_raw = await tqdm_asyncio.gather(*tasks)
    
    # 4. Summarize new session results
    supported = sum(1 for r in results_raw if r)
    total = len(results_raw)
    
    print("\n" + "="*50)
    print(f"NEW SESSION RESULTS: {supported}/{total} facts are fully supported ({(supported/total)*100:.1f}%)")
    print(f"HALLUCINATION RATE: {((total-supported)/total)*100:.1f}%")
    print("="*50)
    print(f"Full details appended to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=0, help="Number of entities to sample. 0 means all.")
    parser.add_argument("--cats", nargs='*', help="Specific categories to process")
    parser.add_argument("--api_url", type=str, default="http://localhost:8093/v1")
    parser.add_argument("--model", type=str, default="RedHatAI/gemma-4-31B-it-FP8-block")
    parser.add_argument("--out", type=str, default="distilled_validation_results.jsonl")
    parser.add_argument("--concurrent", type=int, default=256)
    args = parser.parse_args()
    
    asyncio.run(process_sample(args.sample_size, args.api_url, args.model, args.out, args.concurrent, args.cats))
