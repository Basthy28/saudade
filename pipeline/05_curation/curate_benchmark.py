import asyncio
import json
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import aiohttp
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert benchmark quality reviewer.
Your task is to evaluate the quality of questions from a Portuguese temporal QA dataset (SAUDADE).
You must respond only with a valid JSON object, no markdown, no extra text."""

def build_user_prompt(dataset: str, item: Dict) -> str:
    question = item["question"]
    answers = item["expected_answers"]
    
    base = f"""Evaluate the following question from the '{dataset}' dataset.

Question: {question}
Expected answers (any of these is acceptable): {json.dumps(answers, ensure_ascii=False)}
"""

    if dataset == "awareness":
        target = item.get("fact_target", {})
        context = item.get("fact_context", {})
        base += f"""
This is an AWARENESS question. The structure is:
- Target Fact (what the question should be asking about): "{target.get('fact','')}" (date: {target.get('date','?')})
- Context Fact (the distracting fact that happened at a different date): "{context.get('fact','')}" (date: {context.get('date','?')})
The question gives the Context Fact as context and asks about the Target Fact. Evaluate if this is well-formed and temporally coherent.
"""

    base += """
Evaluate based on these criteria:
1. CLARITY: Is the question clear, unambiguous and grammatically correct in Portuguese?
2. FACTUAL_CORRECTNESS: Is the expected answer factually correct?
3. UNIQUENESS: Does the question have a single correct answer (or a small set of equivalent answers)?
4. NO_LEAKAGE: Is the answer NOT directly given away in the question itself?
5. TEMPORAL_COHERENCE (awareness only): Is the temporal relationship between the embedded context and the question logically sound?

Respond with ONLY this JSON:
{
  "verdict": "KEEP" | "REMOVE" | "MODIFY",
  "reason": "<one sentence explaining the decision>",
  "suggested_fix": "<corrected question if verdict is MODIFY, else null>",
  "failed_criteria": ["CLARITY", "FACTUAL_CORRECTNESS", ...] 
}

If verdict is KEEP, failed_criteria must be an empty list [].
"""
    return base

async def judge_question(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    base_url: str,
    model_name: str,
    item: Dict,
    dataset: str,
) -> Dict:
    prompt = build_user_prompt(dataset, item)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }

    for attempt in range(3):
        try:
            async with sem:
                async with session.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            content = data["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            verdict_data = json.loads(content)
            return {**item, **verdict_data}

        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, KeyError, RuntimeError) as e:
            if attempt == 2:
                log.warning(f"Failed after 3 attempts: {e}")
                return {
                    **item,
                    "verdict": "ERROR",
                    "reason": str(e),
                    "suggested_fix": None,
                    "failed_criteria": [],
                }
            await asyncio.sleep(2 ** attempt)

def extract_cognition_items(lines: List[Dict]) -> List[Dict]:
    items = []
    for record in lines:
        cat = record["category"]
        title = record["entity"]
        qa_data = record.get("cognition_qa", {})
        for event_title, content in qa_data.items():
            questions = content.get("questions", {})
            answers = content.get("answers", [])
            for q_type, q_text in questions.items():
                if not q_text or not isinstance(q_text, str): continue
                items.append({
                    "dataset": "cognition",
                    "category": cat,
                    "event": event_title,
                    "entity": title,
                    "question_type": q_type,
                    "question": q_text,
                    "expected_answers": answers if isinstance(answers, list) else [str(answers)],
                })
    return items

def extract_reasoning_items(lines: List[Dict]) -> List[Dict]:
    items = []
    for record in lines:
        cat = record["category"]
        title = record["entity"]
        qa_data = record.get("reasoning_qa", [])
        for pair_item in qa_data:
            inner_rqa = pair_item.get("ranking_qa", {})
            answers = pair_item.get("answers", [])
            facts_pair = pair_item.get("facts_pair", [])
            for q_type, q_text in inner_rqa.items():
                if not q_text or not isinstance(q_text, str): continue
                items.append({
                    "dataset": "reasoning",
                    "category": cat,
                    "entity": title,
                    "facts_pair": facts_pair,
                    "question_type": q_type,
                    "question": q_text,
                    "expected_answers": answers if isinstance(answers, list) else [str(answers)],
                })
    return items

def extract_awareness_items(lines: List[Dict]) -> List[Dict]:
    items = []
    for record in lines:
        cat = record["category"]
        title = record["entity"]
        qa_list = record.get("awareness_qa", [])
        for qa_item in qa_list:
            past_questions = qa_item.get("past_questions", {})
            past_answers = qa_item.get("past_answers", [qa_item.get("past_ground_truth")])
            for q_type, q_text in past_questions.items():
                if q_text and isinstance(q_text, str):
                    items.append({
                        "dataset": "awareness",
                        "category": cat,
                        "entity": title,
                        "question_type": f"past_{q_type}",
                        "question": q_text,
                        "expected_answers": past_answers,
                        "fact_target": {"fact": qa_item.get("past_fact"), "date": qa_item.get("past_date")},
                        "fact_context": {"fact": qa_item.get("future_fact"), "date": qa_item.get("future_date")}
                    })
            future_questions = qa_item.get("future_questions", {})
            future_answers = qa_item.get("future_answers", [qa_item.get("future_ground_truth")])
            for q_type, q_text in future_questions.items():
                if q_text and isinstance(q_text, str):
                    items.append({
                        "dataset": "awareness",
                        "category": cat,
                        "entity": title,
                        "question_type": f"future_{q_type}",
                        "question": q_text,
                        "expected_answers": future_answers,
                        "fact_target": {"fact": qa_item.get("future_fact"), "date": qa_item.get("future_date")},
                        "fact_context": {"fact": qa_item.get("past_fact"), "date": qa_item.get("past_date")}
                    })
    return items

async def curate_dataset(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    base_url: str,
    model_name: str,
    items: List[Dict],
    dataset_name: str,
    output_dir: Path,
    node_id: int
) -> None:
    audit_path = output_dir / f"{dataset_name}_audit_node_{node_id}.jsonl"

    done_ids = set()
    if audit_path.exists():
        with open(audit_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_ids.add((obj.get("category",""), obj.get("entity",""), obj.get("question_type",""), obj.get("question","")))
                except:
                    pass
        log.info(f"[{dataset_name}] Node {node_id} Resuming — {len(done_ids)} already done")

    pending = [
        it for it in items
        if (it.get("category",""), it.get("entity",""), it.get("question_type",""), it.get("question","")) not in done_ids
    ]
    log.info(f"[{dataset_name}] Node {node_id} {len(pending)} pending / {len(items)} assigned")

    tasks = [
        judge_question(session, sem, base_url, model_name, item, dataset_name)
        for item in pending
    ]

    if not tasks:
        log.info(f"[{dataset_name}] Node {node_id} All valid tasks are already processed! Checkpoint complete.")
        return

    with open(audit_path, "a", buffering=1, encoding='utf-8') as f_out:
        for coro in tqdm_asyncio.as_completed(tasks, desc=dataset_name, total=len(tasks)):
            result = await coro
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    log.info(f"[{dataset_name}] Node {node_id} finished auditing.")

async def get_model_name(base_url: str, session: aiohttp.ClientSession) -> str:
    try:
        async with session.get(f"{base_url}/models") as resp:
            data = await resp.json()
        return data["data"][0]["id"]
    except:
        return "default_model"

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://localhost:8088/v1")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_concurrent", type=int, default=128)
    parser.add_argument("--datasets", nargs="+", default=["cognition", "reasoning", "awareness"])
    parser.add_argument("--cats", nargs='*', help="Specific categories to process")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_id = int(os.getenv("SLURM_NODEID", "0"))
    num_nodes = int(os.getenv("SLURM_NNODES", "1"))

    extractors = {
        "cognition": ("saudade_cognition_pt.jsonl", extract_cognition_items),
        "reasoning": ("saudade_reasoning_pt.jsonl", extract_reasoning_items),
        "awareness": ("saudade_awareness_pt.jsonl", extract_awareness_items),
        "cognition_extra": ("saudade_cognition_pt_extra.jsonl", extract_cognition_items),
        "reasoning_extra": ("saudade_reasoning_pt_extra.jsonl", extract_reasoning_items),
        "awareness_extra": ("saudade_awareness_pt_extra.jsonl", extract_awareness_items),
    }

    sem = asyncio.Semaphore(args.max_concurrent)

    connector = aiohttp.TCPConnector(limit=args.max_concurrent + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        model_name = await get_model_name(args.base_url, session)
        log.info(f"Judge model: {model_name}")

        for ds_name in args.datasets:
            filename, extractor = extractors[ds_name]
            
            # Read all node-sharded inputs for this dataset
            all_lines = []
            file_pattern = filename.replace(".jsonl", "*.jsonl")
            for path in data_dir.glob(file_pattern):
                log.info(f"Loading {path}...")
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            record = json.loads(line)
                            if args.cats and record.get("category") not in args.cats:
                                continue
                            all_lines.append(record)
                        except:
                            pass
            
            items = extractor(all_lines)
            
            if num_nodes > 1:
                items = [it for i, it in enumerate(items) if i % num_nodes == node_id]
                
            log.info(f"[{ds_name}] Extracted {len(items)} questions assigned to Node {node_id}")

            await curate_dataset(
                session, sem, args.base_url, model_name,
                items, ds_name, output_dir, node_id
            )

    log.info("All datasets curated.")

if __name__ == "__main__":
    asyncio.run(main())
