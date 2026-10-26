# tasks.py
# OE scoring logic for saudade open-ended questions.
# Evaluates model responses on 3 criteria via Gemma 4 judge.
# Criteria: accuracy, completeness, temporal - each scored 1-5, reported separately.
# Uses ThreadPoolExecutor to fully saturate the vLLM servers (continuous batching).

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import OE_PROMPT_TEMPLATE
from models import call_model, call_judge

logger = logging.getLogger(__name__)

CONCURRENCY = 64  # simultaneous requests — vLLM continuous batching handles this fine


def _normalize(raw_score: float) -> float:
    """Convert 1-5 judge score to 0-100 scale."""
    return round((raw_score - 1) / 4 * 100, 2)


def _evaluate_item(i: int, item: dict, domain: str, model_cfg: dict) -> dict:
    """Evaluate a single item: model call -> judge call. Thread-safe."""
    criteria = ("accuracy", "completeness", "temporal")

    question  = item.get("question", "")
    reference = item.get("gold_answer", "")
    prompt    = OE_PROMPT_TEMPLATE.format(question=question)

    # Step 1 - model answer
    response = call_model(model_cfg, prompt)

    if not response:
        scores = {c: 0.0 for c in criteria}
        judge_failed = True
        logger.warning(f"[{domain}] [{item.get('id', i)}] Empty model response -- skipping judge")
    else:
        # Step 2 - judge scores
        scores = call_judge(question, reference, response)
        judge_failed = all(scores[c] == 0.0 for c in criteria)

    return {
        "id":           item.get("id", i),
        "entity":       item.get("entity", ""),
        "category":     item.get("category", domain),
        "question":     question,
        "reference":    reference,
        "model_answer": response,
        "scores_raw":   {c: scores[c] for c in criteria},
        "scores_norm":  {c: _normalize(scores[c]) for c in criteria},
        "failed":       judge_failed,
    }


def evaluate_oe_domain(domain: str, data_file: Path,
                       model_cfg: dict, items: list) -> dict:
    """
    Evaluate one open-ended track concurrently.
    """
    criteria = ("accuracy", "completeness", "temporal")
    n = len(items)

    totals     = {c: 0.0 for c in criteria}
    failed     = 0
    per_item   = [None] * n
    lock       = threading.Lock()
    completed  = [0]

    logger.info(f"[{domain}] Submitting {n} items with concurrency={CONCURRENCY}")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(_evaluate_item, i, item, domain, model_cfg): i
                   for i, item in enumerate(items)}

        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"[{domain}] Item {i} raised exception: {e}")
                result = {
                    "id": items[i].get("id", i), "entity": items[i].get("entity", ""),
                    "category": domain, "question": items[i].get("question", ""),
                    "reference": items[i].get("gold_answer", ""), "model_answer": None,
                    "scores_raw": {c: 0.0 for c in criteria},
                    "scores_norm": {c: 0.0 for c in criteria},
                    "failed": True,
                }

            with lock:
                per_item[i] = result
                for c in criteria:
                    totals[c] += result["scores_raw"][c]
                if result["failed"]:
                    failed += 1
                completed[0] += 1
                done = completed[0]
                # Progress log every 50 items
                if done % 50 == 0 or done == n:
                    logger.info(
                        f"[{domain}] {done}/{n} | "
                        f"acc: {totals['accuracy']/done:.2f} | "
                        f"comp: {totals['completeness']/done:.2f} | "
                        f"temp: {totals['temporal']/done:.2f} | "
                        f"failed: {failed}"
                    )

    results = {"n": n, "failed": failed, "items": per_item}

    for c in criteria:
        raw  = round(totals[c] / n, 4) if n > 0 else 0.0
        norm = _normalize(raw)
        results[f"{c}_raw"]  = raw
        results[f"{c}_norm"] = norm

    logger.info(
        f"[{domain}] OE DONE | "
        f"acc: {results['accuracy_raw']:.2f}/5 ({results['accuracy_norm']:.1f}/100) | "
        f"comp: {results['completeness_raw']:.2f}/5 ({results['completeness_norm']:.1f}/100) | "
        f"temp: {results['temporal_raw']:.2f}/5 ({results['temporal_norm']:.1f}/100) | "
        f"n: {n} | failed: {failed}"
    )

    return results
