#!/usr/bin/env python3
"""
run_evaluation.py
saudade — Open-Ended Evaluation Pipeline (Cognition + Awareness)

Evaluates one model on the saudade OE benchmark.
Scores 3 criteria per item via Gemma 4 judge: accuracy, completeness, temporal.
Each criterion is reported independently — no composite score.

Usage:
    # Evaluate a specific model (key must match config.py MODELS)
    python run_evaluation.py --model salamandra_7b

    # Limit items per domain (for testing — use 3 to verify pipeline works)
    python run_evaluation.py --model salamandra_7b --limit 3

    # List all available model keys
    python run_evaluation.py --list-models
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from config import MODELS, OE_DOMAIN_FILES
from tasks import evaluate_oe_domain
from results import aggregate, save, print_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _load_domain_items(data_file: Path, limit: int | None = None) -> list:
    """Load all items from a single domain JSONL file."""
    items = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if limit:
        items = items[:limit]
    return items


def evaluate_model(model_name: str, model_cfg: dict, limit: int | None = None,
                   domain_files: dict | None = None) -> dict:
    """Run the full OE evaluation for one model across specified domains."""
    if domain_files is None:
        domain_files = OE_DOMAIN_FILES

    logger.info("=" * 65)
    logger.info(f"Model : {model_name}")
    logger.info(f"Path  : {model_cfg.get('path', 'n/a')}")
    logger.info("=" * 65)

    oe_results = {}

    for domain, data_file in domain_files.items():
        if not data_file.exists():
            logger.warning(f"[{domain}] Data file not found: {data_file} — skipping")
            continue

        items = _load_domain_items(data_file, limit=limit)
        if not items:
            logger.warning(f"[{domain}] No items found in {data_file.name} — skipping")
            continue

        logger.info(f"\n[{domain}] {len(items)} items")
        result = evaluate_oe_domain(domain, data_file, model_cfg, items)
        oe_results[domain] = result

    if not oe_results:
        logger.error("No domains evaluated — check data file paths in config.py")
        sys.exit(1)

    results = aggregate(model_name, model_cfg, oe_results)
    print_summary(results)
    path = save(results, include_items=True)
    logger.info(f"Saved: {path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="saudade OE Evaluation — MareNostrum 5"
    )
    parser.add_argument(
        "--model",
        help="Model key from config.py MODELS (e.g. salamandra_7b)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit items per domain for testing (e.g. --limit 3)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print all available model keys and exit"
    )
    parser.add_argument(
        "--domain",
        help="Only evaluate this track (cognition or awareness). Default: both."
    )
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models in config.py:\n")
        for key, cfg in MODELS.items():
            print(f"  {key:<20}  {cfg.get('path', '')}")
        print()
        sys.exit(0)

    if not args.model:
        parser.error("--model is required. Use --list-models to see options.")

    if args.model not in MODELS:
        logger.error(
            f"Model '{args.model}' not found in config.py. "
            f"Available: {list(MODELS.keys())}"
        )
        sys.exit(1)

    domain_files = OE_DOMAIN_FILES
    if args.domain:
        if args.domain not in OE_DOMAIN_FILES:
            logger.error(f"Domain '{args.domain}' not in config.py. Available: {list(OE_DOMAIN_FILES.keys())}")
            sys.exit(1)
        domain_files = {args.domain: OE_DOMAIN_FILES[args.domain]}
        logger.info(f"Running only domain: {args.domain}")

    evaluate_model(args.model, MODELS[args.model], limit=args.limit, domain_files=domain_files)


if __name__ == "__main__":
    main()
