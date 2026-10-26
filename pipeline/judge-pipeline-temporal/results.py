# results.py
# Aggregates per-domain OE results and saves to disk.
# Reports accuracy, completeness, temporal separately — no composite score.

import json
import logging
from datetime import datetime
from pathlib import Path

from config import RESULTS_DIR

logger = logging.getLogger(__name__)

CRITERIA = ("accuracy", "completeness", "temporal")


def aggregate(model_name: str, model_cfg: dict, oe_results: dict) -> dict:
    """
    Aggregate per-domain OE results into a single results dict.
    Each criterion (accuracy, completeness, temporal) is reported independently.
    No mean or composite score is computed.
    """
    domains = {}

    for domain, result in oe_results.items():
        domains[domain] = {
            "n":      result["n"],
            "failed": result["failed"],
            "items":  result.get("items", []),
        }
        for c in CRITERIA:
            domains[domain][f"{c}_raw"]  = result.get(f"{c}_raw", 0.0)
            domains[domain][f"{c}_norm"] = result.get(f"{c}_norm", 0.0)

    # Overall averages across all domains — only common criteria
    total_n = sum(r["n"] for r in oe_results.values())
    overall = {}
    for c in CRITERIA:
        if total_n > 0:
            weighted = sum(
                r.get(f"{c}_norm", 0.0) * r["n"] for r in oe_results.values()
            )
            overall[c] = round(weighted / total_n, 2)
        else:
            overall[c] = 0.0

    return {
        "model":     model_name,
        "model_id":  model_cfg["model"],
        "model_path": model_cfg.get("path", ""),
        "timestamp": datetime.utcnow().isoformat(),
        "domains":   domains,
        "overall":   overall,
    }


def print_summary(results: dict) -> None:
    """Print a human-readable results table to stdout."""
    model   = results["model"]
    overall = results["overall"]
    domains = results["domains"]

    print()
    print("=" * 65)
    print(f"  SAUDADE open-ended results \u2014 {model}")
    print("=" * 65)
    print(f"  {'TRACK':<20} {'ACC':>8} {'COMP':>8} {'TEMP':>8}  {'N':>5}  {'FAIL':>5}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}  {'-'*5}  {'-'*5}")

    for domain, d in sorted(domains.items()):
        print(
            f"  {domain:<20} "
            f"{d['accuracy_norm']:>7.1f}% "
            f"{d['completeness_norm']:>7.1f}% "
            f"{d['temporal_norm']:>7.1f}% "
            f"{d['n']:>5}  "
            f"{d['failed']:>5}"
        )

    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}  {'-'*5}  {'-'*5}")
    print(
        f"  {'OVERALL':<20} "
        f"{overall['accuracy']:>7.1f}% "
        f"{overall['completeness']:>7.1f}% "
        f"{overall['temporal']:>7.1f}%"
    )
    print("=" * 65)
    print()


def save(results: dict, include_items: bool = True) -> Path:
    """
    Save results to a JSON file in RESULTS_DIR.
    Filename: {model}_{timestamp}.json
    Returns the path written.
    """
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"{results['model']}_{ts}.json"
    path = RESULTS_DIR / name

    output = dict(results)
    if not include_items:
        # Strip per-item detail to keep file small
        for domain_data in output.get("domains", {}).values():
            domain_data.pop("items", None)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved: {path}")
    return path