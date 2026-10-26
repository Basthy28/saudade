#!/usr/bin/env python3
"""Prepare the binary temporal-ordering track for lm-eval.

The released benchmark file stores each item as a question plus the two candidate
facts. The evaluation harness needs an explicit prompt prefix and the index of the
expected answer inside ``facts_pair``. This script adds those two fields:

    prompt        text before the answer options ("... ocorreu primeiro:")
    answer_index  0 or 1, position of the expected answer in ``facts_pair``

Items whose answer cannot be located are skipped and reported.

Usage:
    python prepare_binary_track.py --lang pt
    python prepare_binary_track.py --lang en --input data/... --output data/...
"""

import argparse
import json
import os
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

DEFAULTS = {
    "pt": {
        "input": DATA / "saudade_pt_binary_temporal_ordering.jsonl",
        "output": DATA / "eval" / "saudade_pt_eval.jsonl",
        "threshold": 0.8,
        "marker": "primeiro:",
    },
    "en": {
        "input": DATA / "saudade_en_binary_temporal_ordering.jsonl",
        "output": DATA / "eval" / "saudade_en_eval.jsonl",
        "threshold": 0.7,
        "marker": "occurred first:",
    },
}


def normalize_text(text):
    return text.strip().lower()


def find_answer_index(expected_answer, facts_pair, threshold):
    """Return ``(index, True)`` when the expected answer is found, else ``(None, False)``."""
    expected_norm = normalize_text(expected_answer)

    for i, fact in enumerate(facts_pair):
        if normalize_text(fact) == expected_norm:
            return i, True

    for i, fact in enumerate(facts_pair):
        fact_norm = normalize_text(fact)
        if expected_norm in fact_norm or fact_norm in expected_norm:
            return i, True

    best_index, best_ratio = None, 0.0
    for i, fact in enumerate(facts_pair):
        ratio = SequenceMatcher(None, expected_norm, normalize_text(fact)).ratio()
        if ratio > best_ratio:
            best_index, best_ratio = i, ratio

    if best_ratio > threshold:
        return best_index, True
    return None, False


def extract_prompt_prefix(question, marker):
    """Return the question stem that precedes the two answer options."""
    if marker in question:
        return question.split(marker)[0].strip() + " " + marker
    if ": " in question:
        return question.split(": ")[0].strip() + ":"
    return question


def process_dataset(input_path, output_path, threshold, marker):
    processed, answer_errors, prompt_errors = [], [], []

    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)

            if "prompt" not in item or "answer_index" not in item:
                prompt = extract_prompt_prefix(item["question"], marker)
                if not prompt:
                    prompt_errors.append((line_num, item["entity"]))
                    continue
                answer_index, ok = find_answer_index(
                    item["expected_answers"][0], item["facts_pair"], threshold
                )
                if not ok:
                    answer_errors.append((line_num, item["entity"]))
                    continue
                item["prompt"] = prompt
                item["answer_index"] = answer_index

            processed.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processed {len(processed)} items -> {output_path}")
    if prompt_errors:
        print(f"Skipped {len(prompt_errors)} items without an extractable prompt")
    if answer_errors:
        print(f"Skipped {len(answer_errors)} items without a matching answer")
        for line_num, entity in answer_errors[:5]:
            print(f"  line {line_num}: {entity}")
        if len(answer_errors) > 5:
            print(f"  ... and {len(answer_errors) - 5} more")

    counts = {}
    for item in processed:
        counts[item["answer_index"]] = counts.get(item["answer_index"], 0) + 1
    for index in sorted(counts):
        share = 100 * counts[index] / len(processed) if processed else 0
        print(f"  answer_index {index}: {counts[index]} ({share:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(DEFAULTS), default="pt")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    defaults = DEFAULTS[args.lang]
    process_dataset(
        args.input or defaults["input"],
        args.output or defaults["output"],
        args.threshold if args.threshold is not None else defaults["threshold"],
        defaults["marker"],
    )


if __name__ == "__main__":
    main()
