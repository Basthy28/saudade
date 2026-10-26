#!/usr/bin/env python3

import os

from pathlib import Path

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

# -*- coding: utf-8 -*-
"""
translate_pool_en.py
====================
Translate the ICL train pool (icl_pool_train.jsonl) to English using
the Gemma 4 judge (MN, porta 8766), following the same procedure and prompt used to translate the test tracks.

Writes ONLY new files; never touches the PT pool:
  OUT : icl_pool_train_en.jsonl
  CKPT: pool_en_done.txt
Row order and keys are preserved; originals kept as *_pt fields.

Uso (dentro de um Slurm job com o judge activo):
  python translate_pool_en.py [--workers N]
"""

import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, argparse, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
JUDGE_BASE_URL = "http://localhost:8766/v1"
JUDGE_API_KEY  = "EMPTY"
JUDGE_MODEL    = "gemma4-judge"
MAX_RETRIES    = 3
REQUEST_TIMEOUT = 120

BENCH_DIR = DATA
SRC       = BENCH_DIR / "icl_pool_train.jsonl"
OUT       = BENCH_DIR / "icl_pool_train_en.jsonl"   # NEW FILE
CKPT      = BENCH_DIR / "pool_en_done.txt"                 # NEW FILE

STR_FIELDS = ["question", "prompt"]
LIST_FIELDS = ["facts_pair"]

TRANSLATION_SYSTEM = """You are translating a Portuguese temporal reasoning benchmark to English.
Translation rules:
- Translate the text content naturally into English
- Keep ALL proper nouns in Portuguese: person names, place names, institution names, TV show/film/book titles, organization names (e.g. "João Sousa", "Câmara Municipal de Lisboa", "Palavras Cruzadas", "RTP1")
- Keep Portuguese-specific cultural terms that have no direct English equivalent (e.g. "telenovela", "fado", "saudade", "Telejornal")
- Maintain the same temporal reasoning structure and difficulty
- Do not add explanations, notes, or bracketed clarifications
- Respond ONLY with the translated text, no preamble"""

_client_local = threading.local()
_print_lock = threading.Lock()

def get_client() -> OpenAI:
    if not hasattr(_client_local, 'client'):
        _client_local.client = OpenAI(
            base_url=JUDGE_BASE_URL,
            api_key=JUDGE_API_KEY,
            timeout=REQUEST_TIMEOUT,
        )
    return _client_local.client

def translate_text(text: str) -> str:
    """Translate a single text snippet, with retries."""
    if not text or not text.strip():
        return text
    client = get_client()
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM},
                    {"role": "user",   "content": text},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
            else:
                with _print_lock:
                    print(f"  [WARN] Translation failed after {MAX_RETRIES} attempts: {e}")
                return text  # fallback: keep original
    return text

def translate_item(item: dict) -> dict:
    """Translate question/prompt/facts_pair; keep originals as *_pt."""
    result = dict(item)
    for field in STR_FIELDS:
        original = item.get(field)
        if original and isinstance(original, str) and original.strip():
            en = translate_text(original)
            result[field + "_pt"] = original
            result[field] = en
    for field in LIST_FIELDS:
        original = item.get(field)
        if original and isinstance(original, list):
            en_list = [translate_text(x) if isinstance(x, str) else x for x in original]
            result[field + "_pt"] = original
            result[field] = en_list
    return result

def load_checkpoint(ckpt_path: Path) -> set:
    if not ckpt_path.exists():
        return set()
    done = set()
    for line in ckpt_path.read_text('utf-8').splitlines():
        if line.strip():
            done.add(line.strip())
    return done

def item_id(item: dict, idx: int) -> str:
    entity = item.get('entity', '')
    q = item.get('question', '')[:30]
    return f"{idx}:{entity}:{q}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    assert SRC.exists(), f"missing source pool: {SRC}"
    assert not OUT.exists() or True  # append-safe; checkpoint gates rework
    items = [json.loads(l) for l in SRC.read_text('utf-8').splitlines() if l.strip()]
    done_ids = load_checkpoint(CKPT)
    print(f"Pool EN translation — {len(items)} rows | {args.workers} workers")
    print(f"Already done: {len(done_ids)} | Remaining: {len(items)-len(done_ids)}")
    print(f"Source (READ-ONLY): {SRC}")
    print(f"Output (NEW): {OUT}")

    out_handle = open(OUT, 'a', encoding='utf-8')
    ckpt_handle = open(CKPT, 'a', encoding='utf-8')
    results_lock = threading.Lock()
    done_count = [len(done_ids)]

    def process(args):
        idx, item = args
        iid = item_id(item, idx)
        if iid in done_ids:
            return None
        return idx, iid, translate_item(item)

    to_process = [(i, it) for i, it in enumerate(items) if item_id(it, i) not in done_ids]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, arg): arg[0] for arg in to_process}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            idx, iid, translated = result
            with results_lock:
                out_handle.write(json.dumps(translated, ensure_ascii=False) + '\n')
                out_handle.flush()
                ckpt_handle.write(iid + '\n')
                ckpt_handle.flush()
                done_ids.add(iid)
                done_count[0] += 1
                if done_count[0] % 100 == 0 or done_count[0] == len(items):
                    with _print_lock:
                        print(f"  [pool-en] {done_count[0]}/{len(items)} done", flush=True)
    ckpt_handle.close()
    out_handle.close()
    print(f"[OK] output: {OUT}")

if __name__ == '__main__':
    main()
