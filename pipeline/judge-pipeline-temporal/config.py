# Open-ended evaluation configuration
#
# Evaluates models on the Fact Retrieval (cognition) and Relation with
# Distractors (awareness) open-ended tracks. The judge is Gemma 4 31B served
# on port 8766; the model under evaluation is served per job on port 8765.
#
# To add a model: copy any entry in MODELS and set its Hugging Face id.
# To submit: sbatch run_eval_mn5.sh <model_key>

import os

from pathlib import Path

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))
BASE_DIR = DATA / "07_openended"
RESULTS_DIR = Path(os.environ.get("OE_RESULTS_DIR", BASE_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Open-ended tracks ─────────────────────────────────────────────────────────
# Each track is a JSONL file with "question", "gold_answer", "entity", "category".
OE_DOMAIN_FILES = {
    "cognition": BASE_DIR / "saudade_cognition_openended_test_clean.jsonl",
    "awareness": BASE_DIR / "saudade_awareness_openended_test_clean.jsonl",
}

if os.environ.get("OE_LANG", "pt") == "en":
    OE_DOMAIN_FILES = {
        "cognition": BASE_DIR / "saudade_cognition_openended_test_clean_en.jsonl",
        "awareness": BASE_DIR / "saudade_awareness_openended_test_clean_en.jsonl",
    }

# ── Models under evaluation ───────────────────────────────────────────────────
# All are served locally via vLLM as "eval-model" on port 8765. The "path" is
# the Hugging Face id (or a local directory) passed to the server.
MODELS = {
    "amalia_sft": {
        "path": "amalia-llm/AMALIA-9B-0626-SFT",
    },
    "amalia_dpo": {
        "path": "amalia-llm/AMALIA-9B-0626-DPO",
    },
    "qwen25_7b": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
    },
    "qwen3_8b": {
        "path": "Qwen/Qwen3-8B",
        # Qwen3 emits a  thinking block by default; the 512-token cap then cuts
        # the answer off entirely. Disable thinking for this model only.
        "no_thinking": True,
    },
    "llama31_8b": {
        "path": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "eurollm_9b": {
        "path": "utter-project/EuroLLM-9B-Instruct",
    },
    "salamandra_7b": {
        "path": "BSC-LT/salamandra-7b-instruct",
    },
    "gervasio_8b": {
        "path": "PORTULAN/gervasio-8b-portuguese-ptpt-decoder",
    },
    "olmo2_7b": {
        "path": "allenai/OLMo-2-1124-7B-Instruct",
    },
    "gemma2_9b": {
        "path": "google/gemma-2-9b-it",
    },
    "gemma3_12b": {
        "path": "google/gemma-3-12b-it",
    },
    "ministral_8b": {
        "path": "mistralai/Ministral-8B-Instruct-2410",
    },
    "mistral_7b": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "apertus_8b": {
        "path": "swiss-ai/Apertus-8B-Instruct-2509",
    },
    # LoRA adapters fine-tuned on SAUDADE Clean (thesis §5.6). Only the r32
    # merge is released; set SAUDADE_LORA_R16 to a local adapter to reproduce
    # the r16 arm.
    "amalia_lora_r16": {
        "path": os.environ.get("SAUDADE_LORA_R16", "amalia-llm/saudade-9b-sft"),
    },
    "amalia_lora_r32": {
        "path": os.environ.get("SAUDADE_LORA_R32", "amalia-llm/saudade-9b-sft"),
    },
}

for entry in MODELS.values():
    entry.setdefault("model", "eval-model")
    entry.setdefault("base_url", "http://localhost:8765/v1")
    entry.setdefault("api_key", "EMPTY")

# ── Judge (Gemma 4 31B FP8) ───────────────────────────────────────────────────
JUDGE = {
    "model": "gemma4-judge",
    "base_url": "http://localhost:8766/v1",
    "api_key": "EMPTY",
}

JUDGE_FALLBACK = {
    "model": "gemma4-judge",
    "base_url": "http://localhost:8766/v1",
    "api_key": "EMPTY",
}

# ── Generation parameters ─────────────────────────────────────────────────────
OE_MAX_TOKENS = 512
TEMPERATURE = 0.0  # deterministic: same answer on every run
REQUEST_TIMEOUT = 60  # seconds
MAX_RETRIES = 3

# ── Open-ended prompt ────────────────────────────────────────────────────────
# Questions are self-contained (the distractor is embedded for awareness).
OE_PROMPT_TEMPLATE = """Responde à seguinte pergunta em português europeu, de forma clara e concisa (2 a 4 frases).

Pergunta: {question}"""
