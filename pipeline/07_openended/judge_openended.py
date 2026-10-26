"""
judge_openended.py
==================
Judge for the open-ended tracks, with an inline filter: only records with
pass=True reach the final file.

Output: overwrites the source files and writes *_clean variants.

Usage:
  JUDGE_MODEL_ID="gemma..." VLLM_ENDPOINTS="ip:port" python3 judge_openended.py
"""

import os, json, asyncio, logging, re
from pathlib import Path
from openai import AsyncOpenAI

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

logging.basicConfig(level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR       = DATA / "07_openended"
JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID",
                    "RedHatAI/gemma-4-31B-it-FP8-block")
VLLM_ENDPOINTS = os.environ.get("VLLM_ENDPOINTS", "")
PORT           = int(os.environ.get("PORT", "8088"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "500"))
N_TEST         = int(os.environ.get("N_TEST", "0"))

FILES = [
    DATA_DIR / "saudade_cognition_openended_test.jsonl",
    DATA_DIR / "saudade_awareness_openended_test.jsonl",
]

# ── Client pool ────────────────────────────────────────────────────────────────
CLIENTS = []
_idx = 0
def build_clients():
    global CLIENTS
    eps = [e.strip() for e in VLLM_ENDPOINTS.split(",") if e.strip()]
    CLIENTS = [AsyncOpenAI(base_url=f"http://{ep}/v1", api_key="x") for ep in eps] \
              or [AsyncOpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="x")]

def next_client():
    global _idx
    c = CLIENTS[_idx % len(CLIENTS)]; _idx += 1; return c

# ── Judge prompt ───────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
És um avaliador de qualidade de dados para um benchmark académico de LLMs em português europeu.

Avalia o seguinte registo em 4 dimensões:

ENTIDADE: {entity}
PERGUNTA OPEN-ENDED: {question}
FACTO ESPERADO (anchor): {anchor_fact}
RESPOSTA DE REFERÊNCIA (gold): {gold_answer}
EXCERTO WIKIPEDIA (completo):
{wiki_excerpt}

Responde APENAS com um JSON válido, sem mais texto:

{{
  "anchor_present": true/false,
  "anchor_present_reason": "explicação breve",
  "gold_grounded": true/false,
  "gold_grounded_reason": "explicação breve",
  "question_openended": true/false,
  "question_openended_reason": "explicação breve",
  "question_single": true/false,
  "question_single_reason": "explicação breve",
  "question_temporal": true/false,
  "question_temporal_reason": "explicação breve"
}}

Definições:
- anchor_present: o anchor_fact ("{anchor_fact}") está coberto na gold_answer (directamente ou por equivalência semântica)?
- gold_grounded: a gold_answer é inteiramente suportada pelo excerto Wikipedia (sem alucinações, sem hedges "o excerto não fornece...")?
- question_openended: a pergunta exige elaboração e contexto (não é apenas uma reformulação da pergunta original fechada)?
- question_single: a pergunta é UMA questão coesa (não duas sub-questões independentes)?
- question_temporal: a pergunta exige estritamente raciocínio cronológico ou conhecimento de ordenação/datas exactas para ser respondida? Se a resposta esperada for apenas um nome (ex: nome de personagem, ator, local, canal de TV, autor) ou um facto estático disfarçado com expressões vazias como "numa determinada fase da vida", DEVE SER FALSE. A mera presença de linguagem temporal na pergunta não a torna temporal se a essência da resposta procurada não depender do tempo.

pass = anchor_present AND gold_grounded AND question_openended AND question_single AND question_temporal
"""

async def judge_record(rec: dict, sem: asyncio.Semaphore) -> dict:
    """Avalia um registo e adiciona o campo quality."""
    async with sem:
        wiki_full = rec.get("wiki_excerpt", "") or ""
        prompt = JUDGE_PROMPT.format(
            entity=rec.get("entity", ""),
            question=rec.get("question", ""),
            anchor_fact=rec.get("anchor_fact", ""),
            gold_answer=rec.get("gold_answer", ""),
            wiki_excerpt=wiki_full,
        )
        for attempt in range(3):
            try:
                r = await next_client().chat.completions.create(
                    model=JUDGE_MODEL_ID,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.0,
                )
                raw = r.choices[0].message.content.strip()
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if not m:
                    raise ValueError("No JSON in response")
                verdict = json.loads(m.group())

                quality = {
                    "pass": bool(
                        verdict.get("anchor_present") and
                        verdict.get("gold_grounded") and
                        verdict.get("question_openended") and
                        verdict.get("question_single", True) and
                        verdict.get("question_temporal", True)
                    ),
                    "anchor_present":     verdict.get("anchor_present", False),
                    "gold_grounded":      verdict.get("gold_grounded", False),
                    "question_openended": verdict.get("question_openended", False),
                    "question_single":    verdict.get("question_single", True),
                    "question_temporal":  verdict.get("question_temporal", False),
                    "reasons": {
                        "anchor":    verdict.get("anchor_present_reason", ""),
                        "grounded":  verdict.get("gold_grounded_reason", ""),
                        "openended": verdict.get("question_openended_reason", ""),
                        "single":    verdict.get("question_single_reason", ""),
                        "temporal":  verdict.get("question_temporal_reason", ""),
                    }
                }
                return {**rec, "quality": quality}

            except Exception as e:
                if attempt == 2:
                    log.warning(f"Judge error for {rec.get('entity')}: {e}")
                    return {**rec, "quality": {
                        "pass": None,
                        "anchor_present": None, "gold_grounded": None,
                        "question_openended": None, "question_single": None,
                        "question_temporal": None,
                        "reasons": {"error": str(e)}
                    }}
                await asyncio.sleep(1)
    return rec

async def process_file(fpath: Path):
    if not fpath.exists():
        log.warning(f"[MISSING] {fpath}")
        return

    log.info(f"[Loading] {fpath.name}")
    records = []
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if N_TEST:
        records = records[:N_TEST]

    log.info(f"  {len(records)} registos a avaliar")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    judged = []
    for i in range(0, len(records), 200):
        batch = await asyncio.gather(*[judge_record(r, sem) for r in records[i:i+200]])
        judged.extend(batch)
        n_pass = sum(1 for r in batch if r.get("quality", {}).get("pass") is True)
        log.info(f"  Batch {i//200+1}: {len(batch)} avaliados | {n_pass} pass ({100*n_pass/len(batch):.1f}%)")

    # Estatísticas
    total = len(judged)
    if total == 0:
        log.info("  RESULTADO: 0 total")
        return

    n_pass    = sum(1 for r in judged if r.get("quality", {}).get("pass") is True)
    n_fail    = sum(1 for r in judged if r.get("quality", {}).get("pass") is False)
    n_unknown = sum(1 for r in judged if r.get("quality", {}).get("pass") is None)

    n_anchor_fail   = sum(1 for r in judged if r.get("quality", {}).get("anchor_present") is False)
    n_ground_fail   = sum(1 for r in judged if r.get("quality", {}).get("gold_grounded") is False)
    n_open_fail     = sum(1 for r in judged if r.get("quality", {}).get("question_openended") is False)
    n_single_fail   = sum(1 for r in judged if r.get("quality", {}).get("question_single") is False)
    n_temporal_fail = sum(1 for r in judged if r.get("quality", {}).get("question_temporal") is False)

    log.info(f"  RESULTADO: {total} total | PASS={n_pass} ({100*n_pass/total:.1f}%) | "
             f"FAIL={n_fail} ({100*n_fail/total:.1f}%) | UNKNOWN={n_unknown}")
    log.info("  Falhas por dimensão:")
    log.info(f"    anchor_present=False    : {n_anchor_fail} ({100*n_anchor_fail/total:.1f}%)")
    log.info(f"    gold_grounded=False     : {n_ground_fail} ({100*n_ground_fail/total:.1f}%)")
    log.info(f"    question_openended=False: {n_open_fail} ({100*n_open_fail/total:.1f}%)")
    log.info(f"    question_single=False   : {n_single_fail} ({100*n_single_fail/total:.1f}%)")
    log.info(f"    question_temporal=False : {n_temporal_fail} ({100*n_temporal_fail/total:.1f}%)")

    # Escreve ficheiro completo (com quality field em todos)
    with open(fpath, "w", encoding="utf-8") as f:
        for rec in judged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info(f"  [Guardado] {fpath} ({total} registos)")

    # Escreve ficheiro _clean (apenas pass=True)
    clean_path = fpath.with_name(fpath.stem + "_clean.jsonl")
    n_clean = 0
    with open(clean_path, "w", encoding="utf-8") as f:
        for rec in judged:
            if rec.get("quality", {}).get("pass") is True:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_clean += 1
    log.info(f"  [Guardado] {clean_path} ({n_clean} registos pass=True)")

async def main():
    build_clients()
    log.info(f"{len(CLIENTS)} endpoint(s) para judge")
    for fpath in FILES:
        await process_file(fpath)
    log.info("Done.")

if __name__ == "__main__":
    asyncio.run(main())
