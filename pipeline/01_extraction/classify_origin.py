#!/usr/bin/env python3
"""
classify_origin.py — Classificador de Origem para Entidades Temporais
=======================================================================
Passo 1: regex no infobox da Wikipedia — SÓ templates explícitos ({{BRAn, {{PORn, etc.)
         Só classifica quando há certeza absoluta. Nada de dedução por texto livre.
Passo 2: LLM (vLLM via OpenAI API) para todos os casos ambíguos.

Classes de saída:
  PT         — Portugal (inclui entidades históricas portuguesas)
  BR         — Brasil
  PALOP      — Angola, Moçambique, Cabo Verde, S.Tomé, Guiné-Bissau
  INT        — Internacional (EUA, UK, França, entidades históricas não PT, etc.)

Uso:
    # Só regex (sem LLM)
    python classify_origin.py --no-llm --out data/01_extraction/analysis_output/origin_labels.csv

    # Regex + LLM para ambíguos (requer vLLM a correr na porta 8089)
    python classify_origin.py --out data/01_extraction/analysis_output/origin_labels.csv

    # Só uma categoria
    python classify_origin.py --cat Biografia

    # Recalcular tudo mesmo que já exista output
    python classify_origin.py --force
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

RANK_DIR      = DATA / "01_extraction"
RANKING_CSV   = RANK_DIR / "entity_ranking.csv"
WIKI_BASE     = RANK_DIR / "newdata" / "temporal_relevant"
OUTPUT_DIR    = RANK_DIR / "analysis_output"
OUTPUT_CSV    = OUTPUT_DIR / "origin_labels.csv"

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_URL       = os.getenv("LOCAL_LLM_URL",   "http://localhost:8089/v1")
LLM_MODEL     = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-32B-FP8")
MAX_CONCURRENT = 30   # semáforo: pedidos LLM em simultâneo

# ── VALID_CLASSES ─────────────────────────────────────────────────────────────
VALID_CLASSES = {"PT", "BR", "PALOP", "INT"}


# ── Regex classifier ───────────────────────────────────────────────────────────

def read_infobox(wikitext: str, max_chars: int = 3000) -> str:
    """Extrai o bloco do infobox biográfico (primeiros max_chars do wikitext)."""
    return wikitext[:max_chars]


def classify_by_regex(wikitext: str) -> Optional[str]:
    """
    Classifica APENAS por templates explícitos de nacionalidade no infobox.
    Ex: {{BRAn, {{PORn, {{GBRn, etc.
    Só devolve um resultado quando há certeza absoluta.
    Qualquer caso ambíguo → None → vai para LLM.
    """
    infobox = wikitext[:4000]

    TEMPLATE_MAP = [
        (r'\{\{(?:BRA[nb]|BRZ)', "BR"),
        (r'\{\{(?:POR[nb]|PRTn|PRTb|PRT[^a-z0-9])', "PT"),
        (r'\{\{(?:AGO[nb]|ANGn)', "PALOP"),
        (r'\{\{MOZ[nb]', "PALOP"),
        (r'\{\{(?:CPV[nb]|CVn[^a-z])', "PALOP"),
        (r'\{\{(?:STP[nb]|STOn)', "PALOP"),
        (r'\{\{(?:GNB[nb]|GNBn)', "PALOP"),
        (r'\{\{(?:GBR[nb]|UKn[^a-z]|ENGn|ENGb|UKb)', "INT"),
        (r'\{\{(?:USA[nb]|EUAn)', "INT"),
        (r'\{\{(?:FRA[nb]|FRAn[^a-z])', "INT"),
        (r'\{\{(?:ESP[nb]|ESPn)', "INT"),
        (r'\{\{(?:DEU[nb]|ALEn|GERn)', "INT"),
        (r'\{\{(?:ITA[nb]|ITAn[^a-z])', "INT"),
        (r'\{\{ARG[nb]', "INT"),
        (r'\{\{(?:RUS[nb]|RUSn)', "INT"),
        (r'\{\{(?:JPN[nb]|JAPn)', "INT"),
        (r'\{\{(?:CHI[nb]|CHNn)', "INT"),
        (r'\{\{(?:NLD[nb]|NLDn|NLDb)', "INT"),
        (r'\{\{(?:HUN[nb]|HUNn)', "INT"),
        (r'\{\{(?:NOR[nb]|NORn|NORb)', "INT"),
        (r'\{\{(?:SWE[nb]|SWEb)', "INT"),
        (r'\{\{(?:POL[nb]|POLn)', "INT"),
        (r'\{\{AUT[nb]', "INT"),
        (r'\{\{(?:CAN[nb]|CANn)', "INT"),
        (r'\{\{(?:AUS[nb]|AUSn)', "INT"),
        (r'\{\{(?:BEL[nb]|BELn)', "INT"),
        (r'\{\{(?:CHE[nb]|CHEn)', "INT"),
        (r'\{\{(?:MEX[nb]|MEXn)', "INT"),
        (r'\{\{(?:KOR[nb]|KORn)', "INT"),
        (r'\{\{ISR[nb]', "INT"),
        (r'\{\{(?:COL[nb]|COLn)', "INT"),
        (r'\{\{(?:URU[nb]|URUb)', "INT"),
    ]
    for pattern, label in TEMPLATE_MAP:
        if re.search(pattern, infobox, re.IGNORECASE):
            return label

    return None  # ambíguo → LLM


def load_wikitext(entity_path_rel: str) -> str:
    path = RANK_DIR / entity_path_rel
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("wikitext", "")
    except Exception:
        return ""


# ── LLM classifier ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """És um classificador de origem geográfica de entidades enciclopédicas.
A tua tarefa é classificar a entidade descrita numa das seguintes classes:
- PT: Portuguesa — nascida/sediada em Portugal, ou entidade histórica com origem portuguesa (ex: exploradores, reis de Portugal, sismos em Portugal, tratados portugueses)
- BR: Brasileira — nascida/sediada no Brasil, ou com origem claramente brasileira
- PALOP: Países africanos de língua portuguesa (Angola, Moçambique, Cabo Verde, S.Tomé, Guiné-Bissau)
- INT: Internacional — qualquer outra origem (EUA, UK, França, entidades históricas sem ligação a Portugal como imperadores romanos, figuras bíblicas, etc.)

IMPORTANTE:
- Não existe classe HIST. Entidades históricas são PT se ligadas a Portugal, INT caso contrário.
- O facto de "Brasil" ou "Portugal" aparecer no texto NÃO significa que a entidade é desse país.
  Ex: uma batalha ocorrida no Brasil pode ser INT se foi iniciada por outro país.
- Para pessoas: usa a naturalidade ou cidadania principal.
- Para eventos/locais/organizações: classifica pelo país de origem/organização.
- Devolve APENAS o JSON: {"class": "PT"|"BR"|"PALOP"|"INT", "reason": "<1 frase>"}"""

USER_PROMPT_TEMPLATE = """Entidade: {title}
Categoria Wikipedia: {category}

Texto (infobox + início do artigo):
---
{text}
---

Classifica a origem desta entidade."""


async def classify_with_llm(
    title: str,
    category: str,
    wikitext: str,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> Tuple[str, str]:
    """Devolve (class, reason). Em caso de erro devolve ('OUTRO', 'LLM error')."""
    # Extrai o infobox + primeiro parágrafo limpo (máx 800 tokens ~3200 chars)
    clean = re.sub(r'\{\{[^}]{0,60}\}\}', '', wikitext[:4000])
    clean = re.sub(r'\[\[(?:[^\|]*\|)?([^\]]+)\]\]', r'\1', clean)
    clean = clean[:3000]

    prompt = USER_PROMPT_TEMPLATE.format(
        title=title, category=category, text=clean
    )
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=80,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content or ""
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            cls    = parsed.get("class", "INT").strip().upper()
            reason = parsed.get("reason", "")
            if cls not in VALID_CLASSES:
                cls = "INT"  # fallback seguro: inclassificável → INT
            return cls, reason
        except Exception as e:
            return "OUTRO", f"error: {e}"


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(args):
    print("🔄  A carregar entity_ranking.csv …")
    df = pd.read_csv(RANKING_CSV)
    df["category"] = df["path"].str.extract(r"temporal_relevant/([^/]+)/")
    df = df.dropna(subset=["category"])
    df = df.sort_values("rank").reset_index(drop=True)

    # Filtrar categorias pedidas
    if args.cat:
        df = df[df["category"].isin(args.cat)]
        if df.empty:
            print(f"❌ Categorias não encontradas: {args.cat}")
            sys.exit(1)

    # Carregar labels já existentes (idempotência)
    output_csv = Path(args.out) if args.out else OUTPUT_CSV
    existing = {}
    if output_csv.exists() and not args.force:
        prev = pd.read_csv(output_csv)
        # ONLY load success cases from cache, discard "fallback" to force retry
        prev_success = prev[prev["origin_method"] != "fallback"]
        existing = dict(zip(prev_success["path"], zip(prev_success["origin"], prev_success["origin_method"], prev_success.get("origin_reason", [""] * len(prev_success)))))
        print(f"✅  {len(existing)} labels já existentes carregadas do cache. (Ignorados fallbacks)")

    # ── Passo 1: Regex ────────────────────────────────────────────────────────
    labels = []
    need_llm = []

    print(f"\n📋  Passo 1 — Regex em {len(df)} entidades …")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="regex"):
        path = row["path"]

        # Já classificado?
        if path in existing:
            orig, method, reason = existing[path]
            labels.append({
                "path": path, "title": row["title"],
                "category": row["category"],
                "backlinks": row["backlinks"], "pageviews": row["pageviews"],
                "origin": orig, "origin_method": method, "origin_reason": reason,
            })
            continue

        wt  = load_wikitext(path)
        cls = classify_by_regex(wt)

        if cls is not None:
            labels.append({
                "path": path, "title": row["title"],
                "category": row["category"],
                "backlinks": row["backlinks"], "pageviews": row["pageviews"],
                "origin": cls, "origin_method": "regex", "origin_reason": "",
            })
        else:
            need_llm.append((row, wt))

    regex_done  = sum(1 for l in labels if l.get("origin_method") == "regex")
    print(f"   → Classificados por regex: {regex_done}")
    print(f"   → Carregados de cache:     {len(existing)}")
    print(f"   → Para LLM:                {len(need_llm)}")

    # ── Passo 2: LLM ─────────────────────────────────────────────────────────
    if need_llm and not args.no_llm:
        print(f"\n🤖  Passo 2 — LLM em {len(need_llm)} entidades ambíguas …")
        print(f"   URL: {LLM_URL}  |  Model: {LLM_MODEL}  |  Concorrência: {MAX_CONCURRENT}")

        client = AsyncOpenAI(api_key="EMPTY", base_url=LLM_URL)
        sem    = asyncio.Semaphore(MAX_CONCURRENT)

        async def process_one(row_wt):
            row, wt = row_wt
            cls, reason = await classify_with_llm(
                row["title"], row["category"], wt, client, sem
            )
            return {
                "path": row["path"], "title": row["title"],
                "category": row["category"],
                "backlinks": row["backlinks"], "pageviews": row["pageviews"],
                "origin": cls, "origin_method": "llm", "origin_reason": reason,
            }

        tasks   = [asyncio.create_task(process_one(rw)) for rw in need_llm]
        pbar    = tqdm(total=len(tasks), desc="LLM")
        
        output_csv = Path(args.out) if args.out else OUTPUT_CSV
        
        for coro in asyncio.as_completed(tasks):
            res = await coro
            labels.append(res)
            # Append in real-time to prevent data loss on crash
            pd.DataFrame([res]).to_csv(output_csv, mode='a', header=not output_csv.exists(), index=False)
            pbar.update(1)
            
        pbar.close()
        llm_done = len(need_llm)
        print(f"   → LLM classificou: {llm_done}")

    elif need_llm and args.no_llm:
        print(f"\n⚠️   {len(need_llm)} entidades ficam como 'fallback' (--no-llm activo)")
        for row, _ in need_llm:
            labels.append({
                "path": row["path"], "title": row["title"],
                "category": row["category"],
                "backlinks": row["backlinks"], "pageviews": row["pageviews"],
                "origin": "INT", "origin_method": "fallback", "origin_reason": "",
            })

    # ── Guardar CSV ───────────────────────────────────────────────────────────
    output_csv = Path(args.out) if args.out else OUTPUT_CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(labels)
    out_df.to_csv(output_csv, index=False)

    print(f"\n✅  Guardado em: {output_csv}")
    print(f"\n{'='*60}")
    print("  Resumo por categoria e origem")
    print(f"{'='*60}")
    for cat, grp in out_df.groupby("category"):
        dist = grp["origin"].value_counts().to_dict()
        dist_str = "  ".join(f"{k}:{v}" for k, v in
                             sorted(dist.items(), key=lambda x: -x[1]))
        print(f"  {cat:<25} ({len(grp):>5})  {dist_str}")

    print(f"\n{'='*60}")
    print("  Total por método:")
    print(out_df["origin_method"].value_counts().to_string())
    print("\n  Total por origem:")
    print(out_df["origin"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Classifica origem geográfica das entidades (regex + LLM)"
    )
    parser.add_argument("--cat",    nargs="+", help="Categorias a processar (default: todas)")
    parser.add_argument("--no-llm", action="store_true", help="Só regex, sem LLM")
    parser.add_argument("--force",  action="store_true", help="Recalcular tudo (ignora cache)")
    parser.add_argument("--out",    default=None, help=f"Path do CSV de output (default: {OUTPUT_CSV})")
    parser.add_argument("--concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"Pedidos LLM em simultâneo (default: {MAX_CONCURRENT})")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
