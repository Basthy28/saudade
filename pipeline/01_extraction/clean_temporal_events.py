"""Review extracted temporal events against the source article and fix dates.

Reads the extracted events from ``01_extraction/newdata/temporal_cleaned``,
checks each one against the stored wikitext and writes the corrected events to
``01_extraction/newdata/temporal_reviewed``.
"""
import json
import re
import os
import asyncio
from pathlib import Path
from typing import List, Optional
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

BASE_DIR = DATA / "01_extraction"
INPUT_DIR = BASE_DIR / "newdata" / "temporal_cleaned"
OUTPUT_DIR = BASE_DIR / "newdata" / "temporal_reviewed"
RAW_WIKI_DIR = BASE_DIR / "newdata" / "temporal_relevant"

MAX_CONCURRENT_LLM = 20

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8088/v1")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct-AWQ")

client = AsyncOpenAI(api_key="EMPTY", base_url=LOCAL_LLM_URL)


def normalize_date(d) -> Optional[str]:
    """Remove ghost days (YYYY-MM-01 → YYYY-MM) e ghost months (YYYY-01-01 → YYYY)."""
    if not d:
        return None
    d = str(d).strip()
    # YYYY-01-01 → YYYY (só o ano é real)
    if re.match(r'^\d{4}-01-01$', d):
        return d[:4]
    # YYYY-12-31 → YYYY
    if re.match(r'^\d{4}-12-31$', d):
        return d[:4]
    # YYYY-MM-01 → YYYY-MM (só o mês é real, dia 1 artificial)
    if re.match(r'^\d{4}-\d{2}-01$', d):
        return d[:7]
    return d


def extract_year_dense_blocks(full_text: str, years_list: List[str]) -> str:
    """Extrai parágrafos que contêm os anos relevantes + vizinhos (sliding window)."""
    if not years_list:
        return full_text[:6000]
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', full_text) if p.strip()]
    indices = set()
    for i, p in enumerate(paragraphs):
        if any(y in p for y in years_list):
            indices.update([max(0, i-1), i, min(len(paragraphs)-1, i+1)])
    blocks = "\n\n---\n\n".join(paragraphs[i] for i in sorted(indices))
    blocks = re.sub(r'\{\{[^}]+\}\}', '', blocks)
    blocks = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', blocks)
    return blocks[:8000]


def get_wiki_text(category: str, entity_stem: str) -> str:
    """Lê o wikitext original da entidade."""
    raw_path = RAW_WIKI_DIR / category / entity_stem / "entity_data.json"
    if not raw_path.exists():
        return ""
    try:
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
        wikitext = raw.get('wikitext', '')
        sections = raw.get('sections', [])
        content = '\n'.join([s.get('content', [''])[0] for s in sections if s.get('content')])
        return wikitext + "\n\n" + content
    except Exception:
        return ""


async def review_event_with_llm(event: dict, title: str, wiki_context: str) -> dict:
    """Revisa um evento: corrige datas e decide se excluir. Devolve o evento corrigido."""

    prompt = f"""És um auditor rigoroso de dados biográficos temporais. Analisa o seguinte evento extraído sobre '{title}' e corrije-o com base no Texto Original da Wikipédia.

TEXTO ORIGINAL (fonte da verdade):
---
{wiki_context}
---

EVENTO A AUDITAR:
{json.dumps(event, ensure_ascii=False, indent=2)}

TAREFAS (por ordem de prioridade):

1. **data_inicio / data_fim — Formato correto:**
   - Se a data for `YYYY-MM-01` mas o texto só menciona o mês → devolve `YYYY-MM` (sem o dia 1 artificial)
   - Se a data for `YYYY-01-01` ou `YYYY-12-31` mas o texto só menciona o ano → devolve só `YYYY`
   - Se a data for `null` mas encontrares a data no texto → preenche com `YYYY`, `YYYY-MM` ou `YYYY-MM-DD`
   - Se a data for `null` e NÃO encontrares nenhuma data no texto → mantém `null`
   - Formato preferido: `YYYY-MM-DD` > `YYYY-MM` > `YYYY` (usa o mais específico que o texto suportar)

2. **excluir:**
   - `true` se: o facto é uma alucinação (não existe no texto), OU ambas as datas são `null` e irrecuperáveis, OU o facto é completamente irrelevante temporalmente
   - `false` em todos os outros casos (incluindo quando só `data_fim` é null — isso é normal)

Devolve APENAS um JSON com EXATAMENTE estas chaves (sem mais, sem menos):
{{
  "data_inicio": "...",
  "data_fim": "...",
  "excluir": true/false
}}
"""
    try:
        response = await client.chat.completions.create(
            model=LOCAL_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict data auditor. Output only valid JSON with exactly the keys: data_inicio, data_fim, excluir."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        out_text = response.choices[0].message.content or ""
        out_text = out_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        correction = json.loads(out_text)

        # Aplica apenas os campos esperados — preserva o resto do evento
        updated = dict(event)
        updated["data_inicio"] = normalize_date(correction.get("data_inicio", event.get("data_inicio")))
        updated["data_fim"] = normalize_date(correction.get("data_fim", event.get("data_fim")))
        updated["excluir"] = bool(correction.get("excluir", False))
        return updated
    except Exception:
        # Em caso de erro, mantém o evento original e marca excluir=False
        updated = dict(event)
        updated["data_inicio"] = normalize_date(event.get("data_inicio"))
        updated["data_fim"] = normalize_date(event.get("data_fim"))
        updated["excluir"] = False
        return updated


async def process_file(category: str, entity_file: Path, sem: asyncio.Semaphore):
    """Processa um ficheiro JSON: revisa todos os seus eventos e guarda em temporal_reviewed/."""
    out_file = OUTPUT_DIR / category / entity_file.name
    if out_file.exists():
        return  # Idempotência — já revisto nesta run ou anterior

    try:
        data = json.loads(entity_file.read_text(encoding='utf-8'))
    except Exception:
        return

    events = data.get("events", [])
    if not events:
        return

    title = data.get("title", "")
    wiki_text = get_wiki_text(category, entity_file.stem)

    # Para cada evento, extrai a janela deslizante focada nos anos desse evento
    async def review_single(event):
        years = []
        for key in ["data_inicio", "data_fim"]:
            v = event.get(key) or ""
            years.extend(re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', v))
        context = extract_year_dense_blocks(wiki_text, years) if years else wiki_text[:4000]
        async with sem:
            return await review_event_with_llm(event, title, context)

    # Todos os eventos desta entidade em paralelo (limitados pelo semáforo global)
    reviewed_events = await asyncio.gather(*[review_single(e) for e in events])

    # Guarda o resultado
    out_file.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "title": title,
        "wikidata_id": data.get("wikidata_id"),
        "events": list(reviewed_events)
    }
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cats', nargs='+', default=[d.name for d in INPUT_DIR.iterdir() if d.is_dir()])
    parser.add_argument('--sample', type=int, default=None, help='Processar só N ficheiros por categoria (teste)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)

    print(f"Iniciando Fase 2 — Revisão LLM em {LOCAL_LLM_URL}")
    print(f"Model: {LOCAL_MODEL_NAME} | Concorrência LLM: {MAX_CONCURRENT_LLM}")

    async def safe_process(cat, f):
        try:
            await process_file(cat, f, sem)
        except Exception:
            pass

    for cat in args.cats:
        cat_dir = INPUT_DIR / cat
        if not cat_dir.is_dir():
            continue
        files = list(cat_dir.glob("*.json"))
        if args.sample:
            files = files[:args.sample]

        print(f"\nCategoria: {cat} ({len(files)} ficheiros)")
        tasks = [safe_process(cat, f) for f in files]
        await tqdm_asyncio.gather(*tasks, desc=cat)

    print("\nFase 2 Concluída!")


if __name__ == "__main__":
    asyncio.run(main())
