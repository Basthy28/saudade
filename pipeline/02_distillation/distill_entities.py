
import json
import re
import os
import asyncio
from pathlib import Path
from typing import List, Dict
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

# Configurações
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = DATA / "01_extraction" / "newdata" / "temporal_relevant"
OUTPUT_DIR = DATA / "02_distillation" / "temporal_cleaned"

MIN_UNIQUE_YEARS = 4
MAX_CONCURRENT_LLM = 30  # pedidos simultâneos REAIS ao LLM

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8088/v1")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")

client = AsyncOpenAI(api_key="EMPTY", base_url=LOCAL_LLM_URL)

def extract_from_infobox(wikitext: str) -> List[Dict]:
    facts = []
    dni_match = re.search(r'\|nascimento_data\s*=\s*(?:\{\{[^}]+\}\}\s*)?\{\{(?:dni|nascimento)\|[^|]*?\|?(\d{1,2})\|(\d{1,2})\|(\d{4})', wikitext, re.I)
    if dni_match:
        facts.append({"data_inicio": f"{dni_match.group(3)}-{int(dni_match.group(2)):02d}-{int(dni_match.group(1)):02d}", "data_fim": None, "facto": "Nasceu", "tipo": "nascimento", "contexto": "Infobox."})
    elif m := re.search(r'\|nascimento_data\s*=\s*.*?(\d{4})', wikitext, re.I):
        facts.append({"data_inicio": m.group(1), "data_fim": None, "facto": "Nasceu", "tipo": "nascimento", "contexto": "Infobox."})
    if morte_match := re.search(r'\|(?:morte|falecimento)_data\s*=\s*(?:\{\{[^}]+\}\}\s*)?\{\{(?:morte|falecimento)\|(\d{1,2})\|(\d{1,2})\|(\d{4})', wikitext, re.I):
        facts.append({"data_inicio": f"{morte_match.group(3)}-{int(morte_match.group(2)):02d}-{int(morte_match.group(1)):02d}", "data_fim": None, "facto": "Faleceu", "tipo": "morte", "contexto": "Infobox."})
    return facts

def clean_content_for_llm(text: str) -> str:
    text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\{\{nowrap\|([^}]+)\}\}', r'\1', text)
    text = re.sub(r'style="[^"]*"|align="[^"]*"|rowspan="\d+"|colspan="\d+"', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def extract_year_dense_blocks(wikitext: str, content: str, years_list: List[str]) -> str:
    full_text = content + "\n\n" + wikitext
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', full_text) if p.strip()]
    indices = set()
    for i, p in enumerate(paragraphs):
        if any(y in p for y in years_list):
            indices.update([max(0, i-1), i, min(len(paragraphs)-1, i+1)])
    return clean_content_for_llm("\n\n---\n\n".join(paragraphs[i] for i in sorted(indices)))

async def extract_events_with_llm(rich_text: str, title: str) -> List[Dict]:
    prompt = f"""Extrai eventos biográficos e relações temporais destas notas sobre '{title}'.
Instruções OBRIGATÓRIAS:
1. Extrai factos temporais importantes. Não te limites a ações predefinidas.
2. Para relações que duram no tempo, TENTA SEMPRE identificar O INÍCIO E O FIM (data_inicio E data_fim).
3. Datas DEVEM ter o formato YYYY-MM-DD, YYYY-MM ou YYYY.
4. Devolve APENAS JSON válido. Nenhuma observação extra.

Estrutura obrigatória com chave raiz "events":
{{
  "events": [
    {{"data_inicio": "YYYY", "data_fim": "YYYY", "facto": "descrição", "tipo": "categoria", "contexto": "detalhe"}}
  ]
}}

Texto:
{rich_text[:6000]}
"""
    try:
        response = await client.chat.completions.create(
            model=LOCAL_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise data extraction agent. You only output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        out_text = response.choices[0].message.content or ""
        out_text = out_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(out_text).get("events", [])
    except Exception:
        return []

async def process_entity(category: str, entity_path: Path, out_dir: Path, sem: asyncio.Semaphore):
    """Só é chamada para entidades que AINDA NÃO têm output — sem skip desnecessário."""
    out_file = out_dir / category / f"{entity_path.name}.json"

    f = entity_path / 'entity_data.json'
    if not f.exists():
        return

    try:
        data = json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        return

    title = data.get('title', '')
    wikitext = data.get('wikitext', '')
    sections = data.get('sections', [])
    content = '\n'.join([s.get('content', [''])[0] for s in sections if s.get('content')])

    years_found = sorted(set(re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', wikitext + ' ' + content)))
    if len(years_found) < MIN_UNIQUE_YEARS:
        return

    structured_facts = extract_from_infobox(wikitext)
    rich_text_blocks = extract_year_dense_blocks(wikitext, content, years_found)

    if rich_text_blocks:
        async with sem:  # semáforo APENAS na chamada ao LLM
            semantic_facts = await extract_events_with_llm(rich_text_blocks, title)
        structured_facts.extend(semantic_facts)

    if structured_facts:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps({"title": title, "wikidata_id": data.get("wikidata_id"), "events": structured_facts}, ensure_ascii=False, indent=2), encoding="utf-8")

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cats', nargs='+', default=[d.name for d in INPUT_DIR.iterdir() if d.is_dir()])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pré-filtrar: só entidades sem output ainda
    all_tasks_info = []
    skipped = 0
    for cat in args.cats:
        cat_dir = INPUT_DIR / cat
        if not cat_dir.exists():
            continue
        out_cat = OUTPUT_DIR / cat
        for ent_dir in cat_dir.iterdir():
            if not ent_dir.is_dir():
                continue
            out_file = out_cat / f"{ent_dir.name}.json"
            if out_file.exists():
                skipped += 1
            else:
                all_tasks_info.append((cat, ent_dir))

    total = len(all_tasks_info)
    node_id = int(os.getenv("SLURM_NODEID", "0"))
    num_nodes = int(os.getenv("SLURM_NNODES", "1"))

    if num_nodes > 1:
        all_tasks_info = [t for i, t in enumerate(all_tasks_info) if i % num_nodes == node_id]

    print(f"Iniciando Destilação ASYNC Global (Fase 1) em {LOCAL_LLM_URL}")
    print(f"Node {node_id}/{num_nodes} | Processando {len(all_tasks_info)} tarefas (de {total} globais)")
    print(f"Model: {LOCAL_MODEL_NAME} | Concorrência LLM: {MAX_CONCURRENT_LLM}")
    print(f"Entidades já feitas (skip): {skipped} | Por processar agora: {len(all_tasks_info)}")

    sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)

    # Wrapper para isolar erros: 1 task falhada não cancela as outras
    async def safe_process(cat, ent_dir):
        try:
            await process_entity(cat, ent_dir, OUTPUT_DIR, sem)
        except Exception:
            pass

    # UM ÚNICO gather global — sem aborts entre categorias
    tasks = [safe_process(cat, ent_dir) for cat, ent_dir in all_tasks_info]
    await tqdm_asyncio.gather(*tasks, desc="Total")

if __name__ == "__main__":
    asyncio.run(main())
