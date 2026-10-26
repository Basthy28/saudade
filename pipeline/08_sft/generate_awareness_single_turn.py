"""
generate_awareness_single_turn.py
=====================================
Gera a fusao fluida (GLUE) de 1 turno entre (Pergunta Alvo + Facto Distractor)
para o dataset de Awareness.

Garante:
- CONECTORES 100% NEUTROS (PROIBIDO 'Antes de', 'Depois de', 'Durante', 'Após')
- Zero alucinações (fidelidade estrita aos factos fornecidos)
- Sem perda de informação
- Sem fallbacks silenciosos: obriga o LLM a responder via retries
- Exclui Freguesia_Portugal, tal como o benchmark
- Concorrência assíncrona ultra-rápida (vLLM / AsyncOpenAI)
"""

import os, json, random, logging, time, asyncio, re
from pathlib import Path

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = os.environ.get("GEN_MODEL_ID", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
PORT           = int(os.environ.get("PORT", "8088"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "64"))
N_TEST         = int(os.environ.get("N_TEST", "0"))

DATA_DIR    = DATA / "06_split"
CHECKPT_DIR = DATA / "08_sft" / "checkpoints"
OUT_DIR     = DATA / "08_sft" / "output"

BLACKLIST = {"Festival RTP: A Melhor Canção de Sempre", "Toquinho"}
SKIP_CATS = {"Freguesia_Portugal"}

# ── System Prompt & Few-Shot Examples ─────────────────────────────────────────
GLUE_PROMPT = """\
Você é um especialista em linguística e síntese de perguntas em Português Europeu.

A sua tarefa é fundir UMA PERGUNTA BASE e UM FACTO CONTEXTUAL (DISTRACTOR) numa ÚNICA PERGUNTA NATURAL e FLUÍDA de 1 turno.

REGRAS ESTRITAS DE NEUTRALIDADE TEMPORAL E FIDELIDADE FACTUAL:
1. **PROIBIDO CONECTORES TEMPORAIS DE ORDEM:** NUNCA uses palavras de sequência ou ordem temporal como "Antes de", "Depois de", "Durante", "Após", "Anteriormente", "Posteriormente" ou "Previamente". Isso daria um vazamento/pista temporal indevida sobre a resposta!
2. **USAR APENAS CONECTORES NEUTROS E VARIADOS:** Usa EXCLUSIVAMENTE conectores de contexto neutros (ex: "Considerando que...", "Sabendo que...", "No contexto em que...", "Tendo presente que...", "Relativamente a...", "A propósito de...", "No âmbito de...", "Tendo em conta que...", etc.). VARIA substancialmente estes conectores entre as perguntas.
3. **ZERO ALUCINAÇÕES:** NUNCA inventes datas, nomes, locais ou factos que não estejam no texto fornecido.
4. **ZERO PERDA DE INFORMAÇÃO:** A pergunta final deve preservar a intenção e os detalhes exatos da pergunta base e mencionar o facto contextual.
5. **PORTUGUÊS EUROPEU:** Usa português europeu correto e natural (ex: "facto", "função", "cargo").
6. **ÚNICO OBJECTIVO DE RESPOSTA:** A pergunta fundida deve continuar a pedir APENAS a resposta à Pergunta Base.
7. Responde APENAS no formato JSON: {{"fused_question": "..."}}

---
EXEMPLO 1 (Conector de enquadramento neutro):
Pergunta Base: "Qual cargo exerceu Caetano Moniz de Vasconcelos no Distrito Autónomo de Ponta Delgada?"
Facto Contextual: "Exerceu o cargo de Governador Civil do Distrito da Horta em 1915."
JSON Output: {{"fused_question": "Sabendo que Caetano Moniz de Vasconcelos exerceu o cargo de Governador Civil do Distrito da Horta em 1915, qual foi o cargo que desempenhou no Distrito Autónomo de Ponta Delgada?"}}

EXEMPLO 2 (Conector neutro contextual):
Pergunta Base: "Qual foi o acontecimento envolvendo Francisco de Andrade Albuquerque em 1856-12-16?"
Facto Contextual: "Faleceu em 23 de fevereiro de 1933."
JSON Output: {{"fused_question": "Considerando que Francisco de Andrade Albuquerque faleceu a 23 de fevereiro de 1933, qual foi o acontecimento registado a 16 de dezembro de 1856?"}}

EXEMPLO 3 (Conector neutro relacional):
Pergunta Base: "Em que ano se iniciou o mandato de Fernando Santos Martins na TAP?"
Facto Contextual: "Especializou-se em Engenharia Industrial na Alemanha em 1955."
JSON Output: {{"fused_question": "A propósito da sua especialização em Engenharia Industrial na Alemanha em 1955, em que ano se iniciou o mandato de Fernando Santos Martins na TAP?"}}

EXEMPLO 4 (Conector neutro de âmbito):
Pergunta Base: "Quem sucedeu a Carlos Lloyd Braga no cargo?"
Facto Contextual: "Foi elevado postumamente a Grã-Cruz da Ordem da Instrução Pública em 1998."
JSON Output: {{"fused_question": "Tendo presente a sua elevação póstuma a Grã-Cruz da Ordem da Instrução Pública em 1998, quem sucedeu a Carlos Lloyd Braga no cargo?"}}
---

Entidade: {entity}
Pergunta Base: {question}
Facto Contextual: {distractor_fact}

JSON Output:
"""

# ── Load Items ────────────────────────────────────────────────────────────────
def load_awareness_items() -> list:
    file_path = DATA_DIR / "saudade_awareness_pt_train.json"
    if not file_path.exists():
        file_path = DATA_DIR / "saudade_awareness_pt_train.json"
    
    log.info(f"A carregar dataset de Awareness: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for category, entities in data.items():
        if not isinstance(entities, dict): continue
        for entity, content in entities.items():
            if not isinstance(content, dict) or entity in BLACKLIST: continue
            
            qa_list = content.get("awareness_qa", [])
            if not qa_list:
                for _, ev in content.items():
                    if isinstance(ev, dict) and "awareness_qa" in ev:
                        qa_list.extend(ev["awareness_qa"])
                        
            for qa in qa_list:
                future_f = qa.get("future_fact", "")
                past_f   = qa.get("past_fact", "")
                
                # Past questions -> Distractor is future_fact
                for qtype, q in qa.get("past_questions", {}).items():
                    if q and future_f:
                        answers = qa.get("past_answers") or [qa.get("past_ground_truth", "")]
                        items.append({
                            "domain": "awareness",
                            "category": category,
                            "entity": entity,
                            "question_type": f"past_{qtype}",
                            "question": q,
                            "answer": answers[0] if answers else "",
                            "all_answers": answers,
                            "target_fact": past_f,
                            "distractor_fact": future_f,
                        })
                        
                # Future questions -> Distractor is past_fact
                for qtype, q in qa.get("future_questions", {}).items():
                    if q and past_f:
                        answers = qa.get("future_answers") or [qa.get("future_ground_truth", "")]
                        items.append({
                            "domain": "awareness",
                            "category": category,
                            "entity": entity,
                            "question_type": f"future_{qtype}",
                            "question": q,
                            "answer": answers[0] if answers else "",
                            "all_answers": answers,
                            "target_fact": future_f,
                            "distractor_fact": past_f,
                        })

    log.info(f"Carregados {len(items)} itens de Awareness para fusao GLUE.")
    return items

def parse_fused_response(raw: str) -> str | None:
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            fq = data.get("fused_question", "").strip()
            if fq and len(fq) > 10:
                # Extra safety check: reject if LLM output still has temporal order words
                bad_words = [r"\bantes de\b", r"\bdepois de\b", r"\bdurante\b", r"\bapós\b", r"\banteriormente\b", r"\bposteriormente\b"]
                if any(re.search(w, fq, re.IGNORECASE) for w in bad_words):
                    return None
                return fq
    except Exception:
        pass
    return None

# ── Main Generator Loop ───────────────────────────────────────────────────────
async def main():
    items = load_awareness_items()
    if N_TEST > 0:
        items = items[:N_TEST]
        log.info(f"Modo de teste: limitado a {N_TEST} itens.")

    CHECKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPT_DIR / "awareness_single_turn_ckpt.jsonl"
    
    done_ids = set()
    results = []
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        human_txt = rec.get("conversations", [{}])[0].get("content", "")
                        # Strict rejection if contains temporal order words
                        bad_words = [r"\bantes de\b", r"\bdepois de\b", r"\bdurante\b", r"\bapós\b", r"\banteriormente\b", r"\bposteriormente\b"]
                        if any(re.search(w, human_txt, re.IGNORECASE) for w in bad_words):
                            continue
                        results.append(rec)
                        fp = f"{rec['entity']}_{rec['question_type']}_{rec['question']}"
                        done_ids.add(fp)
                    except Exception:
                        pass
        log.info(f"Checkpoint validado: {len(results)} itens reais do LLM mantidos.")

    todo_items = []
    for it in items:
        fp = f"{it['entity']}_{it['question_type']}_{it['question']}"
        if fp not in done_ids:
            todo_items.append((fp, it))

    log.info(f"Restantes para processar pelo LLM: {len(todo_items)}")

    if not todo_items:
        log.info("Todos os itens ja foram processados pelo LLM!")
    else:
        from openai import AsyncOpenAI
        endpoints_raw = os.environ.get("VLLM_ENDPOINTS", f"127.0.0.1:{PORT}")
        endpoints = [e.strip() for e in endpoints_raw.split(",") if e.strip()]
        
        clients = [AsyncOpenAI(base_url=f"http://{ep}/v1", api_key="vllm", timeout=120.0, max_retries=1) for ep in endpoints]
        log.info(f"vLLM pool ativado com {len(clients)} endpoints: {endpoints}")

        sem = asyncio.Semaphore(MAX_CONCURRENT)
        ckpt_file = open(ckpt_path, "a", encoding="utf-8")

        async def worker(item_pair):
            fp, item = item_pair
            prompt = GLUE_PROMPT.format(
                entity=item["entity"],
                question=item["question"],
                distractor_fact=item["distractor_fact"]
            )
            
            fused_q = None
            max_attempts = 5
            
            for attempt in range(max_attempts):
                async with sem:
                    client = random.choice(clients)
                    try:
                        resp = await client.chat.completions.create(
                            model=MODEL_ID,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            top_p=0.92,
                            max_tokens=256,
                        )
                        raw_text = resp.choices[0].message.content or ""
                        fused_q = parse_fused_response(raw_text)
                        if fused_q:
                            break
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            log.error(f"Erro final apos {max_attempts} tentativas para {item['entity']}: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))

            if not fused_q:
                log.error(f"[SKIPPED] Falha de geracao LLM para item: {item['entity']} | {item['question']}")
                return None

            record = {
                "domain": "awareness",
                "category": item["category"],
                "entity": item["entity"],
                "question_type": item["question_type"],
                "conversations": [
                    {"role": "human", "content": fused_q},
                    {"role": "gpt", "content": item["answer"]}
                ],
                "all_answers": item["all_answers"],
                "target_fact": item["target_fact"],
                "distractor_fact": item["distractor_fact"],
                "original_question": item["question"]
            }
            
            ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_file.flush()
            return record

        t0 = time.time()
        tasks = [asyncio.create_task(worker(tp)) for tp in todo_items]
        completed = 0
        for fut in asyncio.as_completed(tasks):
            res = await fut
            if res:
                results.append(res)
            completed += 1
            if completed % 1000 == 0 or completed == len(todo_items):
                log.info(f"Progresso LLM GLUE: {completed}/{len(todo_items)} ({completed/max(1, time.time()-t0):.1f} it/s)")

        ckpt_file.close()

    # Save the aggregated output (Freguesia_Portugal excluded: the benchmark
    # drops this category, so the SFT corpus does not use it either).
    final_jsonl = OUT_DIR / "saudade_sft_single_turn_relation_with_distractors.jsonl"
    kept = 0
    with open(final_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            if r.get("category") in SKIP_CATS:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1

    log.info(f"\n[OK] {kept} validated records written to {final_jsonl}")

if __name__ == "__main__":
    asyncio.run(main())
