"""
generate_openended.py
=====================
Geração de perguntas open-ended com temporalidade explícita.

COGNITION: a pergunta é formulada de modo que o anchor temporal seja
necessário na resposta. As estruturas frásicas mantêm-se diversas, mas o
gerador escolhe estruturas que activem o anchor.

AWARENESS: a pergunta activa a relação temporal entre o distractor e o
anchor. Um modelo que não saiba qual facto veio primeiro falha a resposta.

A proibição de "Quando" mantém-se para perguntas trivialmente fechadas; a
instrução de evitar datas é removida, para que o anchor apareça naturalmente
na resposta elaborada. Temperatura: 0.4.

Output: data/07_openended/

Usage:
  N_TEST=8 VLLM_ENDPOINTS="ip:port" python3 generate_openended.py
"""

import os, json, asyncio, logging, hashlib, re, random
from pathlib import Path
from openai import AsyncOpenAI

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

logging.basicConfig(level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_BASE      = DATA
SPLIT_DIR      = DATA_BASE / "06_split"
OUT_DIR        = DATA_BASE / "07_openended"           # ← novo diretório
CKPT_DIR       = OUT_DIR / "checkpoints"
WIKI_DIRS      = [
    DATA / "01_extraction" / "cats",
    DATA / "01_extraction" / "newdata" / "temporal_relevant",
]
GEN_MODEL_ID   = os.environ.get("GEN_MODEL_ID", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
VLLM_ENDPOINTS = os.environ.get("VLLM_ENDPOINTS", "")
PORT           = int(os.environ.get("PORT", "8088"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2000"))
N_TEST         = int(os.environ.get("N_TEST", "0"))   # 0 = run all by default
SHARD_SIZE     = int(os.environ.get("SHARD_SIZE", "500"))  # checkpoint a cada N

# Marcadores de contexto insuficiente para filtrar na geração (fail-fast)
INSUFF_MARKERS = [
    "informação fornecida não permite",
    "não é fornecida no extracto",
    "não é mencionada no excerto",
    "não consta no excerto",
    "não é possível determinar",
    "INSUFFICIENT_CONTEXT",
    "não está disponível no excerto",
    "não é explicitada",
    "não é referido no excerto",
    "não são mencionados no excerto",
    "o excerto não fornece",
    "o texto não menciona",
    "o excerto não menciona",
    "sem informação suficiente",
    "o extracto não contém",
    "o extracto não fornece",
]

# Contadores de qualidade para log final
_stats = {"total": 0, "skipped_wiki": 0, "skipped_insufficient": 0,
          "skipped_anchor_mismatch": 0, "skipped_insuff_gold": 0, "saved": 0}

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

# ── Wiki loader ────────────────────────────────────────────────────────────────
def _strip_wikitext(raw: str) -> str:
    """Strip wiki markup from raw wikitext to get readable prose."""
    text = re.sub(r'\{\{[^{}]*\}\}', '', raw)
    text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'^[\|!].*$', '', text, flags=re.MULTILINE)
    text = re.sub(r"'{2,3}", '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def load_wiki(entity: str, max_words: int = 1500) -> str:
    slug = entity.replace(" ", "_")
    for base in WIKI_DIRS:
        if not base.exists(): continue
        for cat in base.iterdir():
            if not cat.is_dir(): continue
            for p in [cat / slug / "entity_data.json", cat / f"{slug}.json"]:
                if not p.exists(): continue
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if "sections" in data:
                        blocks = []
                        for s in data.get("sections", []):
                            h = s.get("heading", "")
                            c = s.get("content", [])
                            if isinstance(c, list): c = "\n".join(c)
                            t = f"{h}\n{c}".strip() if h else c.strip()
                            if t: blocks.append(t)
                        text = "\n\n".join(blocks)
                    else:
                        wikitext = data.get("wikitext", "")
                        if wikitext and len(wikitext) > 100:
                            stripped = _strip_wikitext(wikitext)
                            if len(stripped.split()) > 30:
                                return " ".join(stripped.split()[:max_words])
                        parts = []
                        for k, v in data.items():
                            if k in ("wikitext", "url", "title", "language"): continue
                            if isinstance(v, str) and len(v) > 20: parts.append(v)
                            elif isinstance(v, list):
                                for item in v:
                                    if isinstance(item, str) and len(item) > 20: parts.append(item)
                                    elif isinstance(item, dict):
                                        fact = item.get("facto") or item.get("content")
                                        if fact: parts.append(str(fact))
                        text = "\n\n".join(parts)
                    words = text.split()
                    return " ".join(words[:max_words])
                except Exception: continue
    return ""

# ── LLM call ──────────────────────────────────────────────────────────────────
async def llm(prompt: str, max_tokens: int = 300, temperature: float = 0.7) -> str:
    for attempt in range(3):
        try:
            r = await next_client().chat.completions.create(
                model=GEN_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2: log.warning(f"LLM error: {e}"); return ""
            await asyncio.sleep(1)
    return ""

# ── Wiki Paragraph Selection ───────────────────────────────────────────────────
def is_useful_paragraph(text: str, anchor: str) -> bool:
    words = text.split()
    if len(words) < 40:
        return False
    has_number = any(
        w.isdigit() or re.match(r'^(19|20)\d{2}$', w)
        for w in words
    )
    if not has_number:
        return False
    anchor_words = [w.lower() for w in anchor.split() if len(w) > 3]
    text_lower = text.lower()
    if anchor_words and not any(w in text_lower for w in anchor_words):
        return False
    return True

def pick_wiki_paragraph(wiki: str, anchor: str, distractor: str = "", max_words: int = 800) -> str:
    if not wiki: return ""
    paragraphs = [p.strip() for p in wiki.split("\n\n") if len(p.strip()) > 80]
    if not paragraphs:
        words = wiki.split()
        return " ".join(words[:max_words])

    anchor_words = set(w.lower() for w in anchor.split() if len(w) > 3)
    distractor_words = set(w.lower() for w in distractor.split() if len(w) > 3)
    scored = []
    for p in paragraphs:
        p_lower = p.lower()
        score = sum(1 for w in anchor_words if w in p_lower) * 2
        score += sum(1 for w in distractor_words if w in p_lower)
        scored.append((score, p))
    scored.sort(key=lambda x: -x[0])

    for score, para in scored[:3]:
        candidate = para
        if len(scored) > 1 and scored[1][0] > 0:
            candidate = para + "\n\n" + scored[1][1]
        if is_useful_paragraph(candidate, anchor):
            words = candidate.split()
            return " ".join(words[:max_words])

    log.warning(f"Nenhum parágrafo útil para anchor='{anchor}' — usando melhor disponível")
    result = scored[0][1]
    words = result.split()
    return " ".join(words[:max_words])

# ── Anchor verification ────────────────────────────────────────────────────────
def anchor_in_gold(anchor: str, gold: str) -> bool:
    anchor_norm = anchor.lower().strip()
    gold_norm = gold.lower()
    if anchor_norm in gold_norm:
        return True
    anchor_years = re.findall(r'\b(19|20)\d{2}\b', anchor)
    gold_years = re.findall(r'\b(19|20)\d{2}\b', gold)
    if anchor_years and any(y in gold_years for y in anchor_years):
        return True
    anchor_key_words = [w for w in anchor.split() if len(w) > 4]
    if anchor_key_words and all(w.lower() in gold_norm for w in anchor_key_words[:2]):
        return True
    return False

def gold_has_insuff_markers(gold: str) -> bool:
    """Verifica se a gold answer tem marcadores de contexto insuficiente."""
    gold_lower = gold.lower()
    return any(m.lower() in gold_lower for m in INSUFF_MARKERS)

# ── Prompts (v9 — TEMPORALIDADE EXPLÍCITA + ANTI-LEAKAGE) ──────────────────────
async def gen_cognition_question(entity: str, original_q: str, anchor: str, wiki: str) -> str:
    # Estruturas que tendem a activar o anchor temporal sem perguntas trivialmente fechadas
    all_structures = [
        '"Em que fase do percurso de {entity} teve lugar [facto/contexto do anchor]?"',
        '"Que marcos temporais caracterizaram [facto/contexto do anchor] na trajetória de {entity}?"',
        '"Reconstrói a sequência de acontecimentos que culminou em [facto/contexto do anchor] na vida de {entity}."',
        '"Situa [facto/contexto do anchor] no panorama cronológico de {entity}, descrevendo o que antecedeu esse momento."',
        '"Que circunstâncias, e em que momento, rodearam [facto/contexto do anchor] na carreira de {entity}?"',
        '"Pormenoriza as etapas que antecederam [facto/contexto do anchor] na trajetória de {entity}."',
        '"Relata os acontecimentos que, numa determinada fase da vida de {entity}, conduziram a [facto/contexto do anchor]."',
        '"Identifica o momento e as condições em que [facto/contexto do anchor] marcou o percurso de {entity}."',
        '"Que contexto histórico e temporal envolveu [facto/contexto do anchor] na trajetória de {entity}?"',
        '"Caracteriza a fase em que [facto/contexto do anchor] ocorreu, relacionando-a com o restante percurso de {entity}."',
        '"Descreve as circunstâncias em que [facto/contexto do anchor] se enquadra na história de {entity}."',
        '"Em que ponto da cronologia de {entity} se situa [facto/contexto do anchor] e o que o antecedeu?"',
    ]
    sampled = random.sample(all_structures, min(5, len(all_structures)))
    structures_block = "\n".join(f"    * {s}" for s in sampled)

    return await llm(f"""Estás a construir um benchmark académico de avaliação de LLMs em português europeu.

Tens esta pergunta fechada sobre a entidade "{entity}":
PERGUNTA ORIGINAL: {original_q}
RESPOSTA CURTA ESPERADA (anchor): {anchor}

EXCERTO DA WIKIPEDIA (a tua pergunta deve ser completamente respondível com este texto):
{wiki}

Transforma-a numa ÚNICA pergunta aberta, focada e específica, seguindo TODAS as regras abaixo:

REGRA CENTRAL (obrigatória): A pergunta deve ser formulada de tal modo que um modelo sem conhecimento temporal explícito (datas, sequências, períodos) NÃO consiga dar uma resposta completa e correcta. O anchor "{anchor}" deve ser necessário e natural na resposta — não basta contextualizar o tema, tem de haver ancoragem temporal.

REGRA ANTI-LEAKAGE (obrigatória): É ESTRITAMENTE PROIBIDO incluir o anchor "{anchor}" — ou qualquer parte sua, incluindo anos, números, nomes específicos que sejam o próprio anchor — na formulação da pergunta. O anchor é a RESPOSTA ESPERADA, não um dado fornecido. A pergunta deve pedir o contexto de forma que o anchor seja a resposta natural, sem o revelar antecipadamente.

REGRA DE NEUTRALIDADE TEMPORAL (obrigatória): É ESTRITAMENTE PROIBIDO incluir datas concretas (anos, meses, dias) na formulação da pergunta — estas datas são precisamente a informação que o modelo avaliado tem de saber. Menciona o facto pelo nome ou descrição, nunca pela data.

DIVERSIDADE ESTILÍSTICA: É PROIBIDO começar por "Como se", "De que forma", "Quando", "Qual foi a data" ou "Em que ano". Varia a estrutura frásica. Usa formulações como as sugeridas abaixo:
{structures_block}

OUTRAS REGRAS:
- PROIBIDO usar pronomes interrogativos que induzam respostas de uma única palavra ("Quem...", "Qual é o nome...", "Onde...").
- PROIBIDO perguntas genéricas ou vagas (ex: "Qual foi o papel de {entity}?"). A pergunta deve ter foco factual claro.
- Foca-te nos factos concretos e objectivos do texto. NÃO faças perguntas interpretativas ou especulativas.
- CRIA UMA ÚNICA PERGUNTA SIMPLES — proibido ligar duas sub-questões com "e como", "e qual foi", ou "além de".
- A pergunta deve ser COMPLETAMENTE respondível pelo excerto fornecido.

Responde APENAS com a nova pergunta, sem mais texto.""", max_tokens=150, temperature=0.4)

async def gen_awareness_question(entity: str, original_q: str, anchor: str,
                                  distractor: str, distractor_relation: str, wiki: str) -> str:
    # Para awareness, a relação temporal distractor↔anchor é a espinha dorsal da pergunta
    all_structures = [
        '"Sabendo que [distractor] é parte da história de {entity}, situa cronologicamente [facto anchor] em relação a esse evento."',
        '"No contexto de [distractor] na vida de {entity}, em que fase e circunstâncias se enquadra [facto anchor]?"',
        '"Relacionando [distractor] com [facto anchor], reconstrói a sequência temporal desses dois momentos na trajetória de {entity}."',
        '"À luz de [distractor], descreve as condições em que [facto anchor] ocorreu no percurso de {entity}."',
        '"Dado que [distractor] marcou um ponto no percurso de {entity}, em que momento se deu [facto anchor]?"',
        '"Entre [distractor] e [facto anchor] na vida de {entity}, como se relacionam cronologicamente estes dois acontecimentos?"',
        '"Considerando [distractor] como um momento da trajetória de {entity}, descreve as circunstâncias em que ocorreu [facto anchor]."',
        '"Situa [facto anchor] em relação a [distractor] na cronologia da vida de {entity}."',
        '"Com [distractor] como ponto de referência, em que circunstâncias e fase surgiu [facto anchor] na trajetória de {entity}?"',
        '"Que relação temporal existe entre [distractor] e [facto anchor] na vida de {entity}?"',
        '"Conhecendo o papel de [distractor] na história de {entity}, descreve quando e em que contexto ocorreu [facto anchor]."',
        '"Perante o facto de [distractor] fazer parte do percurso de {entity}, situa e descreve [facto anchor] cronologicamente."',
    ]
    sampled = random.sample(all_structures, min(4, len(all_structures)))
    structures_block = "\n".join(f"    * {s}" for s in sampled)

    return await llm(f"""Estás a construir um benchmark académico de avaliação de LLMs em português europeu.

Tens esta pergunta fechada sobre a entidade "{entity}":
PERGUNTA ORIGINAL: {original_q}
RESPOSTA CURTA ESPERADA (anchor): {anchor}
FACTO DISTRACTOR (facto verdadeiro sobre a mesma entidade, de outro momento temporal): {distractor}

EXCERTO DA WIKIPEDIA (a tua pergunta deve ser completamente respondível com este texto):
{wiki}

Cria UMA ÚNICA pergunta aberta que siga TODAS as regras abaixo:

REGRA CENTRAL (obrigatória): A pergunta deve activar activamente a relação temporal entre o FACTO DISTRACTOR e o ANCHOR. Um modelo que não saiba qual dos dois factos ocorreu primeiro, ou em que fase da vida de {entity} cada um teve lugar, NÃO deve conseguir responder correctamente. O distractor não pode ser mero contexto decorativo — deve criar confusão temporal genuína que só se resolve com conhecimento cronológico.

REGRA DE NEUTRALIDADE TEMPORAL (obrigatória): É ESTRITAMENTE PROIBIDO incluir datas concretas (anos, meses, dias) na formulação da pergunta — estas datas são precisamente a informação que o modelo avaliado tem de saber. Menciona o facto pelo nome ou descrição, nunca pela data.

- Menciona o FACTO DISTRACTOR como referência temporal na pergunta.
- Pede para situar, descrever ou caracterizar o processo associado a "{anchor}" em relação a esse distractor.
- A pergunta deve ser COMPLETAMENTE respondível pelo excerto fornecido.
- Foca-te nos factos concretos e objectivos do texto. NÃO faças perguntas interpretativas ou especulativas.
- É OBRIGATÓRIO que a pergunta seja TEMPORALMENTE NEUTRA: NÃO uses palavras que revelem a ordem cronológica ("mais tarde", "posteriormente", "anteriormente", "antes de", "depois de", "após").
- DIVERSIDADE ESTILÍSTICA: É PROIBIDO começar por "Como se", "De que forma", "Quando", "Em que ano", "Tendo em conta", "Tendo X como", "Tendo o". Usa formulações variadas como as sugeridas abaixo:
{structures_block}
- PROIBIDO usar pronomes interrogativos que induzam respostas curtas ("Quem...", "Qual...", "Onde...").
- PROIBIDO: ligar duas sub-questões com "e". A pergunta deve ser estritamente uma única questão coesa.

Responde APENAS com a nova pergunta, sem mais texto.""", max_tokens=180, temperature=0.4)

async def gen_gold_answer(entity: str, question: str, anchor: str, wiki: str) -> str:
    return await llm(f"""Estás a criar a resposta de referência (gold answer) para um benchmark de LLMs.

Entidade: {entity}
Pergunta: {question}
Facto que DEVE estar coberto na resposta: {anchor}

Com base neste extracto da Wikipedia:
{wiki}

Escreve uma resposta de referência em português europeu que:
- Responda COMPLETAMENTE à pergunta (2-4 frases)
- Inclua obrigatoriamente o facto "{anchor}" na resposta, de forma natural e integrada
- Seja factualmente grounded no extracto acima — NÃO inventes datas ou factos
- Seja clara, bem redigida e substantiva (não apenas repita o anchor)
- NÃO faças inferências, suposições, hipóteses ou especulações (ex: evite "muito provavelmente", "sugere que", "reforça o enraizamento"). Limita-te aos factos explícitos no excerto.
- Ao relacionar os dois eventos, limita-te estritamente aos factos descritos. Não assumas relações de causa-efeito ou marcos profissionais a menos que estejam explicitamente escritos.
- É estritamente proibido incluir qualquer facto, data, nome, local ou detalhe que não esteja escrito no excerto.

CRÍTICO: Se o extracto NÃO contiver informação suficiente para responder à pergunta na totalidade,
escreve APENAS: INSUFFICIENT_CONTEXT
É PROIBIDO hedge parcial como "o extracto não fornece X mas..." — ou respondes completamente ou escreves INSUFFICIENT_CONTEXT.
Não inventes datas nem factos que não estejam no extracto.

Responde APENAS com a resposta de referência (ou INSUFFICIENT_CONTEXT).""",
        max_tokens=280, temperature=0.0)

# ── Job builders ───────────────────────────────────────────────────────────────
def build_cognition_jobs(data: dict) -> list:
    jobs = []
    for cat, entities in data.items():
        if not isinstance(entities, dict): continue
        for entity, content in entities.items():
            if not isinstance(content, dict): continue
            cqa = content.get("cognition_qa", content.get("temporal_qa", {}))
            for fact_desc, ev in cqa.items():
                if not isinstance(ev, dict): continue
                answers = ev.get("answers", [])
                anchor = answers[0] if answers else ev.get("ground_truth", "")
                qs = ev.get("questions", {})
                original_q = qs.get("generic") or next(iter(qs.values()), "") if qs else ""
                if not anchor or not original_q: continue
                jobs.append({
                    "task": "cognition_openended",
                    "entity": entity, "category": cat,
                    "original_question": original_q,
                    "anchor_fact": anchor,
                    "fact_description": fact_desc,
                    "distractor_context": None,
                    "source_url": ev.get("source_url", ""),
                })
    return jobs

def build_awareness_jobs(data: dict) -> list:
    jobs = []
    for cat, entities in data.items():
        if not isinstance(entities, dict): continue
        for entity, content in entities.items():
            if not isinstance(content, dict): continue
            qa_list = content.get("awareness_qa", [])
            if not qa_list:
                for _, ev in content.items():
                    if isinstance(ev, dict) and "awareness_qa" in ev:
                        qa_list.extend(ev["awareness_qa"])
            for qa in qa_list:
                past_f, future_f = qa.get("past_fact", ""), qa.get("future_fact", "")
                past_anchor = (qa.get("past_answers") or [qa.get("past_ground_truth", "")])[0]
                past_qs = qa.get("past_questions", {})
                past_q = past_qs.get("generic") or next(iter(past_qs.values()), "") if past_qs else ""
                if past_anchor and past_q and future_f:
                    jobs.append({
                        "task": "awareness_openended",
                        "entity": entity, "category": cat,
                        "original_question": past_q,
                        "anchor_fact": past_anchor,
                        "fact_description": past_f,
                        "distractor_context": future_f,
                        "distractor_relation": "posterior",
                        "source_url": qa.get("source_url", ""),
                    })
                future_anchor = (qa.get("future_answers") or [qa.get("future_ground_truth", "")])[0]
                future_qs = qa.get("future_questions", {})
                future_q = future_qs.get("generic") or next(iter(future_qs.values()), "") if future_qs else ""
                if future_anchor and future_q and past_f:
                    jobs.append({
                        "task": "awareness_openended",
                        "entity": entity, "category": cat,
                        "original_question": future_q,
                        "anchor_fact": future_anchor,
                        "fact_description": future_f,
                        "distractor_context": past_f,
                        "distractor_relation": "anterior",
                        "source_url": qa.get("source_url", ""),
                    })
    return jobs

# ── Checkpoint ─────────────────────────────────────────────────────────────────
def jid(job: dict) -> str:
    return hashlib.md5(f"{job['task']}|{job['entity']}|{job['anchor_fact']}".encode()).hexdigest()

def load_ckpt(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line); done[r["_id"]] = r
            except: pass
    return done

def append_record(path: Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ── Process ────────────────────────────────────────────────────────────────────
async def process(job: dict, done: dict, ckpt: Path, sem: asyncio.Semaphore):
    _id = jid(job)
    if _id in done: return
    async with sem:
        _stats["total"] += 1
        entity    = job["entity"]
        anchor    = job["anchor_fact"]
        orig_q    = job["original_question"]
        distractor = job.get("distractor_context") or ""
        distractor_rel = job.get("distractor_relation") or ""

        wiki = load_wiki(entity)
        wiki_para = pick_wiki_paragraph(wiki, anchor, distractor)

        # Validação 1: excerto útil?
        if not wiki_para or not is_useful_paragraph(wiki_para, anchor):
            log.debug(f"[SKIP wiki] {entity} | anchor={anchor}")
            _stats["skipped_wiki"] += 1
            return

        # Gera pergunta com contexto wiki explícito
        if distractor:
            question = await gen_awareness_question(entity, orig_q, anchor, distractor, distractor_rel, wiki_para)
        else:
            question = await gen_cognition_question(entity, orig_q, anchor, wiki_para)
        if not question: return

        # Gera gold answer SEM truncagem do wiki
        gold = await gen_gold_answer(entity, question, anchor, wiki_para)

        # Validação 2: o LLM admitiu não ter contexto?
        if not gold or "INSUFFICIENT_CONTEXT" in gold:
            log.debug(f"[SKIP insufficient] {entity} | anchor={anchor}")
            _stats["skipped_insufficient"] += 1
            return

        # Validação 3: gold tem marcadores de contexto insuficiente parciais?
        if gold_has_insuff_markers(gold):
            log.debug(f"[SKIP insuff_gold] {entity} | anchor={anchor} | gold={gold[:80]}")
            _stats["skipped_insuff_gold"] += 1
            return

        # Validação 4: o anchor aparece na gold answer?
        if not anchor_in_gold(anchor, gold):
            log.debug(f"[SKIP anchor mismatch] {entity} | anchor={anchor} | gold={gold[:80]}")
            _stats["skipped_anchor_mismatch"] += 1
            return

        rec = {
            "_id": _id,
            "task": job["task"],
            "entity": entity,
            "category": job["category"],
            "original_question": orig_q,
            "question": question,
            "anchor_fact": anchor,
            "distractor_context": distractor or None,
            "fact_description": job["fact_description"],
            "gold_answer": gold,
            "wiki_excerpt": wiki_para,
            "source_url": job.get("source_url", ""),
        }
        append_record(ckpt, rec)
        done[_id] = rec
        _stats["saved"] += 1

async def run(name: str, jobs: list):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"{name}.jsonl"
    out  = OUT_DIR  / f"{name}.jsonl"
    done = load_ckpt(ckpt)
    remaining = [j for j in jobs if jid(j) not in done]
    log.info(f"[{name}] {len(jobs)} total | {len(done)} done | {len(remaining)} remaining")
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    saved_before = _stats["saved"]
    for i in range(0, len(remaining), 500):
        batch = remaining[i:i+500]
        await asyncio.gather(*[process(j, done, ckpt, sem) for j in batch])
        saved_now = _stats["saved"] - saved_before
        log.info(f"[{name}] {min(i+500, len(remaining))}/{len(remaining)} processados | {saved_now} guardados até agora")

    with open(out, "w", encoding="utf-8") as f:
        for rec in done.values():
            f.write(json.dumps({k: v for k, v in rec.items() if k != "_id"},
                               ensure_ascii=False) + "\n")
    log.info(f"[{name}] -> {out} ({len(done)} records)")

async def main():
    build_clients()
    log.info(f"{len(CLIENTS)} endpoint(s): {VLLM_ENDPOINTS or f'localhost:{PORT}'}")

    cog_data = json.loads((SPLIT_DIR / "saudade_cognition_pt_test.json").read_text("utf-8"))
    cog_jobs = build_cognition_jobs(cog_data)
    if N_TEST: cog_jobs = cog_jobs[:N_TEST]
    log.info(f"Cognition jobs: {len(cog_jobs)}")

    aw_data  = json.loads((SPLIT_DIR / "saudade_awareness_pt_test.json").read_text("utf-8"))
    aw_jobs  = build_awareness_jobs(aw_data)
    if N_TEST: aw_jobs = aw_jobs[:N_TEST]
    log.info(f"Awareness jobs: {len(aw_jobs)}")

    await run("saudade_cognition_openended_test", cog_jobs)
    await run("saudade_awareness_openended_test", aw_jobs)

    log.info("=" * 60)
    log.info("QUALITY STATS (gerador):")
    log.info(f"  Total processados  : {_stats['total']}")
    log.info(f"  Guardados (pass)   : {_stats['saved']} ({100*_stats['saved']/max(_stats['total'],1):.1f}%)")
    log.info(f"  Skip wiki ruim     : {_stats['skipped_wiki']}")
    log.info(f"  Skip INSUFFICIENT  : {_stats['skipped_insufficient']}")
    log.info(f"  Skip insuff gold   : {_stats['skipped_insuff_gold']}")
    log.info(f"  Skip anchor miss   : {_stats['skipped_anchor_mismatch']}")
    log.info("=" * 60)
    log.info("Done.")

if __name__ == "__main__":
    asyncio.run(main())
