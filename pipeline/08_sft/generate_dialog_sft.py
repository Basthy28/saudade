"""
generate_dialog_sft.py
=========================
Staged pipeline: TISER and casual turns generated separately with appropriate context.

Phases:
  [1] Build jobs (entity + qa_pairs chunks, persona, wiki)
  [2] Generate TISER turns (qa_pairs + persona → TISER-structured Q&A)
  [3] Judge TISER turns (B1+B2 vs qa_pairs, per-turn rejection)
  [4] Generate casual turns (wiki + accepted TISER context + persona)
  [5] Judge casual turns (B1+B2 vs Wikipedia, per-turn rejection)
  [6] Interleave + write output

Concurrency: asyncio.Queue producer-consumer (bounded, back-pressure).
"""

import os, json, random, logging, re, time, asyncio, dataclasses, signal, hashlib
from pathlib import Path
import sys

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

sys.path.insert(0, str(Path(__file__).parent))
from sft_judge import judge_tiser_turns, judge_casual_turns, run_parallel

# ── Graceful stop (SIGUSR1 from sbatch before timeout) ───────────────────────
STOP_REQUESTED = False
def _request_stop(signum, frame):
    global STOP_REQUESTED
    log.warning("[SIGNAL] SIGUSR1 received — will stop after current batch and checkpoint.")
    STOP_REQUESTED = True
signal.signal(signal.SIGUSR1, _request_stop)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID         = os.environ.get("GEN_MODEL_ID", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
JUDGE_MODEL_ID   = os.environ.get("JUDGE_MODEL_ID", "RedHatAI/gemma-4-31B-it-FP8-block")
PORT             = 8088
MAX_CONCURRENT   = int(os.environ.get("MAX_CONCURRENT", "64"))
MIN_TISER_TURNS  = 2
JUDGE_WORKERS    = int(os.environ.get("JUDGE_WORKERS", "64"))
MAX_RETRIES      = 1     # max retry attempts per failed generation turn
N_TEST           = int(os.environ.get("N_TEST", "0"))
RANDOM_SEED      = 42
CATEGORY_FILTER  = os.environ.get("CATEGORY_FILTER", "")
DATASET_FILTER   = os.environ.get("DATASET_FILTER", "")   # "awareness", "cognition", "reasoning" or "" for all
RUN_PHASE        = os.environ.get("RUN_PHASE", 0)
if isinstance(RUN_PHASE, str) and RUN_PHASE.isdigit(): RUN_PHASE = int(RUN_PHASE)
RUN_PHASE_NUM    = 3.5 if RUN_PHASE == "3b" else (4.5 if RUN_PHASE == "4b" else float(RUN_PHASE))
RESUME_FROM      = os.environ.get("RESUME_FROM", 0)
if isinstance(RESUME_FROM, str) and RESUME_FROM.isdigit(): RESUME_FROM = int(RESUME_FROM)

# Data directories
_DATA_BASE = DATA / "06_split"
_train  = _DATA_BASE / "saudade_cognition_pt_train.json"
DATA_DIR    = _DATA_BASE   # kept for compatibility
USE_PREPROCESSED_SPLIT = _train.exists()   # auto-detected at startup
OUTPUT_DIR  = Path(os.environ.get("OUTPUT_DIR", DATA / "08_sft" / "output"))
WIKI_DIR    = DATA / "01_extraction" / "cats"
WIKI_DIR_FALLBACKS = [
    DATA / "01_extraction" / "newdata" / "temporal_relevant",
]
CHECKPT_DIR = Path(os.environ.get("CHECKPT_DIR", DATA / "08_sft" / "checkpoints"))

# ── Personas ──────────────────────────────────────────────────────────────────
PERSONAS = [
    ("curioso",
     "Muito curioso e entusiasta. Faz perguntas de seguimento encadeadas, "
     "gosta de cruzar eventos. Não se contenta com a primeira resposta.",
     5, 9),
    ("cetico",
     "Cético e questionador. Frequentemente inclui eventos alternativos na "
     "própria pergunta. Cria distractores temporais de forma natural.",
     4, 7),
    ("jornalista",
     "Jornalista rigoroso. Vai directo ao ponto, pede datas e factos concretos. "
     "Faz perguntas curtas e precisas.",
     3, 6),
    ("estudante",
     "Estudante a preparar-se para um exame. Quer confirmar datas e sequências. "
     "Por vezes mistura factos de diferentes épocas.",
     4, 8),
    ("historiador",
     "Investigador com tom académico. Cruza eventos cronológicos complexos, "
     "usa linguagem precisa.",
     5, 9),
    ("leigo",
     "Pessoa comum, casual e informal. Tem interesse no tema mas confunde "
     "facilmente datas. Faz perguntas simples mas por vezes com pressupostos errados.",
     3, 6),
    ("entrevistador",
     "Entrevistador profissional. Começa com perguntas abertas e gerais, "
     "depois aprofunda com sub-perguntas temporais específicas.",
     4, 8),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _job_hash(job: dict) -> str:
    """Generate a unique fingerprint for a dialog job based on its QA pairs."""
    fingerprint = "".join(p.get("question", "") + p.get("answer", "") for p in job.get("pairs", []))
    return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()

def hms(s: float) -> str:
    m, sec = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h:02d}h{m:02d}m{sec:02d}s"

def _ckpt(name: str) -> Path:
    """Return checkpoint path for a given phase name."""
    CHECKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{'_'.join(filter(None,[DATASET_FILTER,CATEGORY_FILTER]))}_" if (DATASET_FILTER or CATEGORY_FILTER) else ""
    return CHECKPT_DIR / f"{tag}{name}.jsonl"

def _auto_resume_from() -> int:
    """Detect which phase to resume from based on existing checkpoint files.
    Phase 2 and 4 can be partially done — detected separately."""
    # Full-phase checkpoints (past the generation phases)
    for resume_from, ckpt_name in [
        (6, "p5_casual_judge"),
        ("4b", "p4b_coref"),
        (5, "p4_casual_gen"),
        ("3b", "p3b_tiser_retry"),
        (4, "p3_tiser_judge"),
    ]:
        if _ckpt(ckpt_name).with_suffix(".done").exists():
            log.info(f"  [AUTO-RESUME] Detected '{ckpt_name}.done' -> resuming from phase {resume_from}")
            return resume_from
    # Phase 2 partial or full checkpoint
    if _ckpt("p2_tiser_gen").with_suffix(".done").exists():
        log.info("  [AUTO-RESUME] Detected 'p2_tiser_gen.done' -> resuming phase 3")
        return 3
    if _ckpt("p2_tiser_gen").exists():
        log.info("  [AUTO-RESUME] Detected 'p2_tiser_gen' (partial) -> resuming phase 2")
        return 2
    log.info("  [AUTO-RESUME] No checkpoints found -> starting from phase 1")
    return 1

# Resolve RESUME_FROM=0 -> auto-detect from checkpoints
if RESUME_FROM == 0:
    RESUME_FROM = _auto_resume_from()
RESUME_FROM_NUM = 3.5 if RESUME_FROM == "3b" else (4.5 if RESUME_FROM == "4b" else float(RESUME_FROM))
def _json_default(obj):
    """Custom JSON serializer — converts dataclasses to dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def save_checkpoint(name: str, jobs: list):
    path = _ckpt(name)
    with open(path, "w", encoding="utf-8") as f:
        for job in jobs:
            safe = {k: v for k, v in job.items() if not callable(v)}
            f.write(json.dumps(safe, ensure_ascii=False, default=_json_default) + "\n")
    log.info(f"  [CKPT] Saved {len(jobs)} records → {path}")

def load_checkpoint(name: str) -> list | None:
    path = _ckpt(name)
    if not path.exists():
        return None
    jobs = []
    import gc
    gc.disable()
    
    try:
        import orjson
        parse_json = orjson.loads
    except ImportError:
        parse_json = json.loads
        
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    jobs.append(parse_json(line))
    finally:
        gc.enable()
    log.info(f"  [CKPT] Loaded {len(jobs)} records ← {path}")
    return jobs or None

def build_qa_block(pairs: list) -> str:
    """Format qa_pairs for the generation prompt."""
    lines = []
    for p in pairs:
        entry = f"  - Q: {p['question']}\n    A: {p['answer']}"
        if p.get("distractor_context"):
            # Label as 'contexto adicional' — avoids leaking 'âncora temporal' jargon into the model
            entry += f"\n    [contexto adicional sobre a entidade: {p['distractor_context']}]"
        lines.append(entry)
    return "\n".join(lines)

def _strip_wikitext(raw: str) -> str:
    """Strip wiki markup from raw wikitext to get readable prose."""
    import re as _re
    # Remove templates {{...}} (non-greedy, handles nesting partially)
    text = _re.sub(r'\{\{[^{}]*\}\}', '', raw)
    text = _re.sub(r'\{\{[^{}]*\}\}', '', text)  # second pass for nested
    # Convert [[link|display]] → display, [[link]] → link
    text = _re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)
    # Remove external links [http://... text] → text
    text = _re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    # Remove remaining URLs
    text = _re.sub(r'https?://\S+', '', text)
    # Remove table markup lines
    text = _re.sub(r'^[\|!].*$', '', text, flags=_re.MULTILINE)
    # Remove wiki formatting (bold/italic)
    text = _re.sub(r"'{2,3}", '', text)
    # Remove HTML tags
    text = _re.sub(r'<[^>]+>', '', text)
    # Remove ref tags content
    text = _re.sub(r'<ref[^/]*/>', '', text)
    # Collapse whitespace
    text = _re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _parse_wiki_json(path: Path, max_words: int) -> str:
    """Parse either entity_data.json (with 'sections') or flat temporal_cleaned .json."""
    try:
        data = json.load(open(path, encoding="utf-8"))
        # Format A: {"sections": [{"heading": ..., "content": ...}]}  (temporal_relevant)
        if "sections" in data:
            blocks = []
            for s in data.get("sections", []):
                h = s.get("heading", "")
                c = s.get("content", [])
                if isinstance(c, list): c = "\n".join(c)
                t = f"{h}\n{c}".strip() if h else c.strip()
                if t: blocks.append(t)
            full = "\n\n".join(blocks)
        else:
            parts = []
            # Try wikitext field first — strip markup to get prose
            wikitext = data.get("wikitext", "")
            if wikitext and len(wikitext) > 100:
                stripped = _strip_wikitext(wikitext)
                if len(stripped.split()) > 30:
                    parts.append(stripped)
            # Fallback: flat dict with text fields or lists of dicts (temporal_cleaned)
            if not parts:
                for k, v in data.items():
                    if k in ("wikitext", "url", "title", "language"): continue
                    if isinstance(v, str) and len(v) > 20:
                        parts.append(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and len(item) > 20:
                                parts.append(item)
                            elif isinstance(item, dict):
                                fact = item.get("facto") or item.get("content")
                                if fact:
                                    parts.append(str(fact))
            full = "\n\n".join(parts)
        words = full.split()
        return (" ".join(words[:max_words]) + " [...]") if len(words) > max_words else full
    except Exception:
        return ""

WIKI_PATHS_CACHE = None

def _build_wiki_cache():
    global WIKI_PATHS_CACHE
    if WIKI_PATHS_CACHE is not None: return
    WIKI_PATHS_CACHE = {}
    for wiki_base in [WIKI_DIR] + WIKI_DIR_FALLBACKS:
        if not wiki_base.exists(): continue
        for cat_dir in wiki_base.iterdir():
            if not cat_dir.is_dir(): continue
            for item in cat_dir.iterdir():
                if item.is_dir():
                    # Style A: cat/slug/entity_data.json
                    slug = item.name
                    path_a = item / "entity_data.json"
                    if slug not in WIKI_PATHS_CACHE: # We assume it exists if dir exists, to save stat calls
                        WIKI_PATHS_CACHE[slug] = path_a
                elif item.is_file() and item.name.endswith(".json"):
                    # Style B: cat/slug.json
                    slug = item.stem
                    if slug not in WIKI_PATHS_CACHE:
                        WIKI_PATHS_CACHE[slug] = item

def load_wiki_text(entity: str, max_words: int = 2000) -> str:
    _build_wiki_cache()
    slug = entity.replace(" ", "_")
    path = WIKI_PATHS_CACHE.get(slug)
    if path and path.exists():
        result = _parse_wiki_json(path, max_words)
        if result: return result
    return ""

def collect_qa_items(data: dict, dataset_type: str) -> dict:
    entity_qa = {}
    BLACKLIST = {"Festival RTP: A Melhor Canção de Sempre", "Toquinho"}
    for _, entities in data.items():
        if not isinstance(entities, dict): continue
        for entity, content in entities.items():
            if not isinstance(content, dict) or entity in BLACKLIST:
                continue
            pairs = []
            if dataset_type == "awareness":
                qa_list = content.get("awareness_qa", [])
                if not qa_list:
                    for _, ev in content.items():
                        if isinstance(ev, dict) and "awareness_qa" in ev:
                            qa_list.extend(ev["awareness_qa"])
                for qa in qa_list:
                    future_f = qa.get("future_fact", "")
                    past_f   = qa.get("past_fact", "")
                    for _, q in qa.get("past_questions", {}).items():
                        if q:
                            a = (qa.get("past_answers") or [qa.get("past_ground_truth", "")])[0]
                            pairs.append({"question": q, "answer": a, "distractor_context": future_f})
                    for _, q in qa.get("future_questions", {}).items():
                        if q:
                            a = (qa.get("future_answers") or [qa.get("future_ground_truth", "")])[0]
                            pairs.append({"question": q, "answer": a, "distractor_context": past_f})
            elif dataset_type == "cognition":
                qa_dict = content.get("cognition_qa", content.get("temporal_qa", {}))
                for _, ev in qa_dict.items():
                    if not isinstance(ev, dict): continue
                    questions = ev.get("questions") or ev.get("question")
                    answers   = ev.get("answers", [])
                    if questions:
                        if isinstance(questions, str): questions = {"g": questions}
                        # Collect ALL paraphrases — dedup happens per-chunk in chunk_pairs
                        for _, q in questions.items():
                            if q: pairs.append({"question": q, "answer": answers[0] if answers else ""})
            elif dataset_type == "reasoning":
                outer_rqa = content.get("reasoning_qa", content.get("ranking_qa", []))
                for pair_item in outer_rqa:
                    inner_rqa = pair_item.get("ranking_qa", {})
                    answers = pair_item.get("expected_answers", pair_item.get("answers", []))
                    for _, q in inner_rqa.items():
                        if q: pairs.append({"question": q, "answer": answers[0] if answers else ""})
            if pairs:
                entity_qa[entity] = pairs
    return entity_qa

def _dedup_chunk(pairs: list) -> list:
    """Keep at most one pair per unique (answer, distractor_context) within a single chunk.
    This lets paraphrases of the same fact be spread across different persona chunks."""
    seen, deduped = set(), []
    for p in pairs:
        key = (p["answer"].strip().lower(), p.get("distractor_context", "").strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped

def chunk_pairs(pairs: list, rng: random.Random) -> list:
    """V5 strategy (kept for reference): divides pairs sequentially across personas.
    Each chunk gets one persona. Results in ~N/max_t dialogues per entity total."""
    shuffled = pairs.copy(); rng.shuffle(shuffled)
    chunks, pidx, i = [], 0, 0
    while i < len(shuffled):
        pid, ptxt, min_t, max_t = PERSONAS[pidx % len(PERSONAS)]
        raw_chunk = shuffled[i:i + max_t]
        chunk = _dedup_chunk(raw_chunk)
        n_tiser = len(chunk)
        target_total = rng.randint(min_t, max_t)
        n_casual = max(1, target_total - n_tiser)
        chunks.append({
            "pairs": chunk,
            "persona_id": pid,
            "persona_text": ptxt,
            "n_tiser": n_tiser,
            "n_casual": n_casual,
            "target_total": n_tiser + n_casual,
        })
        i += max_t; pidx += 1
    if len(chunks) >= 2 and len(chunks[-1]["pairs"]) < 2:
        prev = chunks[-2]
        prev["pairs"]   = prev["pairs"] + _dedup_chunk(chunks[-1]["pairs"])
        prev["n_tiser"] = len(prev["pairs"])
        prev["n_casual"] = max(1, prev["n_casual"])
        chunks = chunks[:-1]
    return [c for c in chunks if len(c["pairs"]) >= 2]

def build_jobs_per_persona(pairs: list, rng: random.Random) -> list:
    """V6 strategy: generate a full dialogue for EACH of the 7 personas using ALL pairs.

    For each persona, all QA pairs are given (independently shuffled), then sub-chunked
    into dialogues of max_t size if needed. This multiplies data by ~len(PERSONAS) vs V5.

    Benefits:
      - Same factual knowledge expressed in 7 distinct conversational styles.
      - Model learns to answer the same question very differently based on user persona.
      - ~7x more training examples with genuine stylistic diversity (not just paraphrase).
    """
    jobs = []
    for pid, ptxt, min_t, max_t in PERSONAS:
        # Independent shuffle per persona → different turn order per dialogue
        shuffled = pairs.copy()
        rng.shuffle(shuffled)

        persona_chunks = []
        i = 0
        while i < len(shuffled):
            raw_chunk = shuffled[i:i + max_t]
            chunk = _dedup_chunk(raw_chunk)
            if len(chunk) >= 2:
                n_tiser = len(chunk)
                target_total = rng.randint(min_t, max_t)
                n_casual = max(1, target_total - n_tiser)
                persona_chunks.append({
                    "pairs": chunk,
                    "persona_id": pid,
                    "persona_text": ptxt,
                    "n_tiser": n_tiser,
                    "n_casual": n_casual,
                    "target_total": n_tiser + n_casual,
                })
            i += max_t

        # Merge a tiny trailing chunk (< 2 pairs) into the previous one
        if len(persona_chunks) >= 2 and len(persona_chunks[-1]["pairs"]) < 2:
            prev = persona_chunks[-2]
            prev["pairs"]    = prev["pairs"] + _dedup_chunk(persona_chunks[-1]["pairs"])
            prev["n_tiser"]  = len(prev["pairs"])
            prev["n_casual"] = max(1, prev["n_casual"])
            persona_chunks   = persona_chunks[:-1]

        jobs.extend(persona_chunks)

    return [c for c in jobs if len(c["pairs"]) >= 2]

# ── Prompt templates ──────────────────────────────────────────────────────────

# Cognition: distractor is a wrong date/value in the human's question
TISER_FEW_SHOT_COGNITION = """\
EXEMPLO (2 turnos, persona "leigo"):
{
  "turns": [
    {
      "human": "Ei, e o Sporting, quando ganhou o último campeonato antes de 2000? Não foi tipo em 2001?",
      "gpt": "<reasoning>\\nO utilizador perguntou sobre o último campeonato ganho antes de 2000, mas sugeriu 2001. Vou organizar as datas.\\n<timeline>\\nEm 1999-2000, o Sporting CP venceu a Primeira Liga. Em 2001, foi a vez do Boavista.\\n</timeline>\\n<reflection>\\nA sugestão de 2001 está errada porque é posterior a 2000. O último campeonato do Sporting antes de 2000 foi mesmo na época 1999-2000.\\n</reflection>\\n</reasoning>\\n<answer>\\n1999-2000\\n</answer>"
    },
    {
      "human": "Ah ok. E onde é que fica o estádio deles?",
      "gpt": "O Estádio José Alvalade fica em Lisboa."
    }
  ]
}
"""

# Awareness: distractor is a VALID fact about the entity, but not the answer.
# The human mentions it as temporal context — AMALIA identifies it in the reflection.
TISER_FEW_SHOT_AWARENESS = """\
EXEMPLOS (3 estruturas diferentes de pergunta — varia sempre a forma):

EXEMPLO 1 (persona "curiosa" — comparação directa):
{
  "turns": [
    {
      "human": "E o campeonato do Sporting, isso foi antes ou depois da final da Taça UEFA de 2005?",
      "gpt": "<reasoning>\\nO utilizador quer saber a ordem entre dois eventos: o título de campeonato e a final europeia de 2005.\\n<timeline>\\nEm 2001-02, o Sporting CP venceu a Primeira Liga. Em 2005, perderam a final da Taça UEFA para o CSKA Moscovo.\\n</timeline>\\n<reflection>\\nO campeonato (2001-02) veio antes da final europeia (2005). A ordem está clara.\\n</reflection>\\n</reasoning>\\n<answer>\\nFoi antes — o campeonato foi na época 2001-02, três anos antes da final da Taça UEFA.\\n</answer>"
    },
    {
      "human": "E quem marcou o golo decisivo nessa final europeia?",
      "gpt": "O CSKA Moscovo ganhou 3-1, com Vágner Love a ser o grande destaque da equipa russa."
    }
  ]
}

EXEMPLO 2 (persona "cético" — sobreposição temporal):
{
  "turns": [
    {
      "human": "Espera — o Sporting ganhou o campeonato enquanto ainda estava a disputar a Taça UEFA de 2005?",
      "gpt": "<reasoning>\\nO utilizador está a tentar perceber se os dois eventos se sobrepuseram no tempo.\\n<timeline>\\nA época 2001-02 foi quando o Sporting venceu a liga. A final da Taça UEFA foi em Maio de 2005, época 2004-05.\\n</timeline>\\n<reflection>\\nNão houve sobreposição — são épocas completamente diferentes. O título aconteceu três épocas antes da final europeia.\\n</reflection>\\n</reasoning>\\n<answer>\\nNão, foram épocas separadas. O título foi em 2001-02, a final da Taça UEFA foi em 2004-05 — três anos de diferença.\\n</answer>"
    },
    {
      "human": "Ah ok, e qual foi o adversário na final europeia?",
      "gpt": "O CSKA Moscovo. Ganharam 3-1 em Lisboa."
    }
  ]
}

EXEMPLO 3 (persona "leigo" — pergunta informal sobre intervalo):
{
  "turns": [
    {
      "human": "E esse campeonato do Sporting, foi muito tempo depois da final da Taça UEFA de 2005, ou foi logo a seguir?",
      "gpt": "<reasoning>\\nO utilizador quer saber o intervalo entre dois eventos, mas parece confuso com a ordem.\\n<timeline>\\nO Sporting ganhou a liga em 2001-02. A final da Taça UEFA foi em 2005.\\n</timeline>\\n<reflection>\\nO utilizador inverteu a ordem — o campeonato foi ANTES da final europeia, não depois. Convém corrigir gentilmente.\\n</reflection>\\n</reasoning>\\n<answer>\\nFoi antes, não depois! O campeonato foi em 2001-02, e a final europeia só veio em 2005 — três anos mais tarde.\\n</answer>"
    },
    {
      "human": "Ah pois, enganei-me! E o treinador nessa altura do título era quem?",
      "gpt": "Era o Laszlo Bölöni, um treinador romeno que fez uma época fantástica no Sporting."
    }
  ]
}
"""

TISER_PROMPT = """\
Você é AMALIA, uma assistente de IA especializada em raciocínio temporal em português europeu.

Persona do utilizador: {persona}
Entidade central: {entity}

Conhecimento base para as respostas:
{qa_block}

Gera exactamente {n_tiser} pares de pergunta-resposta sobre {entity}.
Cada par = 1 item de conhecimento → 1 turno (na mesma ordem dos itens acima).

REGRAS OBRIGATÓRIAS:
1. A pergunta (human) deve ser natural, no estilo da persona.{distractor_rule}
2. **USO CONDICIONAL DO TISER:** Se a pergunta exigir raciocínio temporal, cruzar múltiplos eventos, ou se o utilizador sugerir uma data/facto errado, a resposta DEVE usar a estrutura TISER:
   <reasoning>
   [raciocínio interno e natural sobre o problema]
   <timeline>
   [cronologia natural dos eventos relevantes]
   </timeline>
   <reflection>
   [verificação interna para garantir que a resposta faz sentido]
   </reflection>
   </reasoning>
   <answer>
   [resposta directa final]
   </answer>
   *Nota: Se a pergunta for simples e directa (ex: "Onde nasceu?"), NÃO uses tags TISER. Responde apenas directamente no formato normal.*
3. **PROIBIÇÃO DE METALINGUAGEM:** NUNCA uses palavras como "distractor", "ratoeira", "contexto adicional", "facto seed", "texto base", "âncora", "ponto de referência temporal", ou qualquer expressão que descreva a mecânica do prompt. O teu <reasoning> deve soar a pensamento histórico natural, nunca a análise de IA. Em vez de "o utilizador usou um evento X como referência", pensa apenas "X aconteceu em tal data, portanto..."
4. **PERGUNTAS PLAUSÍVEIS:** As perguntas do utilizador devem ser naturais e plausíveis para um humano real. São ESTRITAMENTE PROIBIDOS os seguintes padrões:
   a) **Pós-morte:** NUNCA uses "depois de ter morrido", "após a sua morte", "depois de falecer" como âncora temporal. Ex. PROIBIDO: "Depois de ter morrido em 1895, quando esteve em Paris?" (impossível)
   b) **Contradição cronológica:** NUNCA digas "antes de X" se a resposta é cronologicamente posterior a X. Ex. PROIBIDO: "Antes de ser ministro em 1915, quando foi promovido a general em 1918?" (1918 é depois de 1915)
   c) **Tautologia:** NUNCA incluas a resposta na própria pergunta. Ex. PROIBIDO: "Antes de morrer, quando é que X começou a carreira em 1882?" (1882 já está na pergunta)
   O contexto temporal (distractor) deve ser um evento DIFERENTE do facto-alvo e a ordem cronológica da pergunta deve ser sempre coerente com os factos reais.
5. **SEM REPETIÇÕES:** NUNCA repitas a mesma pergunta ou a mesma resposta que já apareceu num turno anterior. Cada turno deve cobrir um facto ou ângulo diferente.
6. NÃO inventes factos para além do conhecimento base partilhado.
7. Responde SEMPRE em português europeu.
8. Output: JSON com exactamente este formato:
   {{"turns": [{{"human": "...", "gpt": "..."}}, ...]}}

{few_shot}

Output JSON:
"""

CASUAL_FEW_SHOT = """\
EXEMPLO (2 turnos, persona "curioso"):
{
  "turns": [
    {
      "human": "E como é que os portugueses viam esta pessoa? Era popular?",
      "gpt": "Era muito admirada pelo povo — tornou-se num símbolo nacional ao longo da sua carreira."
    },
    {
      "human": "E teve algum reconhecimento internacional?",
      "gpt": "Sim, actuou em vários países europeus e foi reconhecida por públicos muito diferentes."
    }
  ]
}
"""

# ── First-turn openers per persona (sampled randomly each call to force diversity) ──
CASUAL_OPENERS = {
    "curioso": [
        "Queria saber mais sobre {entity} — li que ...",
        "Estou a investigar {entity} e fiquei com uma dúvida:",
        "Vi uma referência a {entity} e não percebi bem...",
        "Descobri {entity} há pouco tempo e tenho curiosidade:",
        "Fiquei intrigado com {entity} — podes explicar-me",
        "{entity} apareceu-me numa pesquisa e queria perceber melhor:",
        "Tenho andado a ler sobre {entity} e surgiu-me uma dúvida:",
        "Estava a explorar o tema de {entity} e não entendo",
    ],
    "cetico": [
        "Ouvi dizer que {entity} ..., mas não tenho a certeza se é verdade.",
        "Há uma coisa sobre {entity} que me parece estranha —",
        "Li que {entity} ..., mas parece-me exagerado. Confirmas?",
        "Alguém me disse que {entity} ..., mas isso não bate certo com o que sei.",
        "Tenho dúvidas sobre {entity} — especificamente:",
        "Questionei-me sobre {entity} e não consigo perceber se",
        "Discordei de algo que li sobre {entity}:",
        "Estive a analisar {entity} e há algo que não me convence —",
    ],
    "jornalista": [
        "{entity} — podes dar-me os factos essenciais?",
        "Para um artigo sobre {entity}:",
        "{entity} — preciso de dados concretos para uma peça.",
        "Estou a trabalhar numa reportagem sobre {entity}:",
        "{entity}: quais são os pontos-chave que não posso ignorar?",
        "A redação quer um perfil de {entity} — começa pelos factos principais.",
        "Para contextualizar {entity} num artigo, preciso que expliques:",
        "{entity} — confirma-me os números e datas essenciais.",
    ],
    "estudante": [
        "Estou a estudar {entity} para um trabalho —",
        "Precisava de perceber melhor {entity}:",
        "Tenho um exame sobre {entity} e queria confirmar:",
        "{entity} apareceu no programa e não percebi bem:",
        "O meu professor mencionou {entity} e fiquei com dúvidas:",
        "Para a minha apresentação sobre {entity}, preciso de esclarecer:",
        "Andei a pesquisar {entity} para a aula e surgiu-me uma questão:",
        "Estou a fazer uma ficha sobre {entity} —",
    ],
    "historiador": [
        "Ao analisar {entity}, surge uma questão interessante:",
        "Do ponto de vista histórico, {entity} levanta uma questão:",
        "Estou a investigar {entity} no contexto mais amplo do período —",
        "A historiografia de {entity} apresenta algumas tensões que me intrigam:",
        "No âmbito da minha investigação sobre {entity}, deparei-me com:",
        "{entity} constitui um caso de estudo relevante — especificamente:",
        "Cruzando fontes sobre {entity}, surgiu-me a seguinte questão:",
        "Do ponto de vista cronológico, {entity} levanta uma série de questões:",
    ],
    "leigo": [
        "Alguém me falou de {entity} e agora tenho curiosidade —",
        "Nunca ouvi muito sobre {entity}, mas parece interessante.",
        "Ouvi falar de {entity} mas não sei bem o que é —",
        "{entity} apareceu-me no telemóvel e queria perceber:",
        "Estava a conversar com um amigo sobre {entity} e fiquei curioso —",
        "Nunca me preocupei muito com {entity} mas ultimamente ouço falar —",
        "Um familiar falou-me de {entity} e queria saber mais:",
        "Vi algo sobre {entity} nas notícias e não percebi bem:",
    ],
    "entrevistador": [
        "Vamos falar de {entity} — para começar:",
        "O nosso tema hoje é {entity}.",
        "Hoje quero explorar {entity} — começa por me dar uma visão geral.",
        "{entity} é o nosso foco desta conversa — por onde quer que comecemos?",
        "Para introduzir {entity} ao nosso público, o que é mais relevante?",
        "Vamos centrar-nos em {entity} — qual é o aspeto mais marcante?",
        "Estamos aqui para falar de {entity} — que ponto de partida sugere?",
        "{entity} merece uma conversa aprofundada — por onde começamos?",
    ],
}

def _sample_openers(persona_text: str, entity: str, n: int = 3) -> str:
    """Return n randomly sampled first-turn openers for the given persona, formatted as a list."""
    key_map = {
        "curioso": "curioso", "cético": "cetico", "cetico": "cetico",
        "jornalista": "jornalista", "estudante": "estudante",
        "historiador": "historiador", "investigador": "historiador",
        "leigo": "leigo", "pessoa": "leigo",
        "entrevistador": "entrevistador",
    }
    matched_key = "leigo"
    persona_lower = persona_text.lower()
    for k, v in key_map.items():
        if k in persona_lower:
            matched_key = v
            break
    openers = CASUAL_OPENERS.get(matched_key, CASUAL_OPENERS["leigo"])
    sampled = random.sample(openers, min(n, len(openers)))
    return "\n".join(f'       * "{s.format(entity=entity)}"' for s in sampled)

CASUAL_PROMPT = """\
Você é AMALIA, uma assistente de IA em português europeu.

Persona do utilizador: {persona}
Entidade central: {entity}

Esta conversa já contém os seguintes turnos de raciocínio temporal:
{tiser_context}

Texto Wikipedia sobre {entity} (referência factual — baseia as respostas aqui):
{wiki_excerpt}

Gera exactamente {n_casual} turnos de conversa casual sobre {entity}
que complementem naturalmente os turnos temporais acima.

REGRAS:
1. NÃO uses estrutura TISER (<reasoning>, <timeline>, <reflection>, <answer>).
2. As PERGUNTAS (human) devem reflectir o ESTILO da persona — a forma como
   a persona fala, questiona, usa o vocabulário e encadeia perguntas.
   Ex: persona "curioso" → encadeia com "E antes disso...? E relaciona com X?"
       persona "cético"  → questiona factos: "Mas tens a certeza? Não foi antes de Y?"
       persona "leigo"   → linguagem simples, informal, pode confundir datas
       persona "jornalista" → directo, curto, pede factos concretos
3. As RESPOSTAS (gpt) devem ser factualmente consistentes com o texto Wikipedia.
4. NÃO inventes factos novos. Baseia as respostas no Wikipedia fornecido.
5. **PROIBIÇÃO DE METALINGUAGEM:** NUNCA uses expressões como "não disponível", "não tenho informação", "não consta nos factos", "contexto não fornece". Se não souberes responder com certeza, cria um turno diferente sobre outro aspecto da entidade.
6. **SEM REPETIÇÕES:** NUNCA repitas a mesma pergunta ou resposta de um turno anterior — nem dos turnos TISER nem dos turnos casuais já gerados.
7. **PRIMEIRO TURNO — INTRODUÇÃO OBRIGATÓRIA E MUITO VARIADA:** O primeiro turno casual (human) DEVE mencionar o nome da entidade "{entity}" de forma natural. É ESTRITAMENTE PROIBIDO repetir a mesma frase de abertura em diálogos diferentes — cada diálogo deve ter uma abertura ÚNICA e DIFERENTE. Para inspiração, aqui estão 3 exemplos adequados à persona (mas NÃO os copies literalmente — adapta ou cria uma variante diferente):
{openers_sample}
   ATENÇÃO: estas sugestões são apenas ponto de partida. É PROIBIDO usar sempre a mesma estrutura. Se todas as sugestões começam com "Estou a estudar", usa algo completamente diferente. O critério de sucesso é: dois diálogos seguidos NÃO devem ter a mesma abertura de frase. Usa a entidade "{entity}", o contexto específico do Wikipedia, e o registo da persona para criar uma abertura genuinamente nova e contextualizada.
   Nunca comeces com um pronome ou referência implícita sem nomear primeiro a entidade.
   A partir do segundo turno, podes usar pronomes e co-referências naturais ("esse evento", "nesse mesmo ano", "naquela cidade") em vez de repetir o nome completo.
8. **REGISTO POR PERSONA:** Ajusta o registo da RESPOSTA à persona:
   - "curioso" / "estudante" / "leigo" → respostas conversacionais, frases curtas, tom próximo
   - "jornalista" → respostas directas, densas em factos, sem rodeios
   - "historiador" / "entrevistador" → respostas mais elaboradas mas nunca robóticas — usa linguagem natural de especialista, não de enciclopédia
   - "cético" → a resposta pode incluir nuance ou ressalvas, mas sempre em tom humano
   NUNCA uses abertura de frase como "Certamente!", "Com prazer!", "Claro, de facto..." — soa a robô. Começa a resposta directamente no conteúdo.
9. Responde SEMPRE em português europeu.
10. Output: JSON com exactamente este formato:
    {{"turns": [{{"human": "...", "gpt": "..."}}, ...]}}
11. **DIVERSIDADE TEMÁTICA:** Os turnos casuais devem explorar dimensões DIFERENTES dos turnos temporais já gerados. Os turnos TISER já tratam datas e cronologia — NÃO repitas isso nos casuais. Nos casuais, explora: personalidade/carácter, contexto histórico-social, impacto/legado, curiosidades, comparações, motivações, descrições, anedotas, obras/contribuições, relações com outras figuras. Uma pergunta casual que seja apenas "Em que ano...?" ou "Quando foi...?" é uma falha — substitui sempre por algo mais rico.

{few_shot}

Output JSON:
"""

# ── LLM call ──────────────────────────────────────────────────────────────────
async def chat_complete(client, prompt: str, max_tokens: int = 4096,
                        temperature: float = 0.7) -> str:
    resp = await client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.92,
    )
    return resp.choices[0].message.content or ""

# ── JSON parser ───────────────────────────────────────────────────────────────
TISER_FULL_RE = re.compile(
    r"<reasoning>.*?<timeline>.*?</timeline>.*?<reflection>.*?</reflection>"
    r".*?</reasoning>.*?<answer>.+?</answer>",
    re.DOTALL,
)

def parse_turns(raw: str, require_tiser: bool = False) -> list | None:
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m: return None
        obj = json.loads(m.group())
        turns = obj.get("turns", [])
        if not turns or not all("human" in t and "gpt" in t for t in turns):
            return None
        if require_tiser and not any(TISER_FULL_RE.search(t["gpt"]) for t in turns):
            return None
        return turns
    except Exception:
        return None

def fmt_tiser_context(tiser_turns: list) -> str:
    """Format accepted TISER turns as a natural dialogue history for the casual gen prompt."""
    if not tiser_turns:
        return "  (nenhum turno temporal ainda)"
    lines = []
    for i, t in enumerate(tiser_turns):
        lines.append(f"  [{i+1}] Utilizador: \"{t['human']}\"")
        # Show only the answer (strip TISER tags so the casual model sees a clean conversation)
        am = re.search(r"<answer>(.*?)</answer>", t["gpt"], re.DOTALL)
        answer = am.group(1).strip() if am else t["gpt"][:150]
        lines.append(f"       AMALIA: \"{answer}\"")
    return "\n".join(lines)

# Awareness: the human embeds the 'contexto adicional' as a temporal reference in the question.
# AMALIA reasons about the chronology naturally — no meta-language allowed.
_DISTRACTOR_RULE_AWARENESS = (
    " Para cada turno, a pergunta do utilizador deve cruzar o facto-alvo com o"
    " 'contexto adicional sobre a entidade' que está entre parênteses nos dados."
    " Incorpora esse facto extra como contexto temporal ORGÂNICO e VARIADO na pergunta."
    " NUNCA uses sempre o mesmo template 'Antes de X, quando é que Y?'."
    " Usa estruturas conversacionais diversas e naturais, por exemplo:\n"
    "   - 'E [facto-alvo], isso foi antes ou depois de [evento X]?'\n"
    "   - 'Tenho uma dúvida — [facto-alvo] aconteceu enquanto [evento X] ainda estava a decorrer?'\n"
    "   - 'Relacionado com [evento X], sabes dizer-me quando foi [facto-alvo]?'\n"
    "   - 'Já que falámos de [evento X], como fica isso em relação a [facto-alvo]?'\n"
    "   - 'E na altura de [evento X], o que é que tinha acontecido com [facto-alvo]?'\n"
    "   - '[facto-alvo] — foi muito tempo depois de [evento X], ou foi logo a seguir?'\n"
    " Varia a estrutura frásica de pergunta para pergunta. Na resposta, raciocina sobre a"
    " cronologia dos dois eventos de forma natural, sem usar palavras como 'distractor',"
    " 'âncora', 'âncora temporal', 'ponto de referência', 'contexto adicional' ou jargão técnico."
)
# Cognition: clean questions, no distractor in the data. No extra rule needed.
_DISTRACTOR_RULE_COGNITION = ""  # empty — cognition pairs have no distractor_context

# ── Phase 2: Generate TISER turns ────────────────────────────────────────────
async def gen_tiser(job: dict, client) -> dict | None:
    dataset = job.get("dataset", "cognition")
    if dataset == "awareness":
        distractor_rule = _DISTRACTOR_RULE_AWARENESS
        few_shot = TISER_FEW_SHOT_AWARENESS
    else:
        distractor_rule = _DISTRACTOR_RULE_COGNITION
        few_shot = TISER_FEW_SHOT_COGNITION
    prompt = TISER_PROMPT.format(
        persona=job["persona_text"],
        entity=job["entity"],
        qa_block=build_qa_block(job["pairs"]),
        n_tiser=job["n_tiser"],
        distractor_rule=distractor_rule,
        few_shot=few_shot,
    )
    try:
        raw = await chat_complete(client, prompt)
        turns = parse_turns(raw, require_tiser=True)
        if turns is None:
            return {"_error": True}
        return {**job, "tiser_raw": turns}
    except Exception as e:
        log.warning(f"Tiser gen failed for {job['entity']}: {e}")
        return {"_error": True}

# ── Phase 4: Generate casual turns ───────────────────────────────────────────
async def gen_casual(job: dict, client) -> dict | None:
    entity = job["entity"]
    persona_text = job["persona_text"]
    wiki = job.get("wiki_text", "")
    wiki_excerpt = wiki[:5000] if wiki else "(sem Wikipedia disponível)"
    openers_sample = _sample_openers(persona_text, entity, n=3)
    prompt = CASUAL_PROMPT.format(
        persona=persona_text,
        entity=entity,
        tiser_context=fmt_tiser_context(job["accepted_tiser"]),
        wiki_excerpt=wiki_excerpt,
        n_casual=job["n_casual"],
        few_shot=CASUAL_FEW_SHOT,
        openers_sample=openers_sample,
    )
    try:
        raw = await chat_complete(client, prompt)
        turns = parse_turns(raw, require_tiser=False)
        if turns is None:
            return {"_error": True}
        return {**job, "casual_raw": turns}
    except Exception as e:
        log.warning(f"Casual gen failed for {job['entity']}: {e}")
        return {"_error": True}

# ── Phase 3b: Retry rejected TISER turns with judge feedback ──────────────────
TISER_RETRY_PROMPT = """\
Você é AMALIA, a assistente de IA especializada em raciocínio temporal em português europeu.

A tua geração anterior para a entidade "{entity}" foi parcialmente rejeitada.
Abaixo estão os turnos que foram REJEITADOS e os MOTIVOS:
{rejected_turns}

Conhecimento base (o mesmo de antes):
{qa_block}

Persona do utilizador: {persona}

Gera {n_retry} turnos novos e corrigidos que resolvam os problemas indicados.
Se’s a mesma persona e o mesmo estilo. Não repitas os turnos já aceites.
Se’s as regras anteriores: sem metalinguagem, perguntas plausíveis, sem repetições.

Para cada turno gpt, deves seguir RIGOROSAMENTE esta estrutura interna no valor do campo gpt:
<reasoning>
O teu raciocínio detalhado...
<timeline>
Linha temporal relevante...
</timeline>
<reflection>
Reflexão sobre os problemas apontados...
</reflection>
</reasoning>
<answer>
A resposta factual curta corrigida...
</answer>

Output JSON:
{{"turns": [{{"human": "...", "gpt": "..."}}, ...]}}
"""

async def gen_tiser_retry(job: dict, client) -> dict | None:
    """Re-generate rejected TISER turns for a job, passing the judge rejection reasons."""
    rejected = job.get("tiser_rejected", [])
    if not rejected:
        return {"_error": True}
    rejected_block = "\n".join(
        f"  - Turno: \"{r['turn'].get('human','')}\" | Motivo: {r.get('reason', '?')}"
        for r in rejected
    )
    prompt = TISER_RETRY_PROMPT.format(
        entity=job["entity"],
        rejected_turns=rejected_block,
        qa_block=build_qa_block(job["pairs"]),
        persona=job["persona_text"],
        n_retry=len(rejected),
    )
    try:
        raw = await chat_complete(client, prompt, temperature=0.8)
        turns = parse_turns(raw, require_tiser=True)
        if not turns:
            return {"_error": True}
        return {**job, "tiser_retry_raw": turns}
    except Exception as e:
        log.warning(f"TISER retry failed for {job['entity']}: {e}")
        return {"_error": True}

# ── Phase 4b: Co-reference introduction ───────────────────────────────────────
COREF_PROMPT = """\
Você é AMALIA, uma assistente de IA em português europeu.

Abaixo está uma conversa sobre a entidade "{entity}".
A conversa está correcta factualmente, mas é repetitiva: o nome "{entity}" e outros
nomes próprios e lugares são repetidos muitas vezes, tornando a conversa mecanizada.

Reescreve APENAS as perguntas (human) e respostas (gpt) casuais (sem tags <reasoning>)
para introduzir co-referências naturais, tornando a conversa mais fluida.
Por exemplo:
  - Depois de mencionar "{entity}" pela primeira vez, usa "ele", "ela", "o clube", "o artista", etc.
  - Depois de mencionar uma cidade, usa "naquela cidade", "aí", "no mesmo local".
  - Depois de mencionar um ano, usa "nesse mesmo ano", "nessa época", "pouco depois".

REGRAS OBRIGATÓRIAS:
1. NÃO alteres os turnos com tags <reasoning> — copia-os exactamente como estão.
2. NÃO alteres nenhum facto. Apenas muda a forma como são expressos.
3. Mantém o mesmo número de turnos na mesma ordem.
4. Output: JSON com exactamente este formato:
   {{"turns": [{{"human": "...", "gpt": "..."}}, ...]}}

Conversa original:
{dialog_json}

Output JSON:
"""

async def gen_coref(job: dict, client) -> dict | None:
    """Introduce co-references into the casual turns of a dialog."""
    all_turns = job.get("all_turns_draft", [])
    if not all_turns:
        return job  # nothing to rewrite
    dialog_json = json.dumps({"turns": all_turns}, ensure_ascii=False, indent=2)
    prompt = COREF_PROMPT.format(
        entity=job["entity"],
        dialog_json=dialog_json[:6000],  # cap to avoid token overflow
    )
    try:
        raw = await chat_complete(client, prompt, temperature=0.5)
        new_turns = parse_turns(raw, require_tiser=False)
        if not new_turns or len(new_turns) != len(all_turns):
            return job  # fallback: keep original if coref fails
        return {**job, "all_turns_draft": new_turns}
    except Exception as e:
        log.warning(f"Coref failed for {job['entity']}: {e}")
        return {"_error": True}

# ── Output helpers ────────────────────────────────────────────────────────────
def strip_tiser_tags(turn: dict) -> dict:
    """Return a copy of the turn with TISER XML tags stripped, keeping only <answer> content."""
    gpt = turn.get("gpt", "")
    if "<reasoning>" not in gpt:
        return turn  # casual turn — already clean
    m = re.search(r"<answer>(.*?)</answer>", gpt, re.DOTALL)
    clean_gpt = m.group(1).strip() if m else gpt
    return {**turn, "gpt": clean_gpt}

# ── Phase 6: Interleave ───────────────────────────────────────────────────────
def interleave(tiser_turns: list, casual_turns: list) -> list:
    """
    Interleave casual and TISER turns:
    casual (intro), TISER, casual, TISER, ..., remaining casual

    The first casual turn introduces the entity naturally (rule 7 in CASUAL_PROMPT
    guarantees the entity name is mentioned). Then TISER temporal questions follow
    with full conversational context established.
    """
    result = []
    t_iter = iter(tiser_turns)
    c_iter = iter(casual_turns)
    t_done, c_done = False, False

    # Alternate casual → TISER (casual first for natural entity introduction)
    while not t_done:
        if not c_done:
            c = next(c_iter, None)
            if c is None:
                c_done = True
            else:
                result.append(c)
        t = next(t_iter, None)
        if t is None:
            t_done = True
        else:
            result.append(t)

    # Remaining casual turns
    for c in c_iter:
        result.append(c)

    return result

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng      = random.Random(RANDOM_SEED)
    datasets = ([DATASET_FILTER] if DATASET_FILTER in ("cognition", "awareness", "reasoning")
                else ["cognition", "awareness", "reasoning"])

    # ── Phase 1: Build jobs ───────────────────────────────────────────────────
    log.info("━" * 64)
    log.info(f"[1/6] Building jobs (TEST={N_TEST or 'FULL'}, FILTER={CATEGORY_FILTER or 'ALL'})")
    all_jobs, job_ds = [], []

    for ds in datasets:
        f = DATA_DIR / f"saudade_{ds}_pt_train.json"
        if not f.exists(): log.warning(f"Missing: {f}"); continue
        data = json.load(open(f, encoding="utf-8"))
        if CATEGORY_FILTER:
            target_cats = [c.strip() for c in CATEGORY_FILTER.split(',')]
            filtered_data = {}
            for cat in target_cats:
                if cat in data:
                    filtered_data[cat] = data[cat]
                else:
                    log.warning(f"[{ds}] Category '{cat}' not found — skipping")
            data = filtered_data
            log.info(f"  [{ds}] CATEGORY_FILTER applied for {len(target_cats)} categories")

        entity_qa = collect_qa_items(data, ds)
        if N_TEST > 0:
            keys = list(entity_qa.keys()); rng.shuffle(keys)
            entity_qa = {k: entity_qa[k] for k in keys[:N_TEST]}
        log.info(f"  [{ds}] {len(entity_qa)} entities")

        for entity, pairs in entity_qa.items():
            wiki_text = load_wiki_text(entity)
            for ci in build_jobs_per_persona(pairs, rng):
                all_jobs.append({
                    "entity": entity,
                    "pairs": ci["pairs"],
                    "persona_id": ci["persona_id"],
                    "persona_text": ci["persona_text"],
                    "n_tiser": ci["n_tiser"],
                    "n_casual": ci["n_casual"],
                    "wiki_text": wiki_text,
                    "dataset": ds,
                })
                job_ds.append(ds)

    log.info(f"  Total jobs: {len(all_jobs)}")
    log.info(f"  Phase 1 done in {hms(time.time()-t0)}")

    log.info(f"  Generator : {MODEL_ID} @ pool")
    log.info(f"  Judge     : {MODEL_ID} @ pool")
    if USE_PREPROCESSED_SPLIT:
        log.info("  Data split: fact-pair 80/20")
    else:
        log.info("  Data split: entity-level 80/20")

    async def run():
        # ── Setup AsyncOpenAI Client Pool (Load Balancing across SLURM nodes) ─
        from openai import AsyncOpenAI
        endpoints_raw = os.environ.get("VLLM_ENDPOINTS", f"127.0.0.1:{PORT}")
        endpoints = [e.strip() for e in endpoints_raw.split(",") if e.strip()]
        
        class ClientPool:
            def __init__(self, eps, api_key="vllm"):
                self.clients = [AsyncOpenAI(base_url=f"http://{ep}/v1", api_key=api_key, timeout=1200.0, max_retries=0) for ep in eps]
            def get(self):
                return random.choice(self.clients)
            # Proxy chat to random client
            class _Chat:
                def __init__(self, pool): self.pool = pool
                @property
                def completions(self): return self
                async def create(self, **kwargs):
                    return await self.pool.get().chat.completions.create(**kwargs)
            @property
            def chat(self): return self._Chat(self)

        client = None
        judge_client = None
        if RUN_PHASE in (2, 4, "3b", "4b"):
            log.info(f"  Generator Endpoints: {len(endpoints)} nodes")
            client = ClientPool(endpoints)
        elif RUN_PHASE in (3, 5):
            log.info(f"  Judge Endpoints: {len(endpoints)} nodes")
            judge_client = ClientPool(endpoints)
        elif RUN_PHASE == 0:
            client = ClientPool(endpoints)
            judge_client = ClientPool(endpoints)
            
        # ── Phase 2: Generate TISER turns ─────────────────────────────────────
        if RUN_PHASE_NUM > 2:
            ok2 = load_checkpoint("p2_tiser_gen")
            if not ok2: log.error("Cannot run Phase > 2 without p2_tiser_gen checkpoint!"); return
        elif RUN_PHASE in (0, 2):
            log.info("━" * 64)
            ck2 = load_checkpoint("p2_tiser_gen") if RESUME_FROM_NUM >= 2 else None
            if ck2 is not None and len(ck2) >= len(all_jobs) * 0.99:
                ok2 = ck2
                log.info(f"[2/6] TISER gen — RESUMED from full checkpoint ({len(ok2)} records)")
            elif ck2 is not None:
                # Partial checkpoint — resume from where we left off
                done_keys = {_job_hash(r) for r in ck2}
                remaining = [j for j in all_jobs if _job_hash(j) not in done_keys]
                log.info(f"[2/6] TISER gen — PARTIAL resume: {len(ck2)} done, {len(remaining)} remaining")
                t2 = time.time()
                async def _gen_tiser(job):
                    return await gen_tiser(job, client)
                new_results = await run_parallel(remaining, _gen_tiser, MAX_CONCURRENT, "tiser-gen", save_fn=lambda r: save_checkpoint("p2_tiser_gen", ck2 + [x for x in r if x is not None]), save_interval=120)
                new_ok2 = [r for r in new_results if r is not None]
                ok2 = ck2 + new_ok2
                save_checkpoint("p2_tiser_gen", ok2)
                partial2 = STOP_REQUESTED
                log.info(f"  TISER gen: +{len(new_ok2)} new, {len(ok2)} total" + (" [PARTIAL]" if partial2 else ""))
                log.info(f"  Phase 2 done in {hms(time.time()-t2)}")
                if partial2:
                    log.warning("  [SIGNAL] Partial checkpoint saved — exiting for resubmission.")
                    return
            else:
                log.info(f"[2/6] Generating TISER turns ({len(all_jobs)} jobs, {MAX_CONCURRENT} workers)")
                t2 = time.time()
                async def _gen_tiser(job):
                    return await gen_tiser(job, client)
                tiser_results = await run_parallel(all_jobs, _gen_tiser, MAX_CONCURRENT, "tiser-gen", save_fn=lambda r: save_checkpoint("p2_tiser_gen", [x for x in r if x is not None]), save_interval=120)
                ok2 = [r for r in tiser_results if r is not None]
                save_checkpoint("p2_tiser_gen", ok2)
                partial2 = STOP_REQUESTED
                log.info(f"  TISER gen: {len(ok2)}/{len(all_jobs)} parsed ({len(all_jobs)-len(ok2)} failed)"  + (" [PARTIAL — will resume next job]" if partial2 else ""))
                log.info(f"  Phase 2 done in {hms(time.time()-t2)}")
            
            if RUN_PHASE == 2:
                _ckpt("p2_tiser_gen").with_suffix(".done").touch()
                log.info("  [SIGNAL] RUN_PHASE=2 requested. Exiting cleanly to allow model swap.")
                return
            if 'partial2' in locals() and partial2:
                log.warning("  [SIGNAL] Partial checkpoint saved — exiting for resubmission.")
                return

        # ── Phase 3: Judge TISER turns (B2 only) ──────────────────────────────
        ck3 = None
        if RUN_PHASE_NUM >= 4.0:
            # For Phase 4 onwards we do NOT load the full p3 into RAM (432k lines / 14GB overhead).
            # Instead we lazily stream it later inside Phase 4. We just confirm the file exists.
            _p3_slim = _ckpt("p3_tiser_judge_slim")
            _p3_full = _ckpt("p3_tiser_judge")
            _p3_src = _p3_slim if _p3_slim.exists() else (_p3_full if _p3_full.exists() else None)
            if not _p3_src:
                log.error("Cannot run Phase > 3 without p3_tiser_judge checkpoint!"); return
            ok3 = None  # will be streamed inside Phase 4
        elif RUN_PHASE in (0, 3, "3b"):
            log.info("━" * 64)
            ck3 = load_checkpoint("p3_tiser_judge")
            if ck3 is not None:
                ok3 = ck3
                log.info(f"[3/6] TISER judge — RESUMED from checkpoint ({len(ok3 or [])} records)")
            else:
                log.info(f"[3/6] Judging TISER turns ({len(ok2)} jobs, B2-only, ≥{MIN_TISER_TURNS} required)")
                t3 = time.time()
                async def _judge_tiser(job):
                    accepted, rejected = await judge_tiser_turns(
                        turns=job["tiser_raw"],
                        qa_pairs=job["pairs"],
                        client=judge_client,
                        model_id=JUDGE_MODEL_ID,
                        entity=job["entity"],
                        wiki=job.get("wiki_text", ""),
                    )
                    if len(accepted) < MIN_TISER_TURNS:
                        return None
                    return {**job, "accepted_tiser": accepted, "tiser_rejected": rejected}
                judged_tiser = await run_parallel(ok2, _judge_tiser, MAX_CONCURRENT // 2, "tiser-judge", save_fn=lambda r: save_checkpoint("p3_tiser_judge", [x for x in r if x is not None]), save_interval=120)
                ok3 = [r for r in judged_tiser if r is not None]
                # ── Structural repetition filter (Python-only, zero LLM cost) ──
                # Reject jobs where >55% of TISER turns start with the same word
                # (signals the model templated "Antes de X..." across all turns)
                def _is_repetitive(job: dict) -> bool:
                    turns = job.get("accepted_tiser", [])
                    if len(turns) < 3:
                        return False
                    first_words = [t["human"].strip().split()[0].lower().rstrip(",") for t in turns if t.get("human")]
                    if not first_words:
                        return False
                    most_common = max(set(first_words), key=first_words.count)
                    ratio = first_words.count(most_common) / len(first_words)
                    return ratio > 0.55

                rep_filtered = [r for r in ok3 if _is_repetitive(r)]
                ok3 = [r for r in ok3 if not _is_repetitive(r)]
                if rep_filtered:
                    log.info(f"  [RepFilter] Removed {len(rep_filtered)} repetitive-template jobs ({len(rep_filtered)/(len(ok3 or [])+len(rep_filtered))*100:.1f}%)")
                # ─────────────────────────────────────────────────────────────────
                save_checkpoint("p3_tiser_judge", ok3)
                partial3 = STOP_REQUESTED
                log.info(f"  TISER judge: {len(ok3 or [])}/{len(ok2)} jobs passed" + (" [PARTIAL]" if partial3 else ""))
                log.info(f"  Phase 3 done in {hms(time.time()-t3)}")

        if RUN_PHASE == 3:
            if 'partial3' not in locals() or not partial3:
                    _ckpt("p3_tiser_judge").with_suffix(".done").touch()
            log.info("  [SIGNAL] RUN_PHASE=3 requested. Exiting cleanly to allow model swap.")
            return
        if 'partial3' in locals() and partial3:
            log.warning("  [SIGNAL] Partial checkpoint saved — exiting for resubmission.")
            return

        # ── Phase 3b: Retry rejected TISER turns (Qwen) ────────────────────────
        if RUN_PHASE not in ("3b", 0):
            ok3b = load_checkpoint("p3b_tiser_retry") or []
        else:
            retry_jobs = [j for j in ok3 if j.get("tiser_rejected")]
            ck3b = load_checkpoint("p3b_tiser_retry")
            if ck3b is not None and len(ck3b) >= len(retry_jobs) * 0.99:
                ok3b = ck3b
                log.info(f"[3b] TISER retry — RESUMED from full checkpoint ({len(ok3b)} records)")
            else:
                ck3b = ck3b or []
                remaining3b = retry_jobs[len(ck3b):]
                log.info(f"[3b] Retrying {len(retry_jobs)} jobs with rejected TISER turns ({len(remaining3b)} remaining)")
                t3b = time.time()
                async def _retry_tiser(job):
                    return await gen_tiser_retry(job, client)
                retry_results = await run_parallel(remaining3b, _retry_tiser, MAX_CONCURRENT, "tiser-retry", save_fn=lambda r: save_checkpoint("p3b_tiser_retry", ck3b + [x for x in r if x is not None]), save_interval=120)
                ok3b = ck3b + [r for r in retry_results if r is not None]
                save_checkpoint("p3b_tiser_retry", ok3b)
                log.info(f"  TISER retry: {len(ok3b)}/{len(retry_jobs)} recovered")
                log.info(f"  Phase 3b done in {hms(time.time()-t3b)}")

        if RUN_PHASE == "3b":
            _ckpt("p3b_tiser_retry").with_suffix(".done").touch()
            log.info("  [SIGNAL] RUN_PHASE=3b done. Exiting to allow model swap.")
            return

        # ── Phase 4: Generate casual turns (STREAMING — no full p3 load into RAM) ──
        if RUN_PHASE_NUM > 4:
            ok4 = load_checkpoint("p4_casual_gen")
            if not ok4: log.error("Cannot run Phase > 4 without p4_casual_gen checkpoint!"); return
        elif RUN_PHASE in (0, 4):
            log.info("━" * 64)
            t4 = time.time()

            # Load only the hashes of already-done jobs (tiny set in RAM)
            ck4_path = _ckpt("p4_casual_gen")
            done_hashes: set = set()
            ck4_count = 0
            if ck4_path.exists():
                try:
                    import orjson as _orjson
                    _parse = _orjson.loads
                except ImportError:
                    _parse = json.loads
                with open(ck4_path, encoding="utf-8") as _f4:
                    for _line in _f4:
                        _line = _line.strip()
                        if _line:
                            done_hashes.add(_job_hash(_parse(_line)))
                            ck4_count += 1
                log.info(f"[4/6] Casual gen — {ck4_count} already done (hashes loaded, zero full-load RAM cost)")
            
            # Stream p3 file and build remaining list — only jobs whose hash isn't in done_hashes
            remaining4 = []
            p3_total = 0
            with open(_p3_src, encoding="utf-8") as _f3:  # type: ignore[possibly-undefined]
                try:
                    import orjson as _orjson
                    _parse3 = _orjson.loads
                except ImportError:
                    _parse3 = json.loads
                for _line in _f3:
                    _line = _line.strip()
                    if not _line:
                        continue
                    p3_total += 1
                    _job = _parse3(_line)
                    if _job_hash(_job) not in done_hashes:
                        remaining4.append(_job)
            
            log.info(f"[4/6] Casual gen — {p3_total} total jobs, {ck4_count} done, {len(remaining4)} remaining")

            if not remaining4:
                ok4 = load_checkpoint("p4_casual_gen") or []
                log.info(f"[4/6] Casual gen — already complete ({len(ok4 or [])} records)")
            else:
                async def _gen_casual(job):
                    return await gen_casual(job, client)
                
                # Streaming save: append sequentially, stopping at the first incomplete item
                _p4_out = open(ck4_path, "a", encoding="utf-8", buffering=1)
                _p4_written = [ck4_count]
                def _streaming_save(results_array):
                    while _p4_written[0] - ck4_count < len(results_array):
                        r = results_array[_p4_written[0] - ck4_count]
                        if r is None:
                            break  # wait for this item to complete before appending further
                        if not (isinstance(r, dict) and r.get("_error")):
                            _p4_out.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
                        _p4_written[0] += 1

                new_results4 = await run_parallel(remaining4, _gen_casual, MAX_CONCURRENT, "casual-gen",
                                                   save_fn=_streaming_save, save_interval=120)
                _p4_out.close()
                new_ok4 = [r for r in new_results4 if r is not None]
                partial4 = STOP_REQUESTED
                total4 = ck4_count + len(new_ok4)
                log.info(f"  Casual gen: +{len(new_ok4)} new, {total4} total" + (" [PARTIAL]" if partial4 else ""))
                log.info(f"  Phase 4 done in {hms(time.time()-t4)}")
                if partial4:
                    log.warning("  [SIGNAL] Partial checkpoint saved — exiting for resubmission.")
                    return
                ok4 = load_checkpoint("p4_casual_gen") or []
            
            if RUN_PHASE == 4:
                if 'partial4' not in locals() or not partial4:
                    _ckpt("p4_casual_gen").with_suffix(".done").touch()
                log.info("  [SIGNAL] RUN_PHASE=4 requested. Exiting cleanly to allow model swap.")
                return
            if 'partial4' in locals() and partial4:
                log.warning("  [SIGNAL] Partial checkpoint saved — exiting for resubmission.")
                return

        # ── Phase 4b: Co-reference introduction (Qwen) ────────────────────────
        if RUN_PHASE not in ("4b", 0):
            ok4b = None # OOM FIX: Don't load 5.7 GB here. Phase 5 will stream it.
        else:
            ck4b = load_checkpoint("p4b_coref")
            if ck4b is not None and len(ck4b) >= len(ok4 or []) * 0.99:
                ok4b = ck4b
                log.info(f"[4b] Co-reference — RESUMED from full checkpoint ({len(ok4b)} records)")
            else:
                ck4b = ck4b or []
                log.info(f"[4b] Introducing co-references into {len(ok4 or [])} dialogs")
                t4b = time.time()
                # Build draft interleaved turns so coref sees the full conversation
                def _make_draft(job):
                    tiser = job.get("accepted_tiser", [])
                    casual = job.get("casual_raw", [])
                    draft = interleave(tiser, casual)
                    return {**job, "all_turns_draft": draft}
                ok4_with_draft = [_make_draft(j) for j in ok4]
                remaining4b = ok4_with_draft[len(ck4b):]
                async def _coref(job):
                    return await gen_coref(job, client)
                coref_results = await run_parallel(remaining4b, _coref, MAX_CONCURRENT, "coref", save_fn=lambda r: save_checkpoint("p4b_coref", ck4b + [x for x in r if x is not None]), save_interval=120)
                ok4b = ck4b + [r for r in coref_results if r is not None]
                save_checkpoint("p4b_coref", ok4b)
                log.info(f"  Co-reference: {len(ok4b)} dialogs processed")
                log.info(f"  Phase 4b done in {hms(time.time()-t4b)}")

        if RUN_PHASE == "4b":
            log.info("  [SIGNAL] RUN_PHASE=4b done. Exiting to allow model swap.")
            return

        # ── Phase 5: Judge casual + retried TISER (B1+B2 vs Wikipedia) ───────
        ck5 = None
        if RUN_PHASE not in (5, 0):
            ok5 = load_checkpoint("p5_casual_judge")
            if not ok5: log.error("Cannot run Phase 5 without p5_casual_judge checkpoint!"); return
        elif RUN_PHASE in (5, 0):
            log.info("━" * 64)
            ck5 = load_checkpoint("p5_casual_judge") or []
            
            t5 = time.time()
            # Sub-step 5a: judge retried TISER turns (p3b) while Gemma is active
            if ok3b:
                log.info(f"  [5a] Judging {len(ok3b)} retried TISER jobs")
                async def _judge_retry_tiser(job):
                    if not job or job.get("_error") or "pairs" not in job:
                        return {**job, "accepted_tiser_retry": []}
                    accepted, _ = await judge_tiser_turns(
                        turns=job.get("tiser_retry_raw", []),
                        qa_pairs=job["pairs"],
                        client=judge_client,
                        model_id=JUDGE_MODEL_ID,
                        entity=job["entity"],
                        wiki=job.get("wiki_text", ""),
                        n_workers=JUDGE_WORKERS,
                    )
                    return {**job, "accepted_tiser_retry": accepted}
                retry_judged = await run_parallel(ok3b, _judge_retry_tiser, MAX_CONCURRENT // 2, "retry-tiser-judge")
                save_checkpoint("p5a_tiser_retry_judged", retry_judged)
                log.info("  [5a] Retry TISER judged")

            # Sub-step 5b: judge casual turns in chunks to avoid OOM
            CHUNK_SIZE = 5000
            total_processed = len(ck5)
            log.info(f"  [5b] Judging casual turns in CHUNKS of {CHUNK_SIZE} (skipping {total_processed} done)")
            import gc
            
            while True:
                chunk = []
                with open(_ckpt("p4b_coref"), encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        if idx < total_processed: continue
                        line = line.strip()
                        if line: chunk.append(json.loads(line))
                        if len(chunk) == CHUNK_SIZE: break
                
                if not chunk:
                    break
                
                log.info(f"  [5b] Processing chunk of {len(chunk)} jobs (already done: {total_processed})...")
                async def _judge_casual(job):
                    if job.get("_error") or "entity" not in job:
                        return job
                    draft = job.get("all_turns_draft", [])
                    casual_only = [t for t in draft if "<reasoning>" not in t.get("gpt", "")] if draft else job.get("casual_raw", [])
                    if not casual_only:
                        casual_only = job.get("casual_raw", [])
                    accepted, rejected = await judge_casual_turns(
                        turns=casual_only,
                        qa_pairs=job["pairs"],
                        client=judge_client,
                        model_id=JUDGE_MODEL_ID,
                        entity=job["entity"],
                        wiki=job.get("wiki_text", ""),
                        n_workers=JUDGE_WORKERS,
                    )
                    return {**job, "accepted_casual": accepted, "casual_rejected": rejected}
                
                judged_chunk = await run_parallel(chunk, _judge_casual, min(MAX_CONCURRENT // 2, 2500), "casual-judge", save_fn=None)
                ck5 = ck5 + [r for r in judged_chunk if r is not None]
                save_checkpoint("p5_casual_judge", ck5)
                total_processed += len(chunk)
                
                del chunk, judged_chunk
                gc.collect()
            
            ok5 = ck5
            partial5 = STOP_REQUESTED
            log.info(f"  Casual judge done: {len(ok5)} jobs" + (" [PARTIAL]" if partial5 else ""))
            log.info(f"  Phase 5 done in {hms(time.time()-t5)}")

        if RUN_PHASE == 5:
            if 'partial5' not in locals() or not partial5:
                    _ckpt("p5_casual_judge").with_suffix(".done").touch()
            log.info("  [SIGNAL] RUN_PHASE=5 requested. Exiting cleanly to allow model swap.")
            return

        # ── Phase 6: Interleave + write dual output (tiser + clean) ───────────
        if RUN_PHASE in (0, 6):
            log.info("━" * 64)
            log.info("[6/6] Interleaving and writing dual output")

        # Load judged retried TISER turns if present
        retry_judged_index = {}
        try:
            retry_judged = load_checkpoint("p5a_tiser_retry_judged")
            if retry_judged:
                for j in retry_judged:
                    if not j or j.get("_error") or "entity" not in j:
                        continue
                    key = (j["entity"], j["persona_id"], str(j.get("pairs", [])))
                    retry_judged_index[key] = j
                log.info(f"Loaded {len(retry_judged_index)} retry-TISER judgements for merging")
        except Exception as e:
            log.warning(f"Could not load retry-TISER judgements: {e}")

        writers_tiser = {}
        writers_clean = {}
        for ds in datasets:
            # Write to chunk-specific files to avoid overwriting previous chunks or duplicating data on restarts
            chunk_prefix = CATEGORY_FILTER.replace(',', '_') if CATEGORY_FILTER else 'all'
            writers_tiser[ds] = open(OUTPUT_DIR / f"{ds}_dialog_sft_{chunk_prefix}_tiser.jsonl", "w", encoding="utf-8")
            writers_clean[ds] = open(OUTPUT_DIR / f"{ds}_dialog_sft_{chunk_prefix}_clean.jsonl", "w", encoding="utf-8")

        accepted_count = 0
        rejected_final = []

        for job in ok5:
            if job.get("_error"):
                continue
            entity = job.get("entity")
            if not entity:
                continue
                
            tiser_raw  = job.get("accepted_tiser", [])
            # Merge retried turns if present while preserving their original chronological positions
            key = (job["entity"], job["persona_id"], str(job.get("pairs", [])))
            if key in retry_judged_index:
                retry_job = retry_judged_index[key]
                orig_tiser_raw = retry_job.get("tiser_raw", [])
                if orig_tiser_raw:
                    rebuilt_tiser = [None] * len(orig_tiser_raw)
                    # 1. Map originally accepted turns by normalized question text
                    accepted_map = {t["human"].strip().lower()[:100]: t for t in tiser_raw if "human" in t}
                    # 2. Map retried accepted turns (matching 1-to-1 with tiser_rejected)
                    tiser_rejected = retry_job.get("tiser_rejected", [])
                    tiser_retry_raw = retry_job.get("tiser_retry_raw", [])
                    accepted_tiser_retry = retry_job.get("accepted_tiser_retry", [])
                    
                    retry_accepted_set = {t["human"].strip().lower()[:100]: t for t in accepted_tiser_retry if "human" in t}
                    retry_map = {}
                    for idx_rej, r_dict in enumerate(tiser_rejected):
                        if idx_rej < len(tiser_retry_raw):
                            retry_turn = tiser_retry_raw[idx_rej]
                            if "human" in retry_turn:
                                retry_q_key = retry_turn["human"].strip().lower()[:100]
                                if retry_q_key in retry_accepted_set:
                                    orig_rej_turn = r_dict["turn"]
                                    if "human" in orig_rej_turn:
                                        orig_q_key = orig_rej_turn["human"].strip().lower()[:100]
                                        retry_map[orig_q_key] = retry_accepted_set[retry_q_key]
                    
                    # Fill the rebuilt list
                    for idx, orig_turn in enumerate(orig_tiser_raw):
                        if "human" not in orig_turn:
                            continue
                        q_key = orig_turn["human"].strip().lower()[:100]
                        if q_key in accepted_map:
                            rebuilt_tiser[idx] = accepted_map[q_key]
                        elif q_key in retry_map:
                            rebuilt_tiser[idx] = retry_map[q_key]
                    tiser_raw = [t for t in rebuilt_tiser if t is not None]
                else:
                    tiser_raw = tiser_raw + retry_job.get("accepted_tiser_retry", [])

            casual_raw = job.get("accepted_casual", [])
            # Dedup TISER turns by question — guards against retry turns duplicating originals
            seen_q: set = set()
            tiser = []
            for t in tiser_raw:
                q_key = t.get("human", "").strip().lower()[:80]
                if q_key not in seen_q:
                    seen_q.add(q_key)
                    tiser.append(t)
            # Dedup casual turns by question (same seen_q set catches cross-type dupes too)
            casual = []
            for t in casual_raw:
                q_key = t.get("human", "").strip().lower()[:80]
                if q_key not in seen_q:
                    seen_q.add(q_key)
                    casual.append(t)
            if len(tiser) < MIN_TISER_TURNS:
                rejected_final.append({"entity": entity, "reason": "insufficient TISER turns"})
                continue
            turns = interleave(tiser, casual)
            if not turns:
                rejected_final.append({"entity": entity, "reason": "empty after interleave"})
                continue
            record = {
                "entity":   entity,
                "dataset":  job["dataset"],
                "persona":  job["persona_id"],
                "n_tiser":  len(tiser),
                "n_casual": len(casual),
                "turns":    turns,
            }
            ds = job["dataset"]
            writer_tiser = writers_tiser.get(ds)
            writer_clean = writers_clean.get(ds)
            if writer_tiser is None or writer_clean is None:
                log.warning(f"No writers for dataset '{ds}' — skipping {job['entity']}")
                continue
            
            writer_tiser.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            record_clean = record.copy()
            record_clean["turns"] = [strip_tiser_tags(t) for t in turns]
            writer_clean.write(json.dumps(record_clean, ensure_ascii=False) + "\n")
            
            accepted_count += 1

        for ds_writers in (writers_tiser, writers_clean):
            for fh in ds_writers.values():
                fh.close()

        # ── Summary ───────────────────────────────────────────────────────────
        wall = time.time() - t0
        log.info("━" * 64)
        log.info("SUMMARY")
        log.info(f"  Jobs attempted:    {len(all_jobs)}")
        log.info(f"  TISER gen ok:      {len(ok2)}")
        log.info(f"  TISER judge ok:    {len(ok3 or [])}")
        log.info(f"  Casual gen ok:     {len(ok4 or [])}")
        log.info(f"  Casual judge ok:   {len(ok5)}")
        log.info(f"  Final accepted:    {accepted_count}")
        log.info(f"  Acceptance rate:   {accepted_count/max(len(all_jobs),1)*100:.1f}%")
        log.info(f"  Total wall time:   {hms(wall)}")
        log.info("━" * 64)
        rej_path = OUTPUT_DIR / "sft_rejected.json"
        json.dump(rejected_final, open(rej_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        log.info(f"  Rejected debug:  {rej_path}")
        log.info(f"  Checkpoints:     {CHECKPT_DIR}")
        log.info("  Resume tip: RESUME_FROM=<phase_number> to skip completed phases")

    asyncio.run(run())

if __name__ == "__main__":
    main()
