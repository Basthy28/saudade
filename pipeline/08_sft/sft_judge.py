"""
sft_judge.py
===============
Staged judge for the SFT pipeline.

Two specialized judges — each with a single, focused reference:

  judge_tiser_turns()  — B1 + B2 with qa_pairs reference
  judge_casual_turns() — B1 + B2 with Wikipedia reference

No B3 (dialog-level): coherence is guaranteed by Stage 2 generation
seeing the accepted TISER turns as context.

Concurrency: asyncio.Queue producer-consumer (bounded, auto back-pressure).
"""
import os
from pathlib import Path

import re
import sys
import json
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

JUDGE_MAX_TOKENS  = 1024
JUDGE_TEMPERATURE = 0.0
SLIDING_WINDOW    = 2

TISER_FULL_RE = re.compile(
    r"<reasoning>.*?<timeline>.*?</timeline>.*?<reflection>.*?</reflection>"
    r".*?</reasoning>.*?<answer>.+?</answer>",
    re.DOTALL,
)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# ── JSON extractor ─────────────────────────────────────────────────────────────

def _extract_last_json(raw: str, required_keys: tuple) -> Optional[dict]:
    clean = re.sub(r"`{3}(?:json)?\s*", "", raw)
    clean = re.sub(r"`{3}\s*", "", clean).strip()
    candidates, depth, start = [], 0, -1
    for idx, ch in enumerate(clean):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidates.append(clean[start:idx + 1])
                start = -1
    for blob in reversed(candidates):
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict) and all(k in obj for k in required_keys):
                return obj
        except json.JSONDecodeError:
            pass
    return None

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class EvalUnit:
    turn_idx:            int
    human_turn:          str
    gpt_turn:            str
    extracted_answer:    str
    context_window:      list
    qa_pairs_block:      str
    entity_name:         str  = ""
    wiki_text:           str  = ""
    all_prior_questions: list = None

@dataclass
class TurnVerdict:
    turn_idx:       int
    fact:           int
    scope:          int
    tiser_required: int
    reasoning:      str
    raw_output:     str
    q_factual:      int = 1
    q_plausible:    int = 1
    q_reasoning:    str = ""
    q_raw_output:   str = ""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_qa(qa_pairs: list) -> str:
    lines = []
    for i, p in enumerate(qa_pairs, 1):
        lines.append(f"  {i}. Q: {p.get('question','')}  A: {p.get('answer','')}")
    return "\n".join(lines) or "  (sem factos de referência)"

def _fmt_window(window: list) -> str:
    if not window:
        return "  (início do diálogo)"
    lines = []
    for t in window:
        lines.append(f"  Utilizador: \"{t['human']}\"")
        gpt = t["gpt"][:200] + "..." if len(t["gpt"]) > 200 else t["gpt"]
        lines.append(f"  AMALIA: \"{gpt}\"")
    return "\n".join(lines)

# _find_expected removed: bag-of-words token overlap fails when qa_pairs is large
# (awareness entities can have 48+ pairs). The judge LLM identifies the relevant
# qa_pair itself from the full qa_block shown in the prompt.

def _build_units(turns: list, qa_pairs: list, entity: str, wiki: str) -> list:
    qa_block = _fmt_qa(qa_pairs)
    units = []
    for idx, turn in enumerate(turns):
        human = turn.get("human", "").strip()
        gpt   = turn.get("gpt",   "").strip()
        window_start = max(0, idx - SLIDING_WINDOW)
        am = ANSWER_TAG_RE.search(gpt)
        prior_qs = [t.get("human","").strip() for t in turns[:idx] if t.get("human","").strip()]
        units.append(EvalUnit(
            turn_idx=idx,
            human_turn=human,
            gpt_turn=gpt,
            extracted_answer=am.group(1).strip() if am else "",
            context_window=turns[window_start:idx],
            qa_pairs_block=qa_block,
            entity_name=entity,
            wiki_text=wiki,
            all_prior_questions=prior_qs,
        ))
    return units

# ── Concurrency helper ────────────────────────────────────────────────────────

async def run_parallel(items: list, async_fn, n_workers: int, desc: str = "", save_fn=None, save_interval: int = 60) -> list:
    """
    Bounded producer-consumer queue. Producer blocks when queue is full
    (back-pressure). Workers pick up immediately when done.
    Returns results in original order.
    """
    if not items:
        return []
    total = len(items)
    q = asyncio.Queue(maxsize=n_workers * 2)
    results = [None] * total
    done = [0]
    
    async def _saver():
        if not save_fn: return
        while True:
            await asyncio.sleep(save_interval)
            try:
                # Pass the raw results array (which includes Nones for uncompleted items)
                # so that save_fn can determine order and sequential completion.
                save_fn(results)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"  [{desc}] Auto-save error: {e}")

    async def worker():
        while True:
            # Check stop flag so workers abort immediately
            gen_mod = sys.modules.get("__main__") or sys.modules.get("generate_dialog_sft")
            if gen_mod and getattr(gen_mod, "STOP_REQUESTED", False):
                # Drain queue without processing to unblock join
                payload = await q.get()
                if payload is not None:
                    q.task_done()
                else:
                    q.task_done()
                    return
                continue

            payload = await q.get()
            if payload is None:
                q.task_done()
                return
            idx, data = payload
            try:
                results[idx] = await async_fn(data)
            except Exception as e:
                log.error(f"[{desc}] item {idx} error: {e}")
                results[idx] = {"_error": True}
            finally:
                done[0] += 1
                step = max(1, total // 10)
                if done[0] % step == 0 or done[0] == total:
                    log.info(f"  [{desc}] {done[0]}/{total} ({done[0]*100//total}%)")
                q.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(n_workers)]
    saver_task = asyncio.create_task(_saver())
    
    for idx, item in enumerate(items):
        # Respect stop flag set by SIGUSR1 in generate_dialog_sft
        gen_mod = sys.modules.get("__main__") or sys.modules.get("generate_dialog_sft")
        if gen_mod and getattr(gen_mod, "STOP_REQUESTED", False):
            log.warning(f"  [{desc}] Stop requested — skipping remaining {total - idx} items")
            break
        await q.put((idx, item))
    for _ in range(n_workers):
        await q.put(None)
    await q.join()
    saver_task.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    return results

# ── B2 prompts ────────────────────────────────────────────────────────────────

_B2_FEW_SHOT = """\
=== EXEMPLOS DE AVALIAÇÃO (Few-Shot) ===

--- EXEMPLO 1: TISER correcto, factualidade OK ---
Factos de referência:
  1. Q: Quando nasceu Eusébio?  A: 1942
Pergunta: "Tendo em conta que ele se reformou em 1979, quando nasceu?"
Resposta: "<reasoning>...<timeline>- 1942: Eusébio nasce...</timeline><reflection>...</reflection></reasoning><answer>1942</answer>"
Veredito: {"fact": 1, "scope": 1, "tiser_required": 1, "reasoning": "1942 correcto. TISER justificado por distrator."}

--- EXEMPLO 2: Resposta com facto inventado ---
Factos de referência:
  1. Q: Quando ganhou o título?  A: 1999
Pergunta: "E quando ganhou o título?"
Resposta: "Ganhou o título em 1987, numa época histórica."
Veredito: {"fact": 0, "scope": 1, "tiser_required": 0, "reasoning": "Resposta diz 1987, referência diz 1999. Facto errado."}

--- EXEMPLO 3: Resposta fora do âmbito (língua errada) ---
Resposta: "The club won the title in 1999, which was a great season."
Veredito: {"fact": 1, "scope": 0, "tiser_required": 0, "reasoning": "Resposta em inglês em vez de português europeu — scope=0."}

--- EXEMPLO 4: Turno casual, conversa legítima ---
Pergunta: "E como era a personalidade dele?"
Resposta (sem <answer>): "Era conhecido pela simpatia e garra dentro do campo."
Veredito: {"fact": 1, "scope": 1, "tiser_required": 0, "reasoning": "Conversa casual OK, sem factos concretos para verificar."}

=== FIM DOS EXEMPLOS ===
"""

_B2_TISER_TEMPLATE = """\
És um avaliador de qualidade de dados de treino em português europeu.
Avalia a RESPOSTA do assistente AMALIA num diálogo multi-turno.

{few_shot}

=== TURNO A AVALIAR ===
Entidade: {entity}
Referência (qa_pairs seed):
{qa_block}

Contexto recente ({window} turnos):
{context}

Pergunta: "{human}"
Resposta AMALIA: "{gpt}"
<answer> extraída: "{extracted}"

=== INSTRUÇÕES ===
Raciocina sobre 3 critérios:

1. FACTUALIDADE (fact): Identifica qual qa_pair acima é mais relevante para
   esta pergunta. A <answer> ou facto concreto na resposta coincide com esse
   par? fact=1 se correcto, fact=0 se contradiz a referência ou inventa factos.

2. ÂMBITO (scope): A resposta é sobre a entidade em discussão e em português
   europeu? scope=0 se completamente fora do tema OU se em inglês.

3. TISER NECESSÁRIO (tiser_required): A pergunta é temporal complexa com
   distractor? tiser_required=1 se sim, 0 se casual.

Veredito (JSON obrigatório na última linha):
{{"fact": <0|1>, "scope": <0|1>, "tiser_required": <0|1>, "reasoning": "<1-2 frases>"}}
"""

_B2_CASUAL_TEMPLATE = """\
És um avaliador de qualidade de dados de treino em português europeu.
Avalia a RESPOSTA do assistente AMALIA num turno casual (sem estrutura TISER).

{few_shot}

=== TURNO A AVALIAR ===
Entidade: {entity}
Texto Wikipedia (referência factual principal):
{wiki}

Tópico âncora (qa_pairs seed — para verificar âmbito):
{qa_block}

Contexto recente ({window} turnos):
{context}

Pergunta: "{human}"
Resposta AMALIA: "{gpt}"

=== INSTRUÇÕES ===
Raciocina sobre 2 critérios:

1. FACTUALIDADE (fact): A resposta introduz afirmações concretas verificáveis
   (datas, nomes, eventos)? Se sim, verifica contra o Wikipedia.
   fact=0 se contradiz o Wikipedia. fact=1 se consistente ou apenas opinativa.
   tiser_required=0 sempre (turnos casuais não precisam TISER).

2. ÂMBITO (scope): A resposta é sobre a entidade e em português europeu?
   scope=0 se totalmente fora do tema OU em inglês.

Veredito (JSON obrigatório na última linha):
{{"fact": <0|1>, "scope": <0|1>, "tiser_required": 0, "reasoning": "<1-2 frases>"}}
"""

def _build_tiser_prompt(unit: EvalUnit) -> str:
    return _B2_TISER_TEMPLATE.format(
        few_shot=_B2_FEW_SHOT,
        entity=unit.entity_name or "(entidade)",
        qa_block=unit.qa_pairs_block,
        window=SLIDING_WINDOW,
        context=_fmt_window(unit.context_window),
        human=unit.human_turn,
        gpt=unit.gpt_turn[:800] + "..." if len(unit.gpt_turn) > 800 else unit.gpt_turn,
        extracted=unit.extracted_answer or "(sem tag)",
    )

def _build_casual_prompt(unit: EvalUnit) -> str:
    wiki_excerpt = unit.wiki_text[:2500] if unit.wiki_text else "(sem Wikipedia)"
    return _B2_CASUAL_TEMPLATE.format(
        few_shot=_B2_FEW_SHOT,
        entity=unit.entity_name or "(entidade)",
        wiki=wiki_excerpt,
        qa_block=unit.qa_pairs_block,
        window=SLIDING_WINDOW,
        context=_fmt_window(unit.context_window),
        human=unit.human_turn,
        gpt=unit.gpt_turn[:600] + "..." if len(unit.gpt_turn) > 600 else unit.gpt_turn,
    )

# ── B1 prompt ─────────────────────────────────────────────────────────────────

_B1_FEW_SHOT = """\
=== EXEMPLOS — AVALIAÇÃO DA PERGUNTA ===

--- EXEMPLO 1: Premissa temporal errada (q_factual=0) ---
Contexto: João Paulo II foi papa de 1978 até 2005.
Pergunta: "Tendo em conta que João Paulo II morreu em 1978, quando é que Bento XVI foi eleito?"
Referência: 1. Q: Quando morreu João Paulo II?  A: 2005
Raciocínio: Premissa "morreu em 1978" é errada (foi em 2005).
Veredito: {"q_factual": 0, "q_plausible": 1, "reasoning": "Premissa errada: João Paulo II morreu em 2005."}

--- EXEMPLO 2: Pergunta casual válida (ambos=1) ---
Contexto: Eusébio nasceu em 1942.
Pergunta: "E era de que cidade?"
Raciocínio: Casual, sem premissas. Plausível.
Veredito: {"q_factual": 1, "q_plausible": 1, "reasoning": "Pergunta casual sem premissas."}

--- EXEMPLO 3: Salto temático total (q_plausible=0) ---
Contexto: conversa sobre Eusébio e futebol.
Pergunta: "Qual é a receita do bacalhau à brás?"
Raciocínio: Desvio total para culinária.
Veredito: {"q_factual": 1, "q_plausible": 0, "reasoning": "Salto temático total."}

--- EXEMPLO 4: Pergunta logicamente impossível — acção após morte (q_plausible=0) ---
Contexto: António Luís de Seabra morreu em 1895.
Pergunta: "Depois de ter morrido em 1895, quando é que António Luís de Seabra esteve em Rennes?"
Raciocínio: É logicamente impossível perguntar o que alguém fez DEPOIS de morrer. Nenhum humano faria esta pergunta.
Veredito: {"q_factual": 1, "q_plausible": 0, "reasoning": "Pergunta impossível: ninguém faz nada depois de morrer. Framing cronológico absurdo."}

--- EXEMPLO 5: Contradição cronológica — 'antes de X' mas a resposta é posterior a X (q_factual=0) ---
Contexto: Mariano Martins foi Governador Civil em 1915 e Capitão-de-Fragata em 1918.
Pergunta: "Mariano Martins foi Governador Civil em 1915. Antes disso, quando é que ascendeu a Capitão-de-Fragata?"
Raciocínio: A pergunta diz 'antes de 1915' mas a resposta (1918) é posterior a 1915. Contradição cronológica.
Veredito: {"q_factual": 0, "q_plausible": 0, "reasoning": "Contradição: pergunta diz 'antes de 1915' mas Capitão-de-Fragata foi em 1918, que é depois."}

--- EXEMPLO 6: Resposta já está na pergunta — tautológica (q_plausible=0) ---
Contexto: José Roçadas começou a carreira em 1882.
Pergunta: "Antes de falecer, quando é que José Roçadas começou a sua carreira militar em 1882?"
Raciocínio: A data 1882 já está na pergunta. Não há nada a responder — tautologia.
Veredito: {"q_factual": 1, "q_plausible": 0, "reasoning": "A resposta já consta na própria pergunta (1882). Pergunta tautológica."}}

=== FIM DOS EXEMPLOS ===
"""

_B1_TEMPLATE = """\
És um avaliador de qualidade de dados de treino em português europeu.
Avalia a PERGUNTA DO UTILIZADOR (não a resposta) num diálogo multi-turno.

{few_shot}

=== PERGUNTA A AVALIAR ===
Entidade: {entity}
Factos de referência:
{qa_block}

Perguntas anteriores no diálogo:
{prior_qs}

Contexto recente ({window} turnos):
{context}

Pergunta a avaliar: "{human}"

=== INSTRUÇÕES ===

Critério 1 – FACTUALIDADE DA PERGUNTA (q_factual):
  A pergunta embute algum pressuposto concreto demonstravelmente errado?
  q_factual=0 se há premissa claramente errada segundo a referência.
  q_factual=0 se a pergunta diz "antes de X" mas a resposta real é posterior a X (contradição cronológica).
  q_factual=1 se a pergunta é casual ou sem premissas concretas.

Critério 2 – PLAUSIBILIDADE NO CONTEXTO (q_plausible):
  A pergunta faz sentido dado o que foi discutido antes?
  Usa TODA a lista de perguntas anteriores (não só as recentes).
  q_plausible=0 se há salto temático total sem ligação ao fio da conversa.
  q_plausible=0 se a pergunta é logicamente impossível (ex: pedir acções de alguém após a sua morte).
  q_plausible=0 se a resposta já está embutida na própria pergunta (tautologia).
  q_plausible=1 para qualquer pergunta sobre a entidade que seja natural e plausível.

Veredito (JSON obrigatório na última linha):
{{"q_factual": <0|1>, "q_plausible": <0|1>, "reasoning": "<1-2 frases>"}}
"""

def _build_b1_prompt(unit: EvalUnit) -> str:
    prior_qs = unit.all_prior_questions or []
    prior_str = (
        "\n".join(f"  [{i+1}] {q}" for i, q in enumerate(prior_qs))
        if prior_qs else "  (primeiro turno)"
    )
    return _B1_TEMPLATE.format(
        few_shot=_B1_FEW_SHOT,
        entity=unit.entity_name or "(entidade)",
        qa_block=unit.qa_pairs_block,
        prior_qs=prior_str,
        window=SLIDING_WINDOW,
        context=_fmt_window(unit.context_window),
        human=unit.human_turn,
    )

# ── LLM call helper ───────────────────────────────────────────────────────────

async def _llm_call(client, model_id: str, prompt: str, label: str) -> str:
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        log.warning(f"[{label}] LLM call failed: {e}")
        return ""

# ── Core per-turn judge (B1 + B2) ─────────────────────────────────────────────

async def _judge_one_tiser(args) -> TurnVerdict:
    """
    B2 only for TISER turns.
    B1 is skipped because:
      - Questions come from qa_pairs — always on-topic (q_plausible=1 by design)
      - Awareness distractors are intentional temporal anchors (q_factual=1 by design)
    Only the answer quality matters here.
    """
    unit, qa_pairs, client, model_id = args
    b2_prompt = _build_tiser_prompt(unit)
    b2_raw = await _llm_call(client, model_id, b2_prompt, f"B2-tiser-t{unit.turn_idx}")
    b2 = _extract_last_json(b2_raw, ("fact", "scope")) or {}
    if not b2:
        log.warning(f"B2 parse fail for TISER turn {unit.turn_idx} — FAIL. Raw: {b2_raw[:200]}")
    return TurnVerdict(
        turn_idx=unit.turn_idx,
        fact=int(b2.get("fact", 0)),
        scope=int(b2.get("scope", 0)),
        tiser_required=int(b2.get("tiser_required", 1)),
        reasoning=str(b2.get("reasoning", "")),
        raw_output=b2_raw,
        q_factual=1,    # skipped for TISER: distractors are intentional
        q_plausible=1,  # skipped for TISER: questions come from qa_pairs
        q_reasoning="[B1 skipped for TISER turns]",
        q_raw_output="",
    )

async def _judge_one_casual(args) -> TurnVerdict:
    unit, client, model_id = args

    b1_prompt = _build_b1_prompt(unit)
    b2_prompt = _build_casual_prompt(unit)

    b1_raw, b2_raw = await asyncio.gather(
        _llm_call(client, model_id, b1_prompt, f"B1-casual-t{unit.turn_idx}"),
        _llm_call(client, model_id, b2_prompt, f"B2-casual-t{unit.turn_idx}"),
    )

    b1 = _extract_last_json(b1_raw, ("q_factual", "q_plausible")) or {}
    b2 = _extract_last_json(b2_raw, ("fact", "scope")) or {}

    if not b2:
        log.warning(f"B2 parse fail for casual turn {unit.turn_idx} — FAIL. Raw: {b2_raw[:200]}")
    if not b1:
        log.warning(f"B1 parse fail for casual turn {unit.turn_idx} — FAIL. Raw: {b1_raw[:200]}")

    return TurnVerdict(
        turn_idx=unit.turn_idx,
        fact=int(b2.get("fact", 0)),
        scope=int(b2.get("scope", 0)),
        tiser_required=0,
        reasoning=str(b2.get("reasoning", "")),
        raw_output=b2_raw,
        q_factual=int(b1.get("q_factual", 0)),
        q_plausible=int(b1.get("q_plausible", 0)),
        q_reasoning=str(b1.get("reasoning", "")),
        q_raw_output=b1_raw,
    )

# ── Structural validation (Python only) ──────────────────────────────────────

def validate_tiser_turn(turn: dict, verdict: TurnVerdict) -> Optional[str]:
    """
    Structural rules for TISER turns (B2 only — B1 skipped).
    Rules:
      1. fact=0   → reject (answer factually wrong vs qa_pairs)
      2. scope=0  → reject (off-topic or English response)
      3. tiser_required=1 but TISER tags absent → reject
    """
    gpt = turn.get("gpt", "")
    if not verdict.raw_output:
        return f"turn {verdict.turn_idx}: judge API failed to evaluate this turn (Timeout/Error)."
    if verdict.fact == 0:
        return f"turn {verdict.turn_idx}: answer factuality failed — {verdict.reasoning}"
    if verdict.scope == 0:
        return f"turn {verdict.turn_idx}: answer out of scope — {verdict.reasoning}"
    if verdict.tiser_required == 1 and not TISER_FULL_RE.search(gpt):
        return f"turn {verdict.turn_idx}: TISER tags required but absent"
    return None

def validate_casual_turn(turn: dict, verdict: TurnVerdict) -> Optional[str]:
    """
    Structural rules for casual turns (B1 + B2).
    Rules:
      1. q_factual=0  → reject (wrong premise in question)
      2. q_plausible=0 → reject (question off-topic)
      3. fact=0       → reject (answer factually wrong vs Wikipedia)
      4. scope=0      → reject (off-topic or English response)
    """
    if verdict.q_factual == 0:
        return f"turn {verdict.turn_idx}: wrong premise in question — {verdict.q_reasoning}"
    if verdict.q_plausible == 0:
        return f"turn {verdict.turn_idx}: question implausible in context — {verdict.q_reasoning}"
    if verdict.fact == 0:
        return f"turn {verdict.turn_idx}: answer factuality failed — {verdict.reasoning}"
    if verdict.scope == 0:
        return f"turn {verdict.turn_idx}: answer out of scope — {verdict.reasoning}"
    return None

# Keep backward-compatible alias used by judge_tiser_turns / judge_casual_turns
def validate_turn(turn: dict, verdict: TurnVerdict, is_tiser: bool = False) -> Optional[str]:
    return validate_tiser_turn(turn, verdict) if is_tiser else validate_casual_turn(turn, verdict)

# ── Public API ────────────────────────────────────────────────────────────────

async def judge_tiser_turns(
    turns: list,
    qa_pairs: list,
    client,
    model_id: str,
    entity: str = "",
    wiki: str = "",
    n_workers: int = 6,
) -> tuple[list, list]:
    """
    Judge TISER turns (Stage 1 judge) — B2 only.
    B1 is skipped: questions come from qa_pairs (always on-topic)
    and awareness distractors are intentional.
    Returns (accepted_turns, rejected_info).
    """
    units = _build_units(turns, qa_pairs, entity, wiki)
    items = [(u, qa_pairs, client, model_id) for u in units]
    verdicts = await asyncio.gather(*[_judge_one_tiser(item) for item in items], return_exceptions=True)
    accepted, rejected_info = [], []
    for turn, verdict in zip(turns, verdicts):
        if verdict is None:
            log.error("No verdict for TISER turn (Worker crashed) — REJECTING")
            rejected_info.append({"turn": turn, "reason": "turn evaluation worker crashed", "verdict": None})
            continue
        reason = validate_tiser_turn(turn, verdict)
        if reason:
            rejected_info.append({"turn": turn, "reason": reason, "verdict": verdict})
        else:
            accepted.append(turn)
    return accepted, rejected_info

async def judge_casual_turns(
    turns: list,
    qa_pairs: list,
    client,
    model_id: str,
    entity: str = "",
    wiki: str = "",
    n_workers: int = 6,
) -> tuple[list, list]:
    """
    Judge casual turns (Stage 2 judge) — B1 + B2 with Wikipedia reference.
    Returns (accepted_turns, rejected_info).
    """
    units = _build_units(turns, qa_pairs, entity, wiki)
    items = [(u, client, model_id) for u in units]
    verdicts = await asyncio.gather(*[_judge_one_casual(item) for item in items], return_exceptions=True)
    accepted, rejected_info = [], []
    for turn, verdict in zip(turns, verdicts):
        if verdict is None:
            log.error("No verdict for casual turn (Worker crashed) — REJECTING")
            rejected_info.append({"turn": turn, "reason": "turn evaluation worker crashed", "verdict": None})
            continue
        reason = validate_casual_turn(turn, verdict)
        if reason:
            rejected_info.append({"turn": turn, "reason": reason, "verdict": verdict})
        else:
            accepted.append(turn)
    return accepted, rejected_info
