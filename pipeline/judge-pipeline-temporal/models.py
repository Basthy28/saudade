# models.py
# API wrappers: model under evaluation + Gemma 4 judge.
# Judge scores 3 criteria independently (no mean): accuracy, completeness, temporal.

import json
import logging
import re
import time

from openai import OpenAI

from config import JUDGE, JUDGE_FALLBACK, MAX_RETRIES, REQUEST_TIMEOUT, TEMPERATURE, OE_MAX_TOKENS

logger = logging.getLogger(__name__)

# ── Client cache — one client per base_url ────────────────────────────────────
_clients: dict[str, OpenAI] = {}


def _get_client(base_url: str, api_key: str) -> OpenAI:
    if base_url not in _clients:
        _clients[base_url] = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=REQUEST_TIMEOUT,
        )
    return _clients[base_url]


# ── Qwen3 thinking-block handling ────────────────────────────────────────────
# Safety net: if a <think>...</think> block survives (template not applied,
# server ignores chat_template_kwargs, model ignores the flag), drop it so the
# scored answer is the final answer and not a truncated reasoning trace.
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def _strip_think(text: str) -> str:
    if not text:
        return text
    return _THINK_RE.sub("", text).strip()

# ── Model under evaluation ────────────────────────────────────────────────────
def call_model(model_cfg: dict, prompt: str) -> str | None:
    """Call the model under evaluation. Returns response text or None on failure."""
    client = _get_client(model_cfg["base_url"], model_cfg["api_key"])
    for attempt in range(MAX_RETRIES):
        try:
            extra = {}
            if model_cfg.get("no_thinking"):
                extra["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            resp = client.chat.completions.create(
                model=model_cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=OE_MAX_TOKENS,
                **extra,
            )
            return _strip_think(resp.choices[0].message.content.strip())
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Model call failed (attempt {attempt+1}/{MAX_RETRIES}): {e} — retrying in {wait}s")
            time.sleep(wait)
    logger.error("Model call failed after all retries")
    return None


# ── Gemma 4 judge ─────────────────────────────────────────────────────────────
# Scores 3 criteria, each 1–5, independently. No aggregation.

_JUDGE_SYSTEM = """És um avaliador especializado em raciocínio temporal sobre entidades portuguesas.
A tua tarefa é avaliar a resposta de um modelo de linguagem a uma pergunta de raciocínio temporal em português europeu.
Recebes a pergunta, a resposta de referência (correta) e a resposta do modelo.
Avalia com base em 3 critérios independentes, cada um de 1 a 5.

CRITÉRIOS:
1. accuracy (precisão factual): A resposta do modelo é factualmente correta em relação à referência?
   5 = totalmente correta, sem erros factuais
   4 = maioritariamente correta, erro menor ou imprecisão
   3 = parcialmente correta, alguns elementos certos mas outros errados
   2 = maioritariamente incorreta, poucos elementos certos
   1 = totalmente incorreta ou completamente fora do tema

2. completeness (completude): A resposta cobre os pontos principais presentes na referência?
   5 = cobre todos os pontos essenciais da referência
   4 = cobre a maioria dos pontos, falta apenas algum detalhe secundário
   3 = cobre cerca de metade dos pontos essenciais
   2 = cobre apenas um ponto menor, falta o essencial
   1 = não cobre nenhum ponto relevante da referência

3. temporal (raciocínio temporal): A resposta demonstra compreensão da ordem e contexto temporal dos eventos?
   5 = Excelente: sequência de eventos correta e datas precisas quando mencionadas
   4 = Boa: a ordem/sequência dos eventos está correta; datas têm imprecisões menores mas não comprometem a cronologia
   3 = Parcial: compreende parcialmente a cronologia mas comete erros em algumas relações temporais ou na ordenação
   2 = Fraca: erros significativos na sequência dos eventos, anacronismos claros, ou confunde o que ocorreu antes e depois
   1 = Nula: ignora completamente a dimensão temporal, sequência totalmente errada, ou resposta completamente atemporal

INSTRUÇÕES:
- Raciocina brevemente antes de atribuir cada pontuação
- Avalia cada critério de forma independente
- Responde APENAS com JSON válido no formato especificado, sem texto adicional"""

_JUDGE_USER = """Pergunta: {question}

Resposta de referência: {reference}

Resposta do modelo: {model_answer}

Avalia e responde com:
{{
  "reasoning": "raciocínio breve sobre cada critério",
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "temporal": <1-5>
}}"""

_FALLBACK_SCORES = {"accuracy": 0.0, "completeness": 0.0, "temporal": 0.0}
_criteria = ("accuracy", "completeness", "temporal")
_using_judge_fallback = False


def call_judge(question: str, reference: str, model_answer: str) -> dict:
    """
    Call the Gemma 4 judge. Returns accuracy, completeness and temporal scores
    (each 1-5). Falls back to JUDGE_FALLBACK on failure and returns zeros when
    both judges fail.
    """
    global _using_judge_fallback

    if not model_answer.strip():
        logger.warning("Empty model answer — returning zero scores")
        return _FALLBACK_SCORES.copy()

    prompt = _JUDGE_USER.format(
        question=question,
        reference=reference,
        model_answer=model_answer,
    )

    for i, judge_cfg in enumerate([JUDGE, JUDGE_FALLBACK]):
        if i == 1 and not _using_judge_fallback:
            logger.warning("Judge primary failed — switching to fallback")
            _using_judge_fallback = True

        for attempt in range(MAX_RETRIES):
            try:
                client = _get_client(judge_cfg["base_url"], judge_cfg["api_key"])
                resp = client.chat.completions.create(
                    model=judge_cfg["model"],
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                raw = resp.choices[0].message.content.strip()

                # Strip markdown fences if model wraps JSON in ```
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

                data = json.loads(raw)

                scores = {}
                for key in _criteria:
                    val = float(data.get(key, 1))
                    scores[key] = max(1.0, min(5.0, val))

                return scores

            except json.JSONDecodeError as e:
                logger.warning(f"Judge JSON parse error (attempt {attempt+1}): {e} | raw: {raw[:200]!r}")
            except Exception as e:
                logger.warning(f"Judge call failed (attempt {attempt+1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)

    logger.error("Judge failed after all retries — returning zero scores")
    return _FALLBACK_SCORES.copy()
