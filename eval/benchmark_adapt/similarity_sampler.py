"""Similarity few-shot sampler for SAUDADE ICL (thesis §5.3, similarity regime).
Plugs into lm-eval via fewshot_config: {sampler: !function similarity_sampler.SimilaritySampler}.
"""
from pathlib import Path

import json
import os
import re
import unicodedata

from lm_eval.api.samplers import ContextSampler

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

NBR_KEY = os.environ.get("ICL_NBR_KEY", "neighbors")

POOL_PATH = str(DATA / "icl_pool_train.jsonl")
NBR_PATH = str(DATA / "icl_neighbors.json")

_POOL = None
_NBR = None

def _norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()

def _load():
    global _POOL, _NBR
    if _POOL is None:
        _POOL = [json.loads(l) for l in open(POOL_PATH, encoding="utf-8")]
        _NBR = json.load(open(NBR_PATH, encoding="utf-8"))[NBR_KEY]
    return _POOL, _NBR

class SimilaritySampler(ContextSampler):
    """Top-n train-pool neighbors (mpnet cosine on entity+question) per eval doc."""

    def __init__(self, df=None, **kwargs):
        super().__init__(df, **kwargs)
        self.df = df or []

    def fewshot_docs(self):
        pool, _ = _load()
        return pool

    def sample(self, n, eval_doc=None, df=None, **kwargs):
        assert n >= 0, "n must be >= 0"
        if n == 0:
            return []
        pool, nbr = _load()
        if eval_doc is not None:
            k = _norm(eval_doc.get("entity")) + " || " + _norm(eval_doc.get("question"))
            idx = nbr.get(k)
            if idx is not None:
                res = [pool[i] for i in idx[:n]]
                assert len(res) == n, (len(res), n)
                return res
        res = list(pool[:n])
        assert len(res) == n, (len(res), n)
        return res
