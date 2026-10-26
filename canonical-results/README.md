# Canonical results and figures

`canonical_full_package.json` is the canonical evaluation package behind **Table 4**
and §4.3 of the paper and **Table 5.1** / **Table D.1** of the thesis (n = 7,435):
per-model Standard and Chain-of-Thought
accuracy, distance and era bins, Fine-Resolution Penalty (FRP), Historical Era Decay
(ED), bootstrap confidence intervals and within-era breakdowns for all fourteen models.

## Files

| File | Purpose |
|---|---|
| `canonical_full_package.json` | canonical per-model results (n = 7,435) |
| `verify_numbers.py` | prints the evaluation receipt: Table 4 / Table 5.1 accuracies, bin counts, FRP/ED, quadrant counts, Pearson *r* |
| `make_canonical_figures.py` | regenerates the four figures (`fig5a`–`fig5d`) |

## Usage

```bash
python verify_numbers.py
python make_canonical_figures.py     # writes ./figures/
```

Requires `numpy` (verification) and `matplotlib` (figures).

## Scope

The package covers the 7,435 binary items of the Portuguese track evaluated in the paper
and the thesis. Of the 7,444 raw items, 9 have no extractable answer and are not scored.
The English mirror
and the open-ended tracks are distributed with the benchmark dataset
([`amalia-llm/saudade-pt`](https://huggingface.co/datasets/amalia-llm/saudade-pt)).
