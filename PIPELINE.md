# Pipeline

Stage-by-stage map from the Portuguese Wikipedia dump to the released
benchmark, the SFT corpus and the open-ended evaluation results.

All scripts read their inputs from `SAUDADE_DATA` (default: `data/`) and write
intermediate artefacts there. `SAUDADE_ROOT` points at the repository root.
The sbatch scripts assume MareNostrum 5 conventions (`#SBATCH`, `srun`, vLLM
servers) and can be adapted to any Slurm cluster; adjust the `--account` and
partition/QoS directives to your own site.

## 1. Entity extraction — `pipeline/01_extraction`

```
fetch_all_entities.py  ->  rank_entities.py  ->  classify_origin.py
                       ->  clean_temporal_events.py  ->  generate_final_ranking.py
```

Candidate articles come from the MediaWiki API across 58 thematic categories.
`fetch_all_entities.py` reads `lists.txt` (one MediaWiki list-page URL per line,
the 58 thematic lists) and an optional `config.yaml` with the keys
`output_dir`, `language`, `lists_file`, `rate_limit_seconds`, `num_workers`,
`list_fetch_workers`, `parallel_image_downloads`, `image_download_workers` and
`wikimedia_api_token`; when the file is absent the defaults in the script apply.
`generate_final_ranking.py` applies the cultural-origin weights used in the
paper (PT 1.00, PALOP 0.20, BR 0.15, INT 0.10), a 15-backlink floor and a 50%
per-category retention.

Output: `data/01_extraction/newdata/final_ranking.csv` and the per-entity
Wikipedia text under `data/01_extraction/newdata/temporal_relevant/`.

## 2. Fact distillation — `pipeline/02_distillation`

`distill_entities.py` prompts Qwen3-235B to extract dated facts from each
article, requiring at least four distinct years per entity.
`loop_distill.sh` resubmits the job until all entities are processed.

Output: `data/02_distillation/temporal_cleaned/`.

## 3. Purification — `pipeline/03_purification`

`check_distilled_hallucinations.py` asks a Gemma 4-31B judge whether each fact
and its dates are fully supported by the source article; `fact_purification.py`
applies the verdicts.

```
sbatch pipeline/03_purification/run_purification_gemma.sbatch
python  pipeline/03_purification/fact_purification.py
```

Output: `data/03_purified/`.

## 4. Candidate generation — `pipeline/04_generation`

Three generators, one per dimension:

| script | dimension |
|---|---|
| `generate_reasoning.py` | Binary Temporal Ordering |
| `generate_cognition.py` | Temporal Fact Retrieval |
| `generate_awareness.py` | Temporal Relation with Distractors |

`run_generation_qwen.sbatch` serves Qwen3-235B with vLLM and runs the three
generators; `run_generation_reasoning.sbatch` runs the ordering dimension alone.

Output: `data/04_generated/`.

## 5. Curation — `pipeline/05_curation`

`curate_benchmark.py` applies the multi-criterion judge (keep / modify /
remove). In the Temporal Ordering track, 300,176 distinct question variants
were submitted to curation: 71.6% were kept, 25.5% removed and 2.8% modified,
with 70 unparsable judge outputs excluded.

```
sbatch pipeline/05_curation/run_curation_gemma.sbatch
```

Output: `data/05_curation/output/`.

## 6. Split and flatten — `pipeline/06_split_flatten`

`split_curated.py` performs the 80/20 split at the fact-pair level within each
entity (seed 42). `flatten_train_dataset.py` and
`flatten_test_dataset.py` produce the flat JSONL views used downstream.

```
sbatch pipeline/06_split_flatten/run_split_flatten.sbatch
```

Output: `data/06_split/`.

## 7. Open-ended tracks — `pipeline/07_openended`

`generate_openended.py` generates the Fact Retrieval (3,244 items) and
Relation with Distractors (7,386 items) questions; `judge_openended.py`
filters them with the Gemma 4 judge.

```
sbatch pipeline/07_openended/run_openended.sbatch
sbatch pipeline/07_openended/run_judge.sbatch
```

Output: `data/07_openended/`.

## 8. SFT corpus — `pipeline/08_sft`

`generate_dialog_sft.py` builds dialogues for seven personas, with the
structured Tiser reasoning format and the casual Clean variant; `sft_judge.py`
scores every turn. The generated total is 292,342 dialogues (292,338 Tiser,
292,258 Clean after filtering). `convert_sharegpt.py`, `convert_single_turn.py`
and `filter_freguesia.py` produce the released partitions.

```
sbatch pipeline/08_sft/run_dialog_sft.sbatch
sbatch pipeline/08_sft/run_awareness_single_turn.sbatch
```

Output: `data/08_sft/output/`, then `data/08_sft/ready_to_train/` with
290,797 train and 1,461 validation dialogues, and the single-turn partitions
(161,329 Fact Retrieval / 179,554 Relation with Distractors).

## Evaluation — `eval/benchmark_adapt`

`prepare_binary_track.py` adds the prompt prefix and answer index required by
lm-eval. The YAML task files cover the standard, CoT, date-first,
self-consistency, in-context (random / similarity / cross-entity / ENSCAF) and
English-mirror regimes. `run_*.sh` submit a single model; `submit_all_closed_en.sh`
submits the full English closed-choice campaign.

Open-ended evaluation lives in `pipeline/judge-pipeline-temporal/`:
`config.py` holds the model registry (the fourteen benchmark models plus the
r16/r32 adapters) and the two open-ended tracks; `run_evaluation.py` prompts
each model and judges the answers with Gemma 4-31B.

## Randomness

- 80/20 split: seed **42** (`split_curated.py`).
- SFT data preparation and shuffling: seed **42** (`prep_data.py`).
- Self-consistency decoding: seeds **1–5** (`saudade_chat_cot_sc_pt`).

Generation temperature is fixed at 0.4 for question generation and 0.0 for
judging; no other stochastic components affect the released artefacts.

## Verification

```bash
python canonical-results/verify_numbers.py
python canonical-results/make_canonical_figures.py
```

`verify_numbers.py` checks the reported accuracies, bins, FRP/ED and
correlations against `canonical_full_package.json`.
