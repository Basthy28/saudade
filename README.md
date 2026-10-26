# SAUDADE

Code for the **SAUDADE** temporal reasoning benchmark and conversational SFT
corpus for European Portuguese, released with the accompanying ECIR 2027
resource paper and the MSc dissertation that documents the construction and
evaluation in full.

This repository contains the construction pipeline, the evaluation harness,
the LoRA training/merge pipeline and the scripts that verify the reported
numbers.

## Resources

| Resource | Location |
|---|---|
| Benchmark (18,074 PT items + English mirror) | [`amalia-llm/saudade-pt`](https://huggingface.co/datasets/amalia-llm/saudade-pt) |
| SFT corpus (multi-turn Tiser/Clean + single-turn) | [`amalia-llm/SAUDADE-SFT`](https://huggingface.co/datasets/amalia-llm/SAUDADE-SFT) |
| Merged LoRA checkpoint (r32) | [`amalia-llm/saudade-9b-sft`](https://huggingface.co/amalia-llm/saudade-9b-sft) |

The datasets are not vendored here. Download them into `data/` (or point
`SAUDADE_DATA` at the directory that holds them) before running the harness.

## Layout

```
pipeline/01_extraction           entity collection, ranking, origin classification
pipeline/02_distillation         temporal fact distillation (Qwen3-235B)
pipeline/03_purification         hallucination filter (Gemma 4 judge)
pipeline/04_generation           candidate question generation for the three dimensions
pipeline/05_curation             multi-criterion curation (keep / modify / remove)
pipeline/06_split_flatten        80/20 fact-pair split + flattening
pipeline/07_openended            open-ended generation and judging
pipeline/08_sft                  dialogue generation, single-turn conversion, filtering
pipeline/judge-pipeline-temporal open-ended evaluation judge (Gemma 4-31B)
eval/benchmark_adapt             closed-choice, CoT, ICL, date-first, self-consistency,
                                 English mirror and prompt templates (YAML)
eval/lora_train                  LoRA training (r16/r32) and adapter merge
canonical-results                canonical results package and verification scripts
PIPELINE.md                      stage-by-stage map and run instructions
```

## Installation

```bash
python -m pip install -r requirements.txt
```

Training and generation additionally require a CUDA environment with `vllm`,
`peft` and `trl`; the pipeline scripts assume an OpenAI-compatible server
(vLLM) and read its address from the environment.

## Reproducing the evaluation

```bash
# 1. Put the released files under data/
#    data/saudade_pt_binary_temporal_ordering.jsonl
#    data/saudade_en_binary_temporal_ordering.jsonl
#    data/saudade_pt_fact_retrieval.jsonl
#    data/saudade_pt_relation_with_distractors.jsonl

# 2. Build the lm-eval view of the binary track
python eval/benchmark_adapt/prepare_binary_track.py --lang pt
python eval/benchmark_adapt/prepare_binary_track.py --lang en

# 3. Run a model (example: released adapter, standard prompt)
lm_eval --model hf \
  --model_args pretrained=amalia-llm/saudade-9b-sft \
  --tasks saudade_chat_pt \
  --include_path eval/benchmark_adapt \
  --apply_chat_template \
  --output_path results/

# 4. Verify the numbers reported in the paper and thesis
python canonical-results/verify_numbers.py
```

The `saudade_chat_*` tasks use the standard prompt, `saudade_chat_cot_*` the
chain-of-thought prompt, `saudade_chat_icl_*` the in-context regimes, and
`saudade_chat_datefirst_pt` the date-first control.

## Environment variables

The scripts do not contain machine-specific paths. Override these when your
layout differs from the defaults:

| Variable | Default | Purpose |
|---|---|---|
| `SAUDADE_ROOT` | repository root | code location |
| `SAUDADE_DATA` | `$SAUDADE_ROOT/data` | corpora and intermediate artefacts |
| `SAUDADE_WORK` | `$HOME/saudade-work` | evaluation outputs and logs |
| `SAUDADE_SCRATCH` | `$HOME/scratch` | caches (HF, vLLM, Ray) |
| `SAUDADE_ENV` | `$HOME/venvs/saudade` | Python environment used by the sbatch scripts |
| `MODEL_ROOT` | `$HOME/models` | local model directories, when not using Hugging Face ids |
| `OE_RESULTS_DIR` | `$SAUDADE_DATA/07_openended/results` | judge output directory |
| `OE_LANG=en` | `pt` | run the open-ended judge on the English mirror |

## Labels used in the code

| label | task in the thesis / paper |
|---|---|
| `awareness` | Temporal Relation with Distractors |
| `cognition` | Temporal Fact Retrieval |
| `reasoning` | Binary Temporal Ordering |

## Licence

Code and data are released under **Apache-2.0** (`LICENSE`). The merged
weights follow the terms of the base model, AMALIA-9B-SFT.
