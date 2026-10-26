# Benchmark evaluation

Evaluation harness for the SAUDADE benchmark. Each item presents two dated
events about the same entity and asks which occurred first. Models answer with
one of the two options (A or B).

Data: [`amalia-llm/saudade-pt`](https://huggingface.co/datasets/amalia-llm/saudade-pt).

## Files

| File | Purpose |
|---|---|
| `prepare_binary_track.py` | adds the `prompt` prefix and `answer_index` expected by lm-eval |
| `similarity_sampler.py` | similarity-based few-shot sampler for the ICL regime |
| `translate_pool_en.py` | translates the ICL training pool to English |
| `utils.py` | shared prompt/answer helpers |
| `saudade_*.yaml` | lm-eval task definitions |
| `run_*.sh` | Slurm runners for one model and one task |
| `submit_all_closed_en.sh` | submits the English closed-choice campaign |

## Data preparation

Download the benchmark files into `data/` (or set `SAUDADE_DATA`):

```
data/saudade_pt_binary_temporal_ordering.jsonl
data/saudade_en_binary_temporal_ordering.jsonl
data/saudade_pt_fact_retrieval.jsonl
data/saudade_pt_relation_with_distractors.jsonl
```

Then build the lm-eval views:

```bash
python prepare_binary_track.py --lang pt   # -> data/eval/saudade_pt_eval.jsonl
python prepare_binary_track.py --lang en   # -> data/eval/saudade_en_eval.jsonl
```

The ICL tasks additionally expect the training pools
`data/icl_pool_train.jsonl` (PT) and `data/icl_pool_train_en.jsonl` (EN). They
are built from the training partition; `translate_pool_en.py` produces the
English pool from the Portuguese one.

## Tasks

| Task | Regime |
|---|---|
| `saudade_chat_pt` | standard chat prompt |
| `saudade_chat_cot_pt` | chain-of-thought prompt |
| `saudade_chat_cot_sc_pt` | CoT with self-consistency (5 samples) |
| `saudade_chat_datefirst_pt` | date-first control |
| `saudade_chat_icl_pt` | in-context, random exemplars |
| `saudade_chat_icl_sim_pt` | in-context, similarity-selected exemplars |
| `saudade_chat_icl_xent_pt` | in-context, cross-entity exemplars |
| `saudade_chat_icl_pt_enscaf` | in-context, ENSCAF setup |
| `saudade_mcq_pt` | multiple-choice view of the binary track |
| `saudade_chat_en`, `saudade_chat_cot_en`, `saudade_chat_icl_en`, `saudade_mcq_en` | English mirror |

## Running a model

```bash
lm_eval --model hf \
  --model_args pretrained=amalia-llm/AMALIA-9B-0626-SFT \
  --tasks saudade_chat_pt,saudade_chat_cot_pt \
  --include_path . \
  --apply_chat_template \
  --output_path results/
```

On Slurm:

```bash
sbatch run_eval_closed_ended_en.sh <model_id_or_path>
sbatch run_icl_full.sh <task> <model_id_or_path> [num_fewshot]
```

`run_extra.sh` is the generic runner used by the ICL and self-consistency
experiments; `run_translate_pool.sh` builds the English ICL pool.

## Outputs and scoring

Results are written as lm-eval JSON under the output directory. Accuracy is
computed over the 7,435 scored items of the Portuguese binary track (9 of the
7,444 raw items have no extractable answer) and over the 7,435-item English
mirror. The canonical per-model results and the verification script are in
`canonical-results/`.
