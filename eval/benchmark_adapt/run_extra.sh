#!/bin/bash
# Generic ICL/DF runner with extra model args (thinking off, tokenizer override).
# Usage: sbatch run_extra.sh <task> <model> <extra_args> [num_fewshot] [limit] [tag]
#SBATCH --job-name=extra_%j
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=epor32
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00

set -eo pipefail

TASKS="${1:?task}"
MODEL_PATH="${2:?model}"
EXTRA="${3:-}"
N_SHOT="${4:-0}"
BENCHMARK_DIR="${SAUDADE_ROOT:-$HOME/saudade}/eval/benchmark_adapt"
TAG="${6:-extra}"
OUTPUT_DIR="${SAUDADE_WORK:-$HOME/saudade-work}/eval/${TAG}"
LIMIT_ARGS=""
if [ -n "${5:-}" ]; then
  LIMIT_ARGS="--limit $5"
fi
FEWSHOT_ARGS=""
if [ "$N_SHOT" != "0" ]; then
  FEWSHOT_ARGS="--num_fewshot $N_SHOT"
fi
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

if [ -f "${SAUDADE_ENV:-$HOME/venvs/saudade}/bin/activate" ]; then
    source "${SAUDADE_ENV:-$HOME/venvs/saudade}/bin/activate"
fi

export HF_HOME="${SAUDADE_SCRATCH:-$HOME/scratch}/.cache/huggingface"
export HF_HUB_OFFLINE="True"
export HF_DATASETS_OFFLINE="True"
export PYTHONPATH="${PYTHONPATH:-}:${BENCHMARK_DIR}"

echo "Model  : $MODEL_PATH"
echo "Tasks  : $TASKS fewshot $N_SHOT $LIMIT_ARGS"
echo "Start  : $(date)"

lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL_PATH},max_model_len=8192,gpu_memory_utilization=0.8,dtype=bfloat16,trust_remote_code=True${EXTRA}" \
    --tasks "$TASKS" \
    --include_path "${BENCHMARK_DIR}" \
    --apply_chat_template \
    --batch_size auto \
    $FEWSHOT_ARGS \
    $LIMIT_ARGS \
    --log_samples \
    --output_path "${OUTPUT_DIR}" \
    --seed 42 \
    --trust_remote_code

echo "Done at $(date)"
