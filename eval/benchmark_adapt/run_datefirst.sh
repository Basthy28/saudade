#!/bin/bash
# Date-first pilot: saudade_chat_datefirst_pt.
# Usage: sbatch run_datefirst.sh <model_path> [limit]
#SBATCH --job-name=datefirst_%j
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=epor32
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00

set -eo pipefail

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${SAUDADE_WORK:-$HOME/saudade-work}/eval/datefirst_pilot"
TASKS="saudade_chat_datefirst_pt"
BENCHMARK_DIR="${SAUDADE_ROOT:-$HOME/saudade}/eval/benchmark_adapt"
LIMIT_ARGS=""
if [ -n "${2:-}" ]; then
  LIMIT_ARGS="--limit $2"
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
echo "Tasks  : $TASKS $LIMIT_ARGS"
echo "Start  : $(date)"

lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL_PATH},max_model_len=8192,gpu_memory_utilization=0.8,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" \
    --include_path "${BENCHMARK_DIR}" \
    --apply_chat_template \
    --batch_size auto \
    $LIMIT_ARGS \
    --log_samples \
    --output_path "${OUTPUT_DIR}" \
    --seed 42 \
    --trust_remote_code

echo "Done at $(date)"
