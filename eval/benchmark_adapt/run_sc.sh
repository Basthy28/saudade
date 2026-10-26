#!/bin/bash
# Self-consistency probe/full: saudade_chat_cot_sc_pt (temp 0.7).
# Usage: sbatch run_sc.sh <model_path> <seed> [limit]
#SBATCH --job-name=sc_probe_%j
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=epor32
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00

set -eo pipefail

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
SEED="${2:-1}"
OUTPUT_DIR="${SAUDADE_WORK:-$HOME/saudade-work}/eval/sc_pilot"
TASKS="saudade_chat_cot_sc_pt"
BENCHMARK_DIR="${SAUDADE_ROOT:-$HOME/saudade}/eval/benchmark_adapt"
LIMIT_ARGS=""
if [ -n "${3:-}" ]; then
  LIMIT_ARGS="--limit $3"
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

echo "Model  : $MODEL_PATH seed $SEED $LIMIT_ARGS"
echo "Start  : $(date)"

lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL_PATH},max_model_len=8192,gpu_memory_utilization=0.8,dtype=bfloat16,trust_remote_code=True,seed=${SEED}" \
    --tasks "$TASKS" \
    --include_path "${BENCHMARK_DIR}" \
    --apply_chat_template \
    --batch_size auto \
    $LIMIT_ARGS \
    --log_samples \
    --output_path "${OUTPUT_DIR}/${SEED}" \
    --seed "$SEED" \
    --trust_remote_code

echo "Done at $(date)"
