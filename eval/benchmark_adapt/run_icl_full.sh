#!/bin/bash
# ICL full: saudade_chat_icl_pt, Qwen2.5-7B, full 7435 items.
# Usage: sbatch run_icl_full.sh <num_fewshot>
#SBATCH --job-name=icl_full_%j
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=epor32
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00

set -eo pipefail

MODEL_PATH="${2:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${SAUDADE_WORK:-$HOME/saudade-work}/eval/icl_pilot"
TASKS="saudade_chat_icl_pt"
N_SHOT="${1:-3}"
BENCHMARK_DIR="${SAUDADE_ROOT:-$HOME/saudade}/eval/benchmark_adapt"
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
echo "Tasks  : $TASKS (${N_SHOT}-shot)"
echo "Start  : $(date)"

lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL_PATH},max_model_len=8192,gpu_memory_utilization=0.8,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" \
    --include_path "${BENCHMARK_DIR}" \
    --num_fewshot "$N_SHOT" \
    --apply_chat_template \
    --batch_size auto \
    --log_samples \
    --output_path "${OUTPUT_DIR}" \
    --seed 42 \
    --trust_remote_code

echo "Done at $(date)"
