#!/bin/bash
# =============================================================================
# saudade (EN) Closed-Ended Reasoning Evaluation — MareNostrum 5
# =============================================================================
# Evaluates models on translated closed-ended temporal reasoning (saudade_chat_en / saudade_chat_cot_en).
#
# Usage:
#   sbatch run_eval_closed_ended_en.sh <model_path_or_key> [output_dir] [tasks]
#
# Examples:
#   sbatch run_eval_closed_ended_en.sh swiss-ai/Apertus-8B-Instruct-2509
#   sbatch run_eval_closed_ended_en.sh amalia-llm/AMALIA-9B-0626-DPO
# =============================================================================

#SBATCH --job-name=reasoning_en_%j
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=epor32
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00

set -eo pipefail   # 2026-08-31: -u removido; CONDA_PREFIX unbound no activate matava o job em 5s

MODEL_PATH="${1:-swiss-ai/Apertus-8B-Instruct-2509}"
OUTPUT_DIR="${2:-${SAUDADE_WORK:-$HOME/saudade-work}/eval/closed_results_en}"
TASKS="${3:-saudade_chat_en,saudade_chat_cot_en}"

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

echo "======================================================="
echo "Temporal Reasoning (EN) — Closed-Ended Evaluation"
echo "Model  : $MODEL_PATH"
echo "Tasks  : $TASKS"
echo "Output : $OUTPUT_DIR"
echo "Start  : $(date)"
echo "======================================================="

lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL_PATH},max_model_len=4096,gpu_memory_utilization=0.8,dtype=bfloat16,think_end_token=</think>,enable_thinking=False,trust_remote_code=True" \
    --tasks "$TASKS" \
    --include_path "${BENCHMARK_DIR}" \
    --apply_chat_template \
    --batch_size auto \
    --log_samples \
    --output_path "${OUTPUT_DIR}" \
    --seed 42 \
    --trust_remote_code

echo "Evaluation completed at $(date)"
