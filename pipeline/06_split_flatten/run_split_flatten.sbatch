#!/bin/bash
#SBATCH --job-name=split_flatten
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --account=epor32
#SBATCH --qos=gp_debug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source /opt/Anaconda-2021.05/etc/profile.d/conda.sh || source /apps/ACC/ANACONDA/2023.07/etc/profile.d/conda.sh
conda activate vllm_new

export HF_HOME=${SAUDADE_SCRATCH:-$HOME/scratch}/.cache/huggingface

PIPELINE_DIR="${SAUDADE_ROOT:-$HOME/saudade}/pipeline"
SPLIT_DIR="${SAUDADE_DATA:-${SAUDADE_ROOT:-$HOME/saudade}/data}/06_split"

echo "=========================================="
echo " SAUDADE split + flatten"
echo " Date: $(date)"
echo "=========================================="

echo "[$(date)] Splitting datasets (curated → train/test)..."
python3 "${PIPELINE_DIR}/06_split_flatten/split_curated.py"
echo "Split complete."

echo "[$(date)] Flattening test sets (with distractor context injection for awareness)..."
python3 "${PIPELINE_DIR}/06_split_flatten/flatten_test_dataset.py"

echo "[$(date)] Split + Flatten complete."
echo "  -> Awareness test flat: $(wc -l < ${SPLIT_DIR}/saudade_awareness_pt_test_flat.jsonl 2>/dev/null || echo 0) questions"
echo "  -> Reasoning test flat: $(wc -l < ${SPLIT_DIR}/saudade_reasoning_pt_test_flat.jsonl 2>/dev/null || echo 0) questions"
echo "  -> Cognition test flat: $(wc -l < ${SPLIT_DIR}/saudade_cognition_pt_test_flat.jsonl 2>/dev/null || echo 0) questions"
