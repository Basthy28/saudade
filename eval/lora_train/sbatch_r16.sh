#!/bin/bash
#SBATCH --job-name=saudade_lora_r16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=144:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT="${SAUDADE_ROOT:-$HOME/saudade}"
DATA_DIR="${SAUDADE_DATA:-$ROOT/data}"
export HF_HOME="${SAUDADE_SCRATCH:-$HOME/scratch}/hf_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

# rank 16: 2 epochs, LR 2e-4, alpha = r, effective batch 16 (4 x 4)
python "$ROOT/eval/lora_train/train_lora.py" \
  --rank 16 --batch 4 --accum 4 \
  --train-file "$DATA_DIR/08_sft/ready_to_train/train.jsonl" \
  --val-file "$DATA_DIR/08_sft/ready_to_train/val.jsonl"
