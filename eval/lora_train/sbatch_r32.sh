#!/bin/bash
#SBATCH --job-name=saudade_lora_r32
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

# rank 32: 2 epochs, LR 1e-4, gradient clipping 0.5, alpha = r
python "$ROOT/eval/lora_train/train_lora.py" \
  --rank 32 --batch 4 --accum 4 --lr 1e-4 --max-grad-norm 0.5 \
  --train-file "$DATA_DIR/08_sft/ready_to_train/train.jsonl" \
  --val-file "$DATA_DIR/08_sft/ready_to_train/val.jsonl"
