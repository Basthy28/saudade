#!/bin/bash
#SBATCH --job-name=saudade_merge
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# CPU-only merge: adapter -> standalone model.
ROOT="${SAUDADE_ROOT:-$HOME/saudade}"
export HF_HOME="${SAUDADE_SCRATCH:-$HOME/scratch}/hf_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

for rank in 16 32; do
  echo "=== $(date) merge r${rank} ==="
  python "$ROOT/eval/lora_train/merge_lora.py" \
    --adapter "$ROOT/outputs/r${rank}" \
    --out "$ROOT/outputs/r${rank}-merged"
done

echo "=== $(date) merge complete ==="
ls -la "$ROOT/outputs/r16-merged" "$ROOT/outputs/r32-merged"
