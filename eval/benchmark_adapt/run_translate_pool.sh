#!/bin/bash
# Translate the ICL train pool to English with the Gemma 4 judge.
# Serves the judge (2 GPUs) and runs translate_pool_en.py on the same node.
# Writes ONLY new files (icl_pool_train_en.jsonl, pool_en_done.txt).
# Usage: sbatch run_translate_pool.sh
#SBATCH --job-name=translate_pool
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=epor32
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=08:00:00

set -euo pipefail

BENCH_DIR="${SAUDADE_ROOT:-$HOME/saudade}/eval/benchmark_adapt"
GEMMA_JUDGE_PATH="RedHatAI/gemma-4-31B-it-FP8-block"
JUDGE_PORT=8766
LOGDIR="${SAUDADE_WORK:-$HOME/saudade-work}/eval/logs_icl"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc/13.2.0 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true

PYTHON="${SAUDADE_ENV:-$HOME/venvs/saudade}/bin/python"
VLLM_BIN="$PYTHON -m vllm.entrypoints.openai.api_server"

export HF_HOME=${MODEL_ROOT:-$HOME/models}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH=""
export LD_LIBRARY_PATH="${SAUDADE_ENV:-$HOME/venvs/saudade}/lib:${LD_LIBRARY_PATH:-}"

unset http_proxy https_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY

echo "=== Starting Gemma 4 judge (GPUs 0-1, port $JUDGE_PORT) ==="
JUDGE_LOG="$LOGDIR/judge_translate_pool_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0,1 $VLLM_BIN \
    --model "$GEMMA_JUDGE_PATH" \
    --served-model-name "gemma4-judge" \
    --port $JUDGE_PORT \
    --max-model-len 4096 \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    > "$JUDGE_LOG" 2>&1 &
JUDGE_PID=$!

echo "Waiting for judge..."
until $PYTHON -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', $JUDGE_PORT)); s.close()" 2>/dev/null; do
    if ! kill -0 $JUDGE_PID 2>/dev/null; then echo "Judge died!"; tail -20 "$JUDGE_LOG"; exit 1; fi
    sleep 10; echo "...waiting"
done
echo "Judge ready!"

echo "=== Translating ICL pool (new files only) ==="
cd "$BENCH_DIR"
$PYTHON translate_pool_en.py --workers 8

echo "=== Shutting down judge ==="
kill $JUDGE_PID 2>/dev/null || true
wait $JUDGE_PID 2>/dev/null || true
echo "DONE"
