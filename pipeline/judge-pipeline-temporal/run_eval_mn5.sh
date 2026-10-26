#!/bin/bash
# =============================================================================
# saudade OE Evaluation — MareNostrum 5
# =============================================================================
# Evaluates ONE model per job. Submit one job per model.
#
# HOW TO USE:
#   sbatch run_eval_mn5.sh <model_key>
#
# Examples:
#   sbatch run_eval_mn5.sh salamandra_7b
#   sbatch run_eval_mn5.sh eurollm_9b
#   sbatch run_eval_mn5.sh amalia_sft
#
# Model keys are defined in config.py MODELS dict.
# To see all available keys: python run_evaluation.py --list-models
#
# To test before a full run (3 items per domain):
#   Edit run_evaluation.py call below and add --limit 3
#
# GPU layout (4x H100 per node):
#   GPU 0-1 — model under evaluation (port 8765)
#   GPU 2-3 — Gemma 4 31B FP8 judge  (port 8766)
# =============================================================================

#SBATCH --job-name=saudade_oe_%j
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=${SLURM_ACCOUNT:-epor32}
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=04:00:00

set -euo pipefail

# =============================================================================
# CHECK ARGUMENT
# =============================================================================
MODEL_KEY="${1:-}"
if [ -z "$MODEL_KEY" ]; then
    echo "ERROR: No model key provided."
    echo "Usage: sbatch run_eval_mn5.sh <model_key> [extra args]"
    echo "Run 'python run_evaluation.py --list-models' to see all keys."
    exit 1
fi
EXTRA_ARGS="${@:2}"  # extra arguments for run_evaluation.py

# =============================================================================
# ENVIRONMENT
# =============================================================================
EVAL_DIR="${SAUDADE_ROOT:-$HOME/saudade}/pipeline/judge-pipeline-temporal"
LOGDIR="${SAUDADE_WORK:-$HOME/saudade-work}/eval/logs"
mkdir -p "$LOGDIR"

GEMMA_JUDGE_PATH="RedHatAI/gemma-4-31B-it-FP8-block"
JUDGE_PORT=8766
EVAL_PORT=8765

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

cd "$EVAL_DIR"

echo "======================================================="
echo "saudade OE Evaluation — MareNostrum 5"
echo "Model key  : $MODEL_KEY"
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : $(hostname)"
echo "GPUs       : $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' '|' | sed 's/|$//')"
echo "Start      : $(date)"
echo "======================================================="

# =============================================================================
# HELPER — wait for vLLM to be ready on a port
# =============================================================================
wait_for_vllm() {
    local port=$1
    local logfile=$2
    local label=$3
    local pid=$4
    local max=600 elapsed=0
    echo "Waiting for $label on port $port..."
    until $PYTHON -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', ${port})); s.close()" 2>/dev/null; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $label process died. Last 20 log lines:"
            tail -20 "$logfile"
            exit 1
        fi
        sleep 10; elapsed=$((elapsed + 10))
        echo "  ...${elapsed}s"
        if [ $elapsed -ge $max ]; then
            echo "ERROR: $label did not start within ${max}s. Last 20 log lines:"
            tail -20 "$logfile"
            exit 1
        fi
    done
    echo "$label ready after ${elapsed}s"
}

# =============================================================================
# PHASE 1 — Start Gemma 4 judge on GPUs 2-3, port 8766
# Stays up for the entire job duration.
# =============================================================================
echo ""
echo "--- [1/3] Starting Gemma 4 judge (GPUs 2-3, port $JUDGE_PORT) ---"

JUDGE_LOG="$LOGDIR/judge_${MODEL_KEY}_${SLURM_JOB_ID}.log"

CUDA_VISIBLE_DEVICES=2,3 $VLLM_BIN \
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

wait_for_vllm $JUDGE_PORT "$JUDGE_LOG" "Gemma 4 judge" $JUDGE_PID

# =============================================================================
# PHASE 2 — Start model under evaluation on GPUs 0-1, port 8765
# =============================================================================
echo ""
echo "--- [2/3] Starting eval model: $MODEL_KEY (GPUs 0-1, port $EVAL_PORT) ---"

# Resolve model path from config.py
MODEL_PATH=$($PYTHON - <<EOF
import sys
sys.path.insert(0, "$EVAL_DIR")
from config import MODELS
key = "$MODEL_KEY"
if key not in MODELS:
    print(f"ERROR: '{key}' not found in config.py MODELS", file=sys.stderr)
    sys.exit(1)
cfg = MODELS[key]
if "path" not in cfg:
    print(f"ERROR: '{key}' has no 'path' in config.py", file=sys.stderr)
    sys.exit(1)
print(cfg["path"])
EOF
)

echo "Path: $MODEL_PATH"

EVAL_LOG="$LOGDIR/eval_model_${MODEL_KEY}_${SLURM_JOB_ID}.log"

CUDA_VISIBLE_DEVICES=0,1 $VLLM_BIN \
    --model "$MODEL_PATH" \
    --served-model-name "eval-model" \
    --port $EVAL_PORT \
    --max-model-len 4096 \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    > "$EVAL_LOG" 2>&1 &
EVAL_PID=$!

wait_for_vllm $EVAL_PORT "$EVAL_LOG" "eval model ($MODEL_KEY)" $EVAL_PID

# =============================================================================
# PHASE 3 — Run evaluation
# Remove --limit once you've verified the pipeline works end-to-end.
# =============================================================================
echo ""
echo "--- [3/3] Running OE evaluation for: $MODEL_KEY ---"

# For a smoke test add --limit 3.
$PYTHON run_evaluation.py --model "$MODEL_KEY" $EXTRA_ARGS

# =============================================================================
# CLEANUP
# =============================================================================
echo ""
echo "--- Shutting down vLLM servers ---"
kill $EVAL_PID  2>/dev/null || true
kill $JUDGE_PID 2>/dev/null || true
wait $EVAL_PID  2>/dev/null || true
wait $JUDGE_PID 2>/dev/null || true

echo ""
echo "======================================================="
echo "DONE : $MODEL_KEY"
echo "End  : $(date)"
echo "Results in: ${SAUDADE_DATA:-${SAUDADE_ROOT:-$HOME/saudade}/data}/07_openended/results/"
echo "======================================================="
