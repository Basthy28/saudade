#!/bin/bash
#SBATCH --job-name=saudade_judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --account=epor32
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH -o ${SAUDADE_SCRATCH:-$HOME/scratch}/openended/logs/%x-%j.out
#SBATCH -e ${SAUDADE_SCRATCH:-$HOME/scratch}/openended/logs/%x-%j.err

set -euo pipefail

source /apps/ACC/ANACONDA/2023.07/etc/profile.d/conda.sh
conda activate vllm_new

export HF_HOME=${SAUDADE_SCRATCH:-$HOME/scratch}/hf_cache
export TMPDIR=${SAUDADE_SCRATCH:-$HOME/scratch}/tmp
export TEMP=${SAUDADE_SCRATCH:-$HOME/scratch}/tmp
export NUMBA_CACHE_DIR=${SAUDADE_SCRATCH:-$HOME/scratch}/numba_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=1
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PIPELINE_DIR="${SAUDADE_ROOT:-$HOME/saudade}/pipeline"
SCRATCH_DIR="${SAUDADE_SCRATCH:-$HOME/scratch}/saudade_openended"
mkdir -p logs

GEMMA_MODEL="RedHatAI/gemma-4-31B-it-FP8-block"
JUDGE_PORT=$(( 10000 + (SLURM_JOB_ID % 50000) ))
ENDPOINT="127.0.0.1:${JUDGE_PORT}"

echo "============================================================"
echo " Open-ended judge (Fact Retrieval + Relation with Distractors)"
echo " Input:  ${SAUDADE_DATA:-${SAUDADE_ROOT:-$HOME/saudade}/data}/07_openended/"
echo " Output: same directory, *_clean.jsonl"
echo " Port:   $JUDGE_PORT | Job: $SLURM_JOB_ID | $(date)"
echo "============================================================"

# Start Gemma4 on GPU 0-1 (no --served-model-name so model ID = full path)
python -m vllm.entrypoints.openai.api_server \
    --model "$GEMMA_MODEL" \
    --tensor-parallel-size 2 \
    --port $JUDGE_PORT \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --disable-log-stats \
    --enforce-eager \
    > "${SCRATCH_DIR}/logs/gemma4_judge_${SLURM_JOB_ID}.log" 2>&1 &
JUDGE_PID=$!

# Wait using python urllib (no curl dependency)
echo "[Health] A aguardar Gemma4 em $ENDPOINT ..."
python3 -c "
import urllib.request, time, sys
ep = '$ENDPOINT'
for attempt in range(90):
    try:
        with urllib.request.urlopen(f'http://{ep}/v1/models', timeout=5) as r:
            if r.status == 200:
                print(f'[Health] {ep} pronto! (tentativa {attempt+1})', flush=True)
                sys.exit(0)
    except Exception as e:
        pass
    time.sleep(10)
print('[Health] TIMEOUT — servidor nao respondeu', flush=True)
sys.exit(1)
"

export JUDGE_MODEL_ID=$GEMMA_MODEL
export PORT=$JUDGE_PORT
export VLLM_ENDPOINTS="$ENDPOINT"
export MAX_CONCURRENT=200
export N_TEST=0

ulimit -n 65535 || true

echo "[$(date)] A correr judge..."
python3 "${PIPELINE_DIR}/07_openended/judge_openended.py"
JUDGE_EXIT=$?

kill $JUDGE_PID 2>/dev/null || true
wait $JUDGE_PID 2>/dev/null || true

echo ""
echo "[$(date)] Judge complete (exit=$JUDGE_EXIT)."
echo "Output: ${SAUDADE_DATA:-${SAUDADE_ROOT:-$HOME/saudade}/data}/07_openended/"
