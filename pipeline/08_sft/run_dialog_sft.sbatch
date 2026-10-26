#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --nodes=25
#SBATCH --ntasks=25
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=72:00:00
#SBATCH --account=epor32
#SBATCH --partition=ehpc
#SBATCH --qos=acc_ehpc
#SBATCH --signal=B:USR1@900
#SBATCH -o ${SAUDADE_SCRATCH:-$HOME/scratch}/sft_run/logs/%x-%j.out
#SBATCH -e ${SAUDADE_SCRATCH:-$HOME/scratch}/sft_run/logs/%x-%j.err

set -euo pipefail

source /apps/ACC/ANACONDA/2023.07/etc/profile.d/conda.sh
conda activate vllm_new

export HF_HOME=${SAUDADE_SCRATCH:-$HOME/scratch}/hf_cache
export TMPDIR=${SAUDADE_SCRATCH:-$HOME/scratch}/tmp
export TEMP=${SAUDADE_SCRATCH:-$HOME/scratch}/tmp
export NUMBA_CACHE_DIR=${SAUDADE_SCRATCH:-$HOME/scratch}/numba_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCH_DISTRIBUTED_TIMEOUT=1800
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

PIPELINE_DIR="${SAUDADE_ROOT:-$HOME/saudade}/pipeline"
export PIPELINE_DIR
SCRATCH_DIR="${SAUDADE_SCRATCH:-$HOME/scratch}/sft_run"
export SCRATCH_DIR
mkdir -p logs

# ── Iteration control ─────────────────────────────────────────────────────────
export CATEGORY_FILTER=${CATEGORY_FILTER:-}
export DATASET_FILTER=${DATASET_FILTER:-}
export N_TEST=${N_TEST:-0}
export MAX_CONCURRENT=${MAX_CONCURRENT:-3800}
export JUDGE_WORKERS=${JUDGE_WORKERS:-25000}
export MAX_TOKENS=4096

export OUTPUT_DIR=${OUTPUT_DIR:-${SAUDADE_DATA:-${SAUDADE_ROOT:-$HOME/saudade}/data}/08_sft/output}
export CHECKPT_DIR=${CHECKPT_DIR:-${SCRATCH_DIR}/checkpoints}

SELF="$(realpath "$0")"
PYTHON_PID=""
declare -a VLLM_PIDS=()

echo "============================================================"
echo " SFT v6 fixwiki — MN5 Model Swapping Architecture (72H)"
echo " Nodes:   ${SLURM_NNODES}"
echo " Date:    $(date)"
echo " Job:     $SLURM_JOB_ID"
echo "============================================================"

# ── Auto-resubmit on timeout (SLURM sends USR1 5min before wall time) ─────────
timeout_handler() {
    echo "[$(date)] [SIGNAL] Timeout approaching — signalling Python to checkpoint..."
    [ -n "$PYTHON_PID" ] && kill -USR1 "$PYTHON_PID" 2>/dev/null
    sleep 840
    echo "[$(date)] [SIGNAL] Exiting bash wrapper. External loop will handle resubmission."
}
trap 'timeout_handler' USR1

# ── Cluster Management Functions ─────────────────────────────────────────────

start_vllm_cluster() {
    local MODEL_PATH=$1
    local TP=$2
    local PORT=$3
    local EXTRA_ARGS=$4
    
    echo "[Launcher] Starting $MODEL_PATH across $SLURM_NNODES nodes (TP=$TP)..."
    
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)
    endpoints=""
    VLLM_PIDS=()

    my_node=$(hostname)
    for node in "${nodes_array[@]}"; do
        node_ip=$(srun --nodes=1 --ntasks=1 -w "$node" bash -c "ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1")
        if [ "$node" == "$my_node" ]; then
            endpoints="${endpoints}127.0.0.1:${PORT},"
        else
            endpoints="${endpoints}${node_ip}:${PORT},"
        fi
    done

    endpoints=${endpoints%,}
    export VLLM_ENDPOINTS="${endpoints}"
    echo "[Launcher] VLLM_ENDPOINTS=$VLLM_ENDPOINTS"

    srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES bash -c "
        node_ip=\$(ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1)
        export VLLM_HOST_IP=\$node_ip
        export HOST_IP=\$node_ip
        export GLOO_SOCKET_IFNAME=ib0
        export NCCL_SOCKET_IFNAME=ib0
        export VLLM_USE_V1=0
        python -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --tensor-parallel-size $TP --port $PORT --trust-remote-code --gpu-memory-utilization 0.95 --max-model-len 16384 --max-num-seqs 1024 --generation-config vllm --kv-cache-dtype fp8 --disable-log-stats --enable-prefix-caching --enforce-eager $EXTRA_ARGS > \"${SCRATCH_DIR}/logs/vllm_${SLURM_JOB_ID}_$(hostname).log\" 2>&1
    " &
    
    VLLM_PIDS+=($!)

    echo "[Launcher] Waiting for servers to be ready..."
    sleep 20
    python3 -c "
import urllib.request, time
for ep in '$endpoints'.split(','):
    if not ep: continue
    print(f'[Health] Checking {ep}...', flush=True)
    for i in range(720):
        try:
            req = urllib.request.Request(f'http://{ep}/v1/models')
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    print(f'[Health] {ep} is ready!', flush=True)
                    break
        except Exception:
            pass
        time.sleep(10)
"
    echo "[Launcher] Cluster is fully operational."
}

stop_vllm_cluster() {
    echo "[Launcher] Stopping vLLM cluster..."
    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES pkill -f vllm.entrypoints.openai.api_server || true
    echo "[Launcher] Cluster stopped."
    VLLM_PIDS=()
}

run_python_phase() {
    local PHASE=$1
    echo "============================================================"
    echo "[Phase $PHASE] Starting Python pipeline for Phase $PHASE..."
    echo "============================================================"
    export RUN_PHASE=$PHASE
    ulimit -n 65535 || true
    python3 "${PIPELINE_DIR}/08_sft/generate_dialog_sft.py" &
    PYTHON_PID=$!
    wait $PYTHON_PID
    PYTHON_EXIT=$?
    echo "[Phase $PHASE] Python exited with code $PYTHON_EXIT"
    
    if [ $PYTHON_EXIT -ne 0 ]; then
        echo "[ERROR] Python script failed. Stopping pipeline."
        stop_vllm_cluster
        exit $PYTHON_EXIT
    fi
}

function needs_phase() {
    local CKPT_NAME=$1
    if ls "${CHECKPT_DIR}/"*"${CKPT_NAME}.done" 1> /dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

GEN_PORT=$(( 10000 + (SLURM_JOB_ID % 50000) ))
QWEN_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
GEMMA_MODEL="RedHatAI/gemma-4-31B-it-FP8-block"

if needs_phase "p2_tiser_gen"; then
    start_vllm_cluster "$QWEN_MODEL" 4 $GEN_PORT ""
    export GEN_MODEL_ID=$QWEN_MODEL
    run_python_phase 2
    stop_vllm_cluster
fi

if needs_phase "p3_tiser_judge"; then
    start_vllm_cluster "$GEMMA_MODEL" 4 $((GEN_PORT+1)) ""
    export JUDGE_MODEL_ID=$GEMMA_MODEL
    run_python_phase 3
    stop_vllm_cluster
fi

if needs_phase "p4b_coref"; then
    start_vllm_cluster "$QWEN_MODEL" 4 $((GEN_PORT+2)) ""
    export GEN_MODEL_ID=$QWEN_MODEL
    if needs_phase "p3b_tiser_retry"; then
        run_python_phase 3b
    fi
    if needs_phase "p4_casual_gen"; then
        run_python_phase 4
    fi
    run_python_phase 4b
    stop_vllm_cluster
fi

if needs_phase "p5_casual_judge"; then
    start_vllm_cluster "$GEMMA_MODEL" 4 $((GEN_PORT+3)) ""
    export JUDGE_MODEL_ID=$GEMMA_MODEL
    run_python_phase 5
    stop_vllm_cluster
fi

echo "============================================================"
echo "[Phase 6] Interleaving and writing dual output..."
run_python_phase 6

echo "[$(date)] Pipeline Run Complete."
