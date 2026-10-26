#!/bin/bash
#SBATCH --job-name=saudade_openended
#SBATCH --nodes=25
#SBATCH --ntasks=25
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --account=epor32
#SBATCH --partition=ehpc
#SBATCH --qos=acc_ehpc
#SBATCH --signal=B:USR1@120
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
SCRATCH_DIR="${SAUDADE_SCRATCH:-$HOME/scratch}/saudade_openended"
mkdir -p logs

export N_TEST=${N_TEST:-0}
export MAX_CONCURRENT=${MAX_CONCURRENT:-2000}
export SHARD_SIZE=${SHARD_SIZE:-500}

# Modelos
QWEN_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
GEMMA_MODEL="RedHatAI/gemma-4-31B-it-FP8-block"

# Portas baseadas no job ID para evitar conflitos
GEN_PORT=$(( 10000 + (SLURM_JOB_ID % 50000) ))
JUDGE_PORT=$(( GEN_PORT + 1 ))

declare -a VLLM_PIDS=()
PYTHON_PID=""

echo "============================================================"
echo " Open-ended generation — temporalidade explícita nas perguntas"
echo " Fixes: anchor temporal necessário (cognition + awareness)"
echo " Nodes:   ${SLURM_NNODES}"
echo " Date:    $(date)"
echo " Job:     $SLURM_JOB_ID"
echo "============================================================"

# ── Signal handler ────────────────────────────────────────────────────────────
timeout_handler() {
    echo "[$(date)] [SIGNAL] Timeout — a sinalizar Python para checkpoint..."
    [ -n "$PYTHON_PID" ] && kill -USR1 "$PYTHON_PID" 2>/dev/null || true
    sleep 840
    echo "[$(date)] [SIGNAL] Saindo."
}
trap 'timeout_handler' USR1

# ── Funções de cluster ────────────────────────────────────────────────────────
start_vllm_cluster() {
    local MODEL_PATH=$1
    local TP=$2
    local PORT=$3

    echo "[Launcher] A iniciar $MODEL_PATH (TP=$TP, PORT=$PORT)..."

    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)
    endpoints=""
    VLLM_PIDS=()
    my_node=$(hostname)

    for node in "${nodes_array[@]}"; do
        node_ip=$(srun --nodes=1 --ntasks=1 -w "$node" bash -c \
            "ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1")
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
        export VLLM_USE_V1=1
        python -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --tensor-parallel-size $TP \
            --port $PORT \
            --trust-remote-code \
            --gpu-memory-utilization 0.95 \
            --max-model-len 32768 \
            --generation-config vllm \
            --kv-cache-dtype fp8 \
            --disable-log-stats \
            --enable-prefix-caching \
            --enforce-eager \
            > \"${SCRATCH_DIR}/logs/vllm_${SLURM_JOB_ID}_\$(hostname).log\" 2>&1
    " &
    VLLM_PIDS+=($!)

    echo "[Launcher] A aguardar servidores..."
    sleep 20
    python3 -c "
import urllib.request, time
healthy = []
for ep in '$endpoints'.split(','):
    if not ep: continue
    print(f'[Health] A verificar {ep}...', flush=True)
    is_ok = False
    for attempt in range(60):
        try:
            with urllib.request.urlopen(f'http://{ep}/v1/models', timeout=5) as r:
                if r.status == 200:
                    print(f'[Health] {ep} pronto!', flush=True)
                    healthy.append(ep)
                    is_ok = True
                    break
        except Exception:
            pass
        time.sleep(10)
    if not is_ok:
        print(f'[Health] [WARNING] {ep} falhou e sera ignorado.', flush=True)

with open('healthy_endpoints.txt', 'w') as f:
    f.write(','.join(healthy))
"
    export VLLM_ENDPOINTS=$(cat healthy_endpoints.txt)
    rm -f healthy_endpoints.txt
    echo "[Launcher] Cluster operacional. Endpoints finais: $VLLM_ENDPOINTS"
}

stop_vllm_cluster() {
    echo "[Launcher] A parar cluster vLLM..."
    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES \
        pkill -f vllm.entrypoints.openai.api_server || true
    echo "[Launcher] Cluster parado."
    VLLM_PIDS=()
}

# ── FASE 1: Geração (Qwen 235B, TP=4) ────────────────────────────────────────
echo ""
echo "============================================================"
echo "[FASE 1] Geracao open-ended com Qwen 235B"
echo "============================================================"

start_vllm_cluster "$QWEN_MODEL" 4 $GEN_PORT

export GEN_MODEL_ID=$QWEN_MODEL
export PORT=$GEN_PORT
ulimit -n 65535 || true

python3 "${PIPELINE_DIR}/07_openended/generate_openended.py" &
PYTHON_PID=$!
wait $PYTHON_PID
GEN_EXIT=$?

stop_vllm_cluster

if [ $GEN_EXIT -ne 0 ]; then
    echo "[ERROR] Geracao falhou com codigo $GEN_EXIT"
    exit $GEN_EXIT
fi

echo "[FASE 1] Geracao concluida."

# ── FASE 2: Judge (Gemma-4 31B, TP=2) ────────────────────────────────────────
echo ""
echo "============================================================"
echo "[FASE 2] Judge de qualidade com Gemma-4 31B"
echo "============================================================"

start_vllm_cluster "$GEMMA_MODEL" 2 $JUDGE_PORT

export JUDGE_MODEL_ID=$GEMMA_MODEL
export PORT=$JUDGE_PORT
ulimit -n 65535 || true

python3 "${PIPELINE_DIR}/07_openended/judge_openended.py" &
PYTHON_PID=$!
wait $PYTHON_PID
JUDGE_EXIT=$?

stop_vllm_cluster

if [ $JUDGE_EXIT -ne 0 ]; then
    echo "[ERROR] Judge falhou com codigo $JUDGE_EXIT"
    exit $JUDGE_EXIT
fi

echo "[FASE 2] Judge concluido."
echo ""
echo "[$(date)] Pipeline open-ended completa."
echo "Ficheiros em: ${SAUDADE_DATA:-${SAUDADE_ROOT:-$HOME/saudade}/data}/07_openended/"
echo "Ficheiros clean (pass=True): *_clean.jsonl"
