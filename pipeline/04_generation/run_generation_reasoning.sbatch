#!/bin/bash
#SBATCH --job-name=saudade_generation
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=40
#SBATCH --account=epor32
#SBATCH --qos=acc_debug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /opt/Anaconda-2021.05/etc/profile.d/conda.sh || source /apps/ACC/ANACONDA/2023.07/etc/profile.d/conda.sh
conda activate vllm_new
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
OUT_DIR="${SAUDADE_DATA:-${SAUDADE_ROOT:-$HOME/saudade}/data}/04_generated"
mkdir -p "${OUT_DIR}"
mkdir -p logs

cd ${SAUDADE_ROOT:-$HOME/saudade}

export HF_HOME="${SAUDADE_SCRATCH:-$HOME/scratch}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_DIR="${SAUDADE_SCRATCH:-$HOME/scratch}/.cache/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCH_DISTRIBUTED_TIMEOUT=1800

# ── Cluster Management Functions ─────────────────────────────────────────────

PORT=$(( 10000 + (SLURM_JOB_ID % 50000) ))

echo "[Launcher] Starting $MODEL across $SLURM_NNODES nodes (TP=4)..."

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
endpoints=""
declare -a VLLM_PIDS=()

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
    unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
    node_ip=\$(ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1)
    export VLLM_HOST_IP=\\\$node_ip
    export HOST_IP=\\\$node_ip
    export GLOO_SOCKET_IFNAME=ib0
    export NCCL_SOCKET_IFNAME=ib0
    export VLLM_USE_V1=1
    python3 -m vllm.entrypoints.openai.api_server --model \"${MODEL}\" --tensor-parallel-size 4 --port ${PORT} --trust-remote-code --gpu-memory-utilization 0.95 --max-model-len 32768 --kv-cache-dtype fp8 --enable-prefix-caching --enforce-eager > \"${OUT_DIR}/vllm_node_\${SLURM_NODEID}.log\" 2>&1
" &

VLLM_PIDS+=($!)

echo "[Launcher] Waiting for servers to be ready..."
sleep 20
python3 -c "
import urllib.request, time
for ep in '$endpoints'.split(','):
    if not ep: continue
    print(f'[Health] Checking {ep}...', flush=True)
    while True:
        try:
            req = urllib.request.Request(f'http://{ep}/health')
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    print(f'[Health] {ep} is ready!', flush=True)
                    break
        except Exception as e:
            print(f'[Health] {ep} not ready yet: {e}', flush=True)
        time.sleep(10)
"
echo "[Launcher] Cluster is fully operational."

export LOCAL_LLM_URL="http://127.0.0.1:${PORT}/v1"
export LOCAL_MODEL_NAME="${MODEL}"

CAT_ARGS=""
if [ ! -z "$GEN_CATS" ]; then
    CAT_ARGS="--cats $GEN_CATS"
fi

# Run the python generation scripts in parallel on each node
srun --overlap --ntasks=$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=1 --exact bash -c "
    unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
    echo '=== 3. REASONING ==='
    python3 "${SAUDADE_ROOT:-$HOME/saudade}/pipeline/04_generation/generate_reasoning.py" $CAT_ARGS
"

echo "[Launcher] Stopping vLLM cluster..."
for pid in "${VLLM_PIDS[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
done
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES pkill -f vllm.entrypoints.openai.api_server || true
echo "[Launcher] Cluster stopped."
