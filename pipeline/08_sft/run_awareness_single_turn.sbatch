#!/bin/bash
#SBATCH --job-name=saudade_awareness
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --time=06:00:00
#SBATCH --account=epor32
#SBATCH --partition=ehpc
#SBATCH --qos=acc_ehpc
#SBATCH --signal=B:USR1@900
#SBATCH -o ${SAUDADE_SCRATCH:-$HOME/scratch}/saudade_awareness/logs/%x-%j.out
#SBATCH -e ${SAUDADE_SCRATCH:-$HOME/scratch}/saudade_awareness/logs/%x-%j.err

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
SCRATCH_DIR="${SAUDADE_SCRATCH:-$HOME/scratch}/saudade_awareness"
export SCRATCH_DIR
mkdir -p logs

export MODEL_PATH="${SAUDADE_SCRATCH:-$HOME/scratch}/models/Qwen3-235B-A22B-Instruct-2507-FP8"
if [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
fi
export GEN_MODEL_ID="$MODEL_PATH"
export PORT=8088
export MAX_CONCURRENT=4000

echo "============================================================"
echo " Awareness single-turn generation (MN5)"
echo " Nodes:    ${SLURM_NNODES}"
echo " Date:     $(date)"
echo " Job ID:   $SLURM_JOB_ID"
echo " Model:    $MODEL_PATH"
echo "============================================================"

# Launch vLLM on all allocated nodes
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
endpoints=""

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
    export VLLM_USE_V1=1
    python -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --tensor-parallel-size 4 --port $PORT --trust-remote-code --gpu-memory-utilization 0.95 --max-model-len 16384 --generation-config vllm --kv-cache-dtype fp8 --disable-log-stats --enable-prefix-caching --enforce-eager > \"${SCRATCH_DIR}/logs/vllm_\${SLURM_JOB_ID}_\$(hostname).log\" 2>&1
" &

VLLM_PID=$!

echo "[Launcher] Waiting for vLLM servers..."
sleep 25
python3 -c "
import urllib.request, time
for ep in '$endpoints'.split(','):
    if not ep: continue
    print(f'[Health] Checking {ep}...', flush=True)
    while True:
        try:
            r = urllib.request.urlopen(f'http://{ep}/v1/models', timeout=5)
            if r.status == 200:
                print(f'[Health] {ep} READY!', flush=True)
                break
        except Exception:
            time.sleep(5)
"

echo "[Launcher] Running single-turn generation..."
python3 ${PIPELINE_DIR}/08_sft/generate_awareness_single_turn.py

echo "[Launcher] Completed successfully! Cleaning up..."
kill $VLLM_PID 2>/dev/null || true
echo "[$(date)] DONE!"
