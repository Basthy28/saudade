#!/bin/bash
#SBATCH --job-name=saudade_curation
#SBATCH --partition=acc

#SBATCH --nodes=25
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --account=epor32
#SBATCH --qos=acc_ehpc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source /opt/Anaconda-2021.05/etc/profile.d/conda.sh || source /apps/ACC/ANACONDA/2023.07/etc/profile.d/conda.sh
conda activate vllm_new

export HF_HOME=${SAUDADE_SCRATCH:-$HOME/scratch}/.cache/huggingface
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCH_DISTRIBUTED_TIMEOUT=1800
export TEMP=${SAUDADE_SCRATCH:-$HOME/scratch}/tmp
export NUMBA_CACHE_DIR=${SAUDADE_SCRATCH:-$HOME/scratch}/numba_cache

PIPELINE_DIR="${SAUDADE_ROOT:-$HOME/saudade}/pipeline"
CURATION_DIR="${PIPELINE_DIR}/05_curation"
DATA_DIR="${SAUDADE_DATA:-${SAUDADE_ROOT:-$HOME/saudade}/data}/04_generated"
OUTPUT_DIR="${CURATION_DIR}/output"
VLLM_PORT=8091

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "=========================================="
echo " SAUDADE curation (25 nodes)"
echo " Date: $(date)"
echo "=========================================="

echo "[$(date)] Starting vLLM Llama 3.3 70B (TP=2) across $SLURM_NNODES nodes..."
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES bash -c "
    unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
    python3 -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --tensor-parallel-size 2 \
        --port ${VLLM_PORT} \
        --gpu-memory-utilization 0.92 \
        --max-model-len 4096 \
        --dtype bfloat16 \
        > \"${CURATION_DIR}/logs/vllm_curation_node_\${SLURM_NODEID}.log\" 2>&1
" &
VLLM_PID=$!

echo "[$(date)] Waiting 90s for all vLLM instances to boot..."
sleep 90

echo "[$(date)] Running curation on awareness and cognition across $SLURM_NNODES nodes..."
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES bash -c "
    unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
    python3 \"${CURATION_DIR}/curate_benchmark.py\" \
        --base_url \"http://localhost:${VLLM_PORT}/v1\" \
        --data_dir \"${DATA_DIR}\" \
        --output_dir \"${OUTPUT_DIR}\" \
        --max_concurrent 256 \
        --datasets awareness cognition
"

echo "[$(date)] Cleanup vLLM..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null || true

echo "[$(date)] Curation EHPC complete."
