#!/bin/bash
#SBATCH --job-name=saudade_purification
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

MODEL="RedHatAI/gemma-4-31B-it-FP8-block"
OUT_DIR="${SAUDADE_DATA:-${SAUDADE_ROOT:-$HOME/saudade}/data}/03_purification/output"
mkdir -p "${OUT_DIR}"
mkdir -p logs

cd ${SAUDADE_ROOT:-$HOME/saudade}

export HF_HOME="${SAUDADE_SCRATCH:-$HOME/scratch}/.cache/huggingface"
export VLLM_CACHE_DIR="${SAUDADE_SCRATCH:-$HOME/scratch}/.cache/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCH_DISTRIBUTED_TIMEOUT=1800

# We use srun to launch an instance on each of the 8 nodes simultaneously
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
    BASE_PORT=\$(( 10000 + (SLURM_JOB_ID % 40000) ))
    PORT=\$((BASE_PORT + SLURM_NODEID))

    echo '=========================================='
    echo 'Fact purification (Gemma 31B, checkpointed) '
    echo 'Node: \$(hostname) (ID: \${SLURM_NODEID}) | Port: \${PORT}'
    echo '=========================================='

    python3 -m vllm.entrypoints.openai.api_server \
      --model \"${MODEL}\" \
      --port \${PORT} \
      --tensor-parallel-size 2 \
      --max-model-len 16384 \
      --max-num-batched-tokens 65536 \
      --dtype bfloat16 \
      > \"${OUT_DIR}/vllm_node_\${SLURM_NODEID}.log\" 2>&1 &
    VLLM_PID=\$!

    echo \"Waiting for vLLM to start on port \${PORT}...\"
    for i in \$(seq 1 120); do
      if curl -s \"http://localhost:\${PORT}/health\" > /dev/null 2>&1; then
        echo \"[\$(date)] vLLM ready on Node \$(hostname)!\"
        break
      fi
      sleep 5
    done

    CAT_ARGS=\"\"
    if [ ! -z \"\$PURIFY_CATS\" ]; then
        CAT_ARGS=\"--cats \$PURIFY_CATS\"
    fi

    python3 pipeline/03_purification/check_distilled_hallucinations.py \
      --sample_size 0 \
      --api_url \"http://localhost:\${PORT}/v1\" \
      --model \"${MODEL}\" \
      --concurrent 256 \
      --out \"${OUT_DIR}/distilled_hallucination_report_ALL.jsonl\" \
      \$CAT_ARGS

    kill -9 \${VLLM_PID}
    echo \"[\$(date)] Node \$(hostname) done\"
"

