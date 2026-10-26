#!/bin/bash
#SBATCH --job-name=saudade_distill
#SBATCH --account=epor32
#SBATCH --partition=ehpc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "--------------------------------------------------------"
echo "Job V5 Distillation Start: $(date)"
echo "Alocou as seguintes GPUs: $CUDA_VISIBLE_DEVICES"
echo "Modelo: Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
echo "--------------------------------------------------------"

# Ativa o ambiente conda com vLLM/Transformers atualizado para Qwen3
source /opt/Anaconda-2021.05/etc/profile.d/conda.sh
conda activate vllm_new

# Configurações Cruciais para não Crashar a Máquina (Apontar para o disco Data)
export HF_HOME=${SAUDADE_SCRATCH:-$HOME/scratch}/hf_cache
export TMPDIR=${SAUDADE_SCRATCH:-$HOME/scratch}/tmp
export TEMP=${SAUDADE_SCRATCH:-$HOME/scratch}/tmp
export NUMBA_CACHE_DIR=${SAUDADE_SCRATCH:-$HOME/scratch}/numba_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Evitar que NCCL comunique por caminhos não ideais
export NCCL_P2P_DISABLE=0
# Retirei o IB_DISABLE porque as H200 no mesmo node usam o IB/NVLink de forma eficiente para MoEs gigantes
export NCCL_IB_DISABLE=0

echo "[vLLM] A iniciar Servidor LLM Local (Qwen 235B MoE) em 4x GPUs (Tensor Parallelism = 4)..."

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 4 \
  --port 8088 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --swap-space 0 \
  --kv-cache-dtype fp8 > vllm_server.log 2>&1 &

VLLM_PID=$!

echo "[vLLM] A aguardar carregamento do Modelo MoE 235B (pode demorar alguns minutos)..."
while ! curl -s http://localhost:8088/v1/models > /dev/null; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[ERRO] Processo vLLM terminou inesperadamente. Ultimos logs:"
    tail -n 30 vllm_server.log
    exit 1
  fi
  sleep 10
done
echo "[vLLM] 🟢 Servidor Qwen3-235B Online e Pronto na porta 8088!"
echo "--------------------------------------------------------"

echo "[Phase 2 V5] A iniciar Script de Distilação na partição ehpc..."
python "${SAUDADE_ROOT:-$HOME/saudade}/pipeline/02_distillation/distill_entities.py"

echo "--------------------------------------------------------"
echo "[Concluído] Script de Python finalizado com sucesso."
echo "[Cleanup] A matar o servidor vLLM (PID: $VLLM_PID)..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

echo "Job Terminado: $(date)"
