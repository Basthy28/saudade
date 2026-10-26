#!/bin/bash
# Submit all models for English Closed-Ended evaluation (saudade_chat_en + saudade_chat_cot_en)
cd ${SAUDADE_ROOT:-$HOME/saudade}/eval/benchmark_adapt

MODELS=(
    "amalia-llm/AMALIA-9B-0626-DPO"
    "amalia-llm/AMALIA-9B-0626-SFT"
    "swiss-ai/Apertus-8B-Instruct-2509"
    "allenai/OLMo-2-1124-7B-Instruct"
    "BSC-LT/salamandra-7b-instruct"
    "utter-project/EuroLLM-9B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "PORTULAN/gervasio-8b-portuguese-ptpt-decoder"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-8B"
    "google/gemma-2-9b-it"
    "google/gemma-3-12b-it"
    "mistralai/Ministral-8B-Instruct-2410"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

for m in "${MODELS[@]}"; do
    sbatch run_eval_closed_ended_en.sh "$m"
    echo "Submitted Closed-Ended EN for: $m"
done
