#!/bin/bash
cd ${SAUDADE_ROOT:-$HOME/saudade}/pipeline/judge-pipeline-temporal
for model in amalia_sft amalia_dpo qwen25_7b qwen3_8b llama31_8b eurollm_9b salamandra_7b gervasio_8b olmo2_7b gemma2_9b gemma3_12b ministral_8b mistral_7b apertus_8b amalia_lora_r16 amalia_lora_r32; do
    sbatch run_eval_mn5.sh "$model"
    echo "Submitted: $model"
done
