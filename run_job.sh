#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16g
#SBATCH -J "PrefDS"
#SBATCH -p short
#SBATCH -t 1-00:00:00
###SBATCH --gres=gpu:A100:1,gpumem:80G
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH -o pref3.out
#SBATCH -e pref3.out

# Avoid GPU memory fragmentation
# export PYTORCH_ALLOC_CONF=expandable_segments:True

nvidia-smi

# -----------------------------
# Run the Job (Example: Python Script / Module)
# -----------------------------
python -u -m run_experiment --scripts dpo.generate_preference_dataset \
    -n 10 -t 0.5 --max-retry-temperature 1.5 \
    --lora-weights-path ./shared/mapper_lora_weights/DPO_SFT2_rougeL_0.1/meta-llama/Llama-3.1-8B-Instruct \
    --score-fn ROUGE-L --use-vllm


# accelerate launch --config_file fsdp_config.yaml -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.accelerate_train_softprompts --batch_size 16
