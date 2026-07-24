#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16g
#SBATCH -J "PrefDS"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH -o pref2.out
#SBATCH -e pref2.out

# Avoid GPU memory fragmentation
# export PYTORCH_ALLOC_CONF=expandable_segments:True

# nvidia-smi

# -----------------------------
# Run the Job (Example: Python Script / Module)
# -----------------------------
python -u -m run_experiment --scripts dpo.generate_preference_dataset --score-fn ROUGE-L --use-vllm
# accelerate launch --config_file fsdp_config.yaml -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.accelerate_train_softprompts --batch_size 16
