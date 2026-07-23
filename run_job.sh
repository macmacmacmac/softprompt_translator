#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=180g
#SBATCH -J "70B SP"
#SBATCH -p long
#SBATCH -t 5-00:00:00
#SBATCH --gres=gpu:A100:2
#SBATCH -o logs.out
#SBATCH -e logs.out

# Avoid GPU memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

# nvidia-smi

# -----------------------------
# Run the Job (Example: Python Script / Module)
# -----------------------------
# python -u -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.train_softprompts
accelerate launch --config_file fsdp_config.yaml -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.accelerate_train_softprompts --batch_size 16
# python -u -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.apply_InSPEcT_on_DoD --peft
# python -u -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.generate_paraphrasals
# python -u -m run_experiment --scripts soft_prompt_mapper.classification_DoD.inference_mapper_dataset
