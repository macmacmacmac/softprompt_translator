#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16g
#SBATCH -J "InSPEcT"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:L40S:1
#SBATCH --gres=gpu:rtx_pro_6000_b:1
##SBATCH -C A100-80G
#SBATCH -o inspect.out
#SBATCH -e inspect.out

# Avoid GPU memory fragmentation
# export PYTORCH_ALLOC_CONF=expandable_segments:True

nvidia-smi

# -----------------------------
# Run the Job (Example: Python Script / Module)
# -----------------------------
# python -u -m run_experiment --scripts dpo.generate_preference_dataset \
#     -n 10 -t 0.5 --max-retry-temperature 1.5 \
#     --lora-weights-path ./shared/mapper_lora_weights_overfit/General-DoD-10x/meta-llama/Llama-3.1-8B-Instruct \
#     --split val --score-fn ROUGE-L --use-vllm


# python -u -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.train_softprompts --train_only_test

python -u -m run_experiment --scripts soft_prompt_mapper.classification_DoD.train_InSPEcT_softprompts_peft


# python -u -m run_experiment --scripts comparisons.vllm_pipeline --methods mapper10x mapper10x_z_W \
#     --input ./shared/verbalizations/master_verbalizations_8b.json \
#     --output ./shared/verbalizations/master_8b.json



# python run_experiment.py --scripts soft_prompt_mapper.supernat_instruct_DoD.generate_paraphrasals


# accelerate launch --config_file fsdp_config.yaml -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.accelerate_train_softprompts --batch_size 16
