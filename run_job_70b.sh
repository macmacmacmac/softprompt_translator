#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16g
#SBATCH -J "Mapper70B"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:A100:2
#SBATCH -o mapper_70b.out
#SBATCH -e mapper_70b.out

nvidia-smi

# -----------------------------
# Run the 70B Mapper Training
# -----------------------------
python -u -m run_experiment \
  --scripts soft_prompt_mapper.supernat_instruct_DoD.train_mapper \
  --model_name meta-llama/Llama-3.1-70B-Instruct \
  --batch_size 8 \
  --mapper_dataset_path ./shared/datasets/mapper_training_dataset/General-DoD-10x


# -----------------------------
# Run the 70B Mapper Testing
# -----------------------------
# python -u -m run_experiment \
#   --scripts soft_prompt_mapper.supernat_instruct_DoD.test_mapper \
#   --model_name meta-llama/Llama-3.1-70B-Instruct \
#   --batch_size 16