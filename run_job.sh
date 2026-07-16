#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16g
#SBATCH -J "Retrain"
#SBATCH -p long
#SBATCH -t 5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C L40S
#SBATCH -o retrain.out
#SBATCH -e retrain.out

# -----------------------------
# Load Required Modules
# -----------------------------
module load python/3.12.3
module load cuda/12.9.0

# -----------------------------
# Create / Activate venv
# -----------------------------
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Activating existing virtual environment..."
    source "$VENV_DIR/bin/activate"
fi

# Avoid GPU memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

# -----------------------------
# Run the Job (Example: Python Script / Module)
# -----------------------------
python -u -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.train_softprompts --save_dir ./trained_soft_prompts/General-DoD-wt-decay
# python -u -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.apply_InSPEcT_on_DoD --peft
# python -u -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.generate_paraphrasals
# python -u -m run_experiment --scripts soft_prompt_mapper.classification_DoD.inference_mapper_dataset
