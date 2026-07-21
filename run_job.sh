#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=16g
#SBATCH -J "PrefData"
#SBATCH -p long
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -o vllm.out
#SBATCH -e vllm.out

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
# python -u -m run_experiment --scripts dpo.generate_preference_dataset -n 10
python -u -m run_experiment --scripts dpo.generate_preference_dataset --score-fn ROUGE-L --use-vllm

