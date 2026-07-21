#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64g
#SBATCH -J "Mapper70B"
#SBATCH -p long
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:2,gpumem:150G
#SBATCH -C A100
#SBATCH -o mapper_70b.out
#SBATCH -e mapper_70b.out

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
# Run the 70B Mapper Training
# -----------------------------
python -u -m run_experiment \
  --scripts soft_prompt_mapper.supernat_instruct_DoD.train_mapper \
  --model_name meta-llama/Llama-3.1-70B-Instruct \
  --batch_size 2
