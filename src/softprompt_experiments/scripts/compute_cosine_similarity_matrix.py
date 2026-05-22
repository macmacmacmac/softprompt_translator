import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm
import argparse
import os

# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--proportion_to_use", type=float, default=1.0)
    parser.add_argument("--mapper_dataset_path", type=str, default="supnat_eng_fil_orig")
    # parser.add_argument("--verbalization_json_path", type=str, default="./SupNatInstruct_verbalizations_original_instructions.json")
    parser.add_argument("--embed_model_name", type=str, default="all-MiniLM-L6-v2")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MAPPER_DATASET_PATH = os.path.join('./datasets/mapper_training_dataset',args.mapper_dataset_path)
    PROPORTION_FOLDER = f"{int(100*args.proportion_to_use)}_percent"
    LORA_DIR = os.path.join(MAPPER_DATASET_PATH, "mapper_lora_weights", PROPORTION_FOLDER)
    VERBALIZATION_JSON_PATH = os.path.join(LORA_DIR, "verbalizations.json")
    EMBED_MODEL_NAME = args.embed_model_name

    print(f"Loading data from {VERBALIZATION_JSON_PATH}...")
    with open(VERBALIZATION_JSON_PATH, "r") as f:
        data = json.load(f)

    # Extract texts
    # p_i: the generated verbalizations
    # g_i: the ground-truth hard prompts
    tasks = []
    predicted_hard_prompts = []
    original_hard_prompts = []
    rougels = []

    for item in data:
        # Some verbalizations might be empty or None, ensure they are strings
        predicted = str(item.get("verbalization", "")).strip()
        original = str(item.get("hard_prompt", "")).strip()
        verbalization_rouge_l = float(item.get("verbalization_rouge_l"))
        rougels.append(verbalization_rouge_l)
        # Only evaluate items where we have both
        if predicted and original:
            tasks.append(item.get("task_name", "unknown"))
            predicted_hard_prompts.append(predicted)
            original_hard_prompts.append(original)

    N = len(predicted_hard_prompts)
    if N == 0:
        print("No valid pairs found to evaluate.")
        return
    print(f"Extracted {N} valid (verbalization, hard_prompt) pairs.")

    # Compute Embeddings
    print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    print("Computing embeddings for verbalizations...")
    # Shape: (N, embed_dim)
    verbalization_embeddings = model.encode(predicted_hard_prompts, convert_to_tensor=True, show_progress_bar=True)

    print("Computing embeddings for ground-truth hard prompts...")
    # Shape: (N, embed_dim)
    original_embeddings = model.encode(original_hard_prompts, convert_to_tensor=True, show_progress_bar=True)

    # Calculate Cosine Similarity Matrix
    # We want a matrix where entry (i, j) is the cosine similarity between p_j and g_i
    # PyTorch's cos_sim computes pairwise similarities
    print("Calculating cosine similarity matrix...")
    
    # cos_mat[i][j] = cos(g_i, p_j)
    cos_mat = cos_sim(original_embeddings, verbalization_embeddings).cpu().numpy()

    # Compute the Percentile Rank metric
    # For each test example i=1...N: fraction |j: cos(p_i, g_i) >= cos(p_j, g_i)| / N
    print("Computing metric...")
    fractions = []
    
    for i in range(N):
        # Target similarity: cos(p_i, g_i) -> entry (i,i) on the diagonal
        target_sim = cos_mat[i, i]
        
        # All similarities against g_i: cos(p_j, g_i) for all j -> row i
        row_sims = cos_mat[i, :]
        
        # Count how many j predictions have a similarity <= our target similarity
        # (This matches: cos(p_i, g_i) >= cos(p_j, g_i))
        count_less_or_equal = np.sum(target_sim >= row_sims)
        
        # Fraction out of N
        fraction = count_less_or_equal / N
        fractions.append(fraction)

    # Final Average
    mean_percentile_rank = np.mean(fractions)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Examples Evaluated : {N}")
    print(f"Mean Percentile Rank     : {mean_percentile_rank:.4f} ({mean_percentile_rank*100:.2f}%)")
    print("="*50)
    
    # Print out the best and worst performing tasks for debugging
    target_sims = np.diag(cos_mat) # cos(p_i, g_i)
    
    best_idx = np.argmax(target_sims)
    worst_idx = np.argmin(target_sims)
    
    print("\n[Insights]")
    print(f"Best match task: {tasks[best_idx]} (cos_sim={target_sims[best_idx]:.4f})")
    print(f"Worst match task: {tasks[worst_idx]} (cos_sim={target_sims[worst_idx]:.4f})")