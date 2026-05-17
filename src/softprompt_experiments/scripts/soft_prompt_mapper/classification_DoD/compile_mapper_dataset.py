import os
import argparse
import sqlite3
from tqdm import tqdm
import torch
import random


def extract_just_keywords_from_hard_prompt(hard_prompt: str) -> str:
    return hard_prompt.split("Classify the following sentence as:")[-1].strip()


# Driver Code
def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="./datasets/mapper_classification_datasets/DoD_3_5k.sqlite")
    parser.add_argument("--trained_soft_prompts_dir", type=str, default="./trained_soft_prompts/DoD_3_5k_peft_sample_vocab")
    parser.add_argument("--compiled_dataset_dir", type=str, default="./datasets/mapper_training_dataset")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of mini-datasets to process (e.g., 2000)")
    parser.add_argument("--peft", action="store_true", help="Use PEFT style way of loading soft prompts")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    DB_PATH = args.db_path
    TRAINED_SOFT_PROMPTS_DIR = args.trained_soft_prompts_dir
    COMPILED_DATASET_DIR = args.compiled_dataset_dir
    SEED = args.seed
    LIMIT = args.limit
    LOAD_LIKE_PEFT = args.peft

    # Determine Dataset Name
    DATASET_NAME = TRAINED_SOFT_PROMPTS_DIR.split('/')[-1]

    # Fetch all hard prompts from SQLite
    print(f"Connecting to database: {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT dataset_id, hard_prompt FROM datasets")
    rows = cursor.fetchall()
    conn.close()

    # Optional: Limit the number of datasets if an argument is passed
    if LIMIT is not None and LIMIT > 0:
        print(f"Limiting processing to a random subset of {LIMIT} datasets...")
        random.seed(SEED)  # To ensure reproducibility 
        random.shuffle(rows)
        rows = rows[:LIMIT]

    # Create a Dataset ID -> Hard Prompt Dict (Map)
    dataset_map = {row[0]: extract_just_keywords_from_hard_prompt(row[1]) for row in rows}
    print(f"Found {len(dataset_map)} hard prompts in the database.")

    # Iterate through the directories and compile the data
    compiled_data = []
    missing_count = 0

    print(f"Scanning soft prompt directories in: {TRAINED_SOFT_PROMPTS_DIR} ...")

    # For each dataset_id and its associated hard prompt in the Dataset Map    
    for dataset_id, hard_prompt in tqdm(dataset_map.items(), desc="Compiling Data"):

        # Construct a path to fetch to trained the prompts for the dataset_id 
        soft_prompt_path = os.path.join(TRAINED_SOFT_PROMPTS_DIR, f"dataset_{dataset_id}", "softprompt.pt")
        
        # Skip if the soft prompt doesn't exist for the current dataset id
        if not os.path.exists(soft_prompt_path):
            missing_count += 1
            tqdm.write(f"Warning: Missing soft prompts for dataset: {dataset_id}")
            continue

        # Load Like how Peft Stores soft prompts
        if LOAD_LIKE_PEFT:
            soft_prompt_tensor = torch.load(soft_prompt_path, map_location="cpu", weights_only=True)

        # Load like how Custom Implementation of SoftPrompt class stores soft prompts
        else:
            # Load the saved state dict
            # weights_only=True is a PyTorch security best practice for loading tensors
            state_dict = torch.load(soft_prompt_path, map_location="cpu", weights_only=True)
            
            # Extract the prompt embeddings
            # The SoftPrompt class saves it as shape (1, num_tokens, embed_dim).
            soft_prompt_tensor = state_dict['prompt_embeddings'].squeeze(0)         # (num_tokens, embed_dim)
        
        # Accumulate Dataset ID, Soft Prompt Tensor, and the Hard Prompt Tensor into the list of compiled data
        compiled_data.append({
            "dataset_id": dataset_id,
            "soft_prompt": soft_prompt_tensor,
            "hard_prompt": hard_prompt
        })

    # Save the Compiled Data List to a Torch File
    print(f"\nCompilation Complete! Successfully paired: {len(compiled_data)} datasets.")

    # Log how many datasets were skipped
    if missing_count > 0:
        print(f"Skipped {missing_count} datasets (softprompt.pt not found).")


    # Perform Train / Validation Split (90 / 10)
    print("\nShuffling and splitting datasets...")
    random.seed(SEED)
    random.shuffle(compiled_data)

    split_idx = int(len(compiled_data) * 0.9)
    train_data = compiled_data[:split_idx]
    val_data = compiled_data[split_idx:]

    # Determine the directory name based on whether a LIMIT was provided
    if LIMIT is not None and LIMIT > 0:
        save_dir_name = f"{DATASET_NAME}_{LIMIT}"
    else:
        save_dir_name = DATASET_NAME

    # Create the Directory for saving the datasets
    save_dir = os.path.join(COMPILED_DATASET_DIR, save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the Training and Validation Datasets
    train_dataset_path = os.path.join(save_dir, 'train_mapper_dataset.pt')
    val_dataset_path = os.path.join(save_dir, 'val_mapper_dataset.pt')
    
    torch.save(train_data, train_dataset_path)
    torch.save(val_data, val_dataset_path)
    
    print(f"Saved Train Split ({len(train_data)} samples) to: {train_dataset_path}")
    print(f"Saved Val Split ({len(val_data)} samples) to: {val_dataset_path}")



