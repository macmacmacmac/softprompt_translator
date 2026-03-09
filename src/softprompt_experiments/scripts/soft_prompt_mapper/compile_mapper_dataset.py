import os
import argparse
import sqlite3
from tqdm import tqdm
import torch
import re


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
    parser.add_argument("--db_path", type=str, default="./datasets/mapper_classification_datasets/classification_5k.sqlite")
    parser.add_argument("--trained_soft_prompts_path", type=str, default="./trained_soft_prompts")
    parser.add_argument("--mapper_dataset_path", type=str, default="./datasets/mapper_training_dataset/compiled_mapper_dataset.pt")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    DB_PATH = args.db_path
    TRAINED_SOFT_PROMPTS_PATH = args.trained_soft_prompts_path
    MAPPER_DATASET_PATH = args.mapper_dataset_path

    # Fetch all hard prompts from SQLite
    print(f"Connecting to database: {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # # Detect and isolate datasets with non-English/weird characters
    # print("Scanning database for non-standard characters...")

    # # Fetch all dataset_ids and sentences from the sentences table
    # cursor.execute("SELECT dataset_id, sentence FROM sentences")
    
    # invalid_datasets = set()
    
    # # Regex for matching any character that is NOT a standard English letter, number, space, or basic punctuation
    # # invalid_char_regex = re.compile(r'[^a-zA-Z0-9\s.,!?\'"()\-:]')
    # # invalid_char_regex = re.compile(r'[^a-zA-Z0-9\s.,!?\'"()\-:;&$/%_+*#@\u2018\u2019\u201C\u201D\u2013\u2014]')

    # # Matches standard Chinese characters
    # invalid_char_regex = re.compile(r'[\u4e00-\u9fff]')
    
    # for dataset_id, sentence in tqdm(cursor.fetchall(), desc="Filtering sentences"):

    #     # If the dataset_id is already not in the set of invalid dataset ids
    #     if dataset_id not in invalid_datasets:

    #         # If the search finds an illegal character, then add the dataset id into the invalid datasets set
    #         if invalid_char_regex.search(sentence):
    #             invalid_datasets.add(dataset_id)
    
    # print(f"Found {len(invalid_datasets)} corrupted datasets containing illegal characters. These will be excluded from the Mapper Dataset.")

    # # Fetch all dataset_ids and hard prompts from the datasets table
    # cursor.execute("SELECT dataset_id, hard_prompt FROM datasets")
    # rows = cursor.fetchall()
    # conn.close()

    # # Create a Dataset ID -> Hard Prompt Dict (Map)
    # # Exclude all which appear in the set of invalid_datasets
    # dataset_map = {}
    # for dataset_id, hard_prompt in rows:
    #     if dataset_id not in invalid_datasets:
    #         dataset_map[dataset_id] = hard_prompt
            
    # print(f"Found {len(dataset_map)} hard prompts ready for compilation.")

    cursor.execute("SELECT dataset_id, hard_prompt FROM datasets")
    rows = cursor.fetchall()
    conn.close()

    # Create a Dataset ID -> Hard Prompt Dict (Map)
    dataset_map = {row[0]: extract_just_keywords_from_hard_prompt(row[1]) for row in rows}
    print(f"Found {len(dataset_map)} hard prompts in the database.")

    # Iterate through the directories and compile the data
    compiled_data = []
    missing_count = 0

    print(f"Scanning soft prompt directories in: {TRAINED_SOFT_PROMPTS_PATH}...")

    # For each dataset_id and its associated hard prompt in the Dataset Map    
    for dataset_id, hard_prompt in tqdm(dataset_map.items(), desc="Compiling Data"):

        # Construct a path to fetch to trained the prompts for the dataset_id 
        soft_prompt_path = os.path.join(TRAINED_SOFT_PROMPTS_PATH, f"dataset_{dataset_id}", "softprompt.pt")
        
        # Skip if the soft prompt doesn't exist for the current dataset id
        if not os.path.exists(soft_prompt_path):
            missing_count += 1
            tqdm.write(f"Warning: Missing soft prompts for dataset: {dataset_id}")
            continue
            
        # Load the saved state dict
        # weights_only=True is a PyTorch security best practice for loading tensors
        state_dict = torch.load(soft_prompt_path, map_location="cpu", weights_only=True)
        
        # Extract the prompt embeddings. 
        # The SoftPrompt class saves it as shape (1, num_tokens, embed_dim).
        soft_prompt_tensor = state_dict['prompt_embeddings'].squeeze(0)         # (num_tokens, embed_dim)
        
        # Accumulate Dataset ID, Soft Prompt Tensor, and the Hard Prompt Tensor into the list of compiled data
        compiled_data.append({
            "dataset_id": dataset_id,
            "soft_prompt": soft_prompt_tensor,
            "hard_prompt": hard_prompt
        })

    # Save the Compiled Data List to a Torch File
    print(f"\nCompilation Complete!")
    print(f"Successfully paired: {len(compiled_data)} datasets.")

    # Log how many datasets were skipped
    if missing_count > 0:
        print(f"Skipped {missing_count} datasets (softprompt.pt not found).")


    os.makedirs(os.path.dirname(MAPPER_DATASET_PATH), exist_ok=True)
    torch.save(compiled_data, MAPPER_DATASET_PATH)
    print(f"Saved unified PyTorch dataset to: {MAPPER_DATASET_PATH}")



