import os
import argparse
from tqdm import tqdm
import torch
from datasets import load_dataset
from typing import List, Dict, Any
import random


def compile_data_list(dataset_records: List[Dict[str, Any]], 
                      trained_soft_prompts_dir: str, 
                      dataset_name: str) -> List[Dict]:

    compiled_data = []
    missing_count = 0
    last_task_name = ""

    print(f"Scanning soft prompt directories in: {trained_soft_prompts_dir} ...")

    # For each Task Name and its associated hard prompt in the Train Dataset Map    
    for task_map in tqdm(dataset_records, desc="Compiling Data"):

        # Unpack task_name and hard_prompt
        task_name = task_map['task_name']
        # hard_prompt = task_map['reduced_instructions']
        hard_prompt = task_map['instruction']
        instances = task_map['instances']

        # If we encounter a new task name
        # Then we refresh the soft prompt tensor
        if task_name != last_task_name:
            last_task_name = task_name # Update last task name

            # Construct a path to fetch to trained the prompts for the task name
            # soft_prompt_path = os.path.join(trained_soft_prompts_dir, dataset_name, task_name, "softprompt.pt")
            soft_prompt_path = os.path.join(trained_soft_prompts_dir, task_name, "softprompt.pt")
        
            # Skip if the soft prompt doesn't exist for the current task name
            if not os.path.exists(soft_prompt_path):
                missing_count += 1
                tqdm.write(f"Warning: Missing soft prompts for task: {task_name}")
                continue
            
            # Load the saved state dict
            # weights_only=True is a PyTorch security best practice for loading tensors
            state_dict = torch.load(soft_prompt_path, map_location="cpu", weights_only=True)
            
            # Extract the prompt embeddings. 
            # The SoftPrompt class saves it as shape (1, num_tokens, embed_dim).
            soft_prompt_tensor = state_dict['prompt_embeddings'].squeeze(0)         # (num_tokens, embed_dim)

            # Extract the prompt initial_embeddings
            soft_prompt_init_embeddings = state_dict['initial_embeddings'].squeeze(0)  # (num_token, embed_dim)
        
        # Accumulate Dataset ID, Soft Prompt Tensor, and the Hard Prompt Tensor into the list of compiled data
        compiled_data.append({
            "task_name": task_name,
            "soft_prompt": soft_prompt_tensor,
            "soft_prompt_init_embeddings": soft_prompt_init_embeddings,
            "hard_prompt": hard_prompt,
            "instances": instances
        })

    # Log how many datasets were skipped
    if missing_count > 0:
        print(f"Skipped {missing_count} datasets (softprompt.pt not found).")

    return compiled_data



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
    parser.add_argument("--dataset_path", type=str, default="Suryanshg/SUPER-NATURALINSTRUCTIONS-english-filtered-100x-augmented")
    parser.add_argument("--trained_soft_prompts_dir", type=str, default="./trained_soft_prompts/SUPER-NATURALINSTRUCTIONS-english-filtered")
    parser.add_argument("--compiled_dataset_dir", type=str, default="./datasets/mapper_training_dataset")
    parser.add_argument("--num_instances", type=int, default=10)
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    DATASET_PATH = args.dataset_path
    TRAINED_SOFT_PROMPTS_DIR = args.trained_soft_prompts_dir
    COMPILED_DATASET_DIR = args.compiled_dataset_dir
    NUM_INSTANCES = args.num_instances
    SEED = args.seed

    # Determine Dataset Name
    DATASET_NAME = DATASET_PATH.split('/')[-1]

    # Fetch all hard prompts from Hugging Face Dataset
    hf_dataset = load_dataset(DATASET_PATH).select_columns(['task_name', 'instruction', 'reduced_instructions', 'input', 'output'])
    
    # Convert to Pandas
    train_dataset_df = hf_dataset['train'].to_pandas()
    test_dataset_df = hf_dataset['test'].to_pandas()

    # Add instruction field to reduced_instructions for train_df
    train_dataset_df['reduced_instructions'] = train_dataset_df.apply(
        lambda row: list(row['reduced_instructions']) + [row['instruction']],
        axis=1 # Apply row by row
    )


    # # Pick only 1 augmentation hard Prompt
    # # Sample 1 reduced_instruction per task_name
    # sampled_instructions = train_dataset_df.groupby('task_name')['reduced_instructions'].first().apply(
    #     lambda row: random.sample(list(row), 1)
    # )
    
    # # Map the chosen sampled instruction back to all rows of the corresponding task
    # train_dataset_df['reduced_instructions'] = train_dataset_df['task_name'].map(sampled_instructions)
    
    # Drop the instruction column in train_df and reduced_instructions in test_df
    train_dataset_df = train_dataset_df.drop(columns=['instruction'], axis=1)
    test_dataset_df = test_dataset_df.drop(columns=['reduced_instructions'], axis=1)

    # Explode the reduced instructions for train_df and rename to instructions
    train_dataset_df = train_dataset_df.explode('reduced_instructions').rename(columns={
        'reduced_instructions': 'instruction'
    })

    # Group by task and instruction, and take the first NUM_INSTANCE rows from each group
    train_dataset_df = train_dataset_df.groupby(['task_name', 'instruction']).head(NUM_INSTANCES).reset_index(drop=True)
    test_dataset_df = test_dataset_df.groupby(['task_name', 'instruction']).head(NUM_INSTANCES).reset_index(drop=True)

    # Group by task and instruction and fold the input/output pairs into an 'instances' column
    train_dataset_df = train_dataset_df.groupby(['task_name', 'instruction']).apply(
        lambda x: x[['input', 'output']].to_dict('records')
    ).reset_index(name='instances')

    test_dataset_df = test_dataset_df.groupby(['task_name', 'instruction']).apply(
        lambda x: x[['input', 'output']].to_dict('records')
    ).reset_index(name='instances')


    # Create a List of {Dataset Task, Hard Prompt} Dicts for Train and Test sets
    train_dataset_records = train_dataset_df.to_dict(orient='records')
    test_dataset_records = test_dataset_df.to_dict(orient='records')

    print(f"Found {len(train_dataset_records)} hard prompts in training dataset")
    print(f"Found {len(test_dataset_records)} hard prompts in testing dataset")

    # Get Compiled Data for train and test sets
    train_compiled_data = compile_data_list(train_dataset_records, TRAINED_SOFT_PROMPTS_DIR, DATASET_NAME)
    test_compiled_data = compile_data_list(test_dataset_records, TRAINED_SOFT_PROMPTS_DIR, DATASET_NAME)

    # Save the Compiled Data List to a Torch File
    print(f"\nCompilation Complete! Successfully paired: {len(train_compiled_data)} train datasets and {len(test_compiled_data)} test datasets.")

    # Create the Directory for saving the datasets
    # save_dir = os.path.join(COMPILED_DATASET_DIR, DATASET_NAME)
    save_dir = os.path.join(COMPILED_DATASET_DIR, DATASET_NAME)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the Training and Validation Datasets
    train_dataset_path = os.path.join(save_dir, 'train_mapper_dataset.pt')
    val_dataset_path = os.path.join(save_dir, 'val_mapper_dataset.pt')
    
    torch.save(train_compiled_data, train_dataset_path)
    torch.save(test_compiled_data, val_dataset_path)
    
    print(f"Saved Train Split ({len(train_compiled_data)} samples) to: {train_dataset_path}")
    print(f"Saved Val Split ({len(test_compiled_data)} samples) to: {val_dataset_path}")
    



