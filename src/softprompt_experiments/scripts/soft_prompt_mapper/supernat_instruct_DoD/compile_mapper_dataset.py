import os
import argparse
from tqdm import tqdm
import torch
from datasets import load_dataset
from typing import List, Dict, Any
import random


def build_task_records(split_df, num_examples: int, seed: int, num_train_instances: int) -> List[Dict]:
    """
    Reproduce train_softprompts.py's per-task sample + 90/10 split so that
    validation_instances are the exact rows the soft prompt was RougeL-scored on,
    and training_instances are drawn from the rows the soft prompt trained on.
    Returns one record per task_name.
    """
    records = []
    grouped = split_df.groupby('task_name')

    for task_name in grouped.groups.keys():
        task_df = grouped.get_group(task_name)

        # Mirror train_softprompts.py exactly: cap to num_examples (seeded), then 90/10 split.
        # sample() selects by position, so identical row order + count + seed => identical rows.
        if len(task_df) > num_examples:
            task_df = task_df.sample(n=num_examples, random_state=seed)

        split_idx = int(len(task_df) * 0.9)
        train_rows = task_df.iloc[:split_idx]
        val_rows = task_df.iloc[split_idx:]

        train_instances = train_rows[['input', 'output']].head(num_train_instances).to_dict('records')
        train_instances = train_rows[['input', 'output']].to_dict('records')
        val_instances = val_rows[['input', 'output']].to_dict('records')

        # reduced_instructions explosion is disabled -> one instruction per task
        instruction = task_df['instruction'].iloc[0]

        records.append({
            'task_name': task_name,
            'instruction': instruction,
            'train_instances': train_instances,
            'val_instances': val_instances
        })

    return records


def compile_data_list(dataset_records: List[Dict[str, Any]],
                      trained_soft_prompts_dir: str) -> List[Dict]:

    compiled_data = []
    missing_count = 0
    last_task_name = ""

    print(f"Scanning soft prompt directories in: {trained_soft_prompts_dir} ...")

    # For each Task Name and its associated hard prompt in the Train Dataset Map    
    for task_map in tqdm(dataset_records, desc="Compiling Data"):

        # Unpack task_name and hard_prompt
        task_name = task_map['task_name']
        hard_prompt = task_map['instruction']
        train_instances = task_map['train_instances']
        val_instances = task_map['val_instances']

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
            "train_instances": train_instances,
            "val_instances": val_instances
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
    parser.add_argument("--dataset_path", type=str, default="SoftPromptTranslator/SUPER-NATURALINSTRUCTIONS-english-filtered")
    parser.add_argument("--trained_soft_prompts_dir", type=str, default="./shared/trained_soft_prompts/General-DoD")
    parser.add_argument("--compiled_dataset_dir", type=str, default="./shared/datasets/mapper_training_dataset/General-DoD-DPO")
    parser.add_argument("--num_examples", type=int, default=500, help="Per-task cap; MUST match train_softprompts.py to reproduce its val split")
    parser.add_argument("--num_train_instances", type=int, default=5, help="Number of training-split instances to store per task")
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    DATASET_PATH = args.dataset_path
    TRAINED_SOFT_PROMPTS_DIR = args.trained_soft_prompts_dir
    COMPILED_DATASET_DIR = args.compiled_dataset_dir
    NUM_EXAMPLES = args.num_examples
    NUM_TRAIN_INSTANCES = args.num_train_instances
    SEED = args.seed

    # Fetch all hard prompts from Hugging Face Dataset
    hf_dataset = load_dataset(DATASET_PATH).select_columns([
        'task_name', 
        'instruction',
        # 'reduced_instructions',
        'input', 
        'output'
    ])
    
    # Convert to Pandas
    train_dataset_df = hf_dataset['train'].to_pandas()
    test_dataset_df = hf_dataset['test'].to_pandas()

    # Add instruction field to paraphrased_instructions for train_df
    # train_dataset_df['reduced_instructions'] = train_dataset_df.apply(
    #     lambda row: list(row['reduced_instructions']) + [row['instruction']],
    #     axis=1 # Apply row by row
    # )
    
    # Drop the instruction column in train_df and reduced_instructions in test_df
    # train_dataset_df = train_dataset_df.drop(columns=['instruction'], axis=1)
    # test_dataset_df = test_dataset_df.drop(columns=['reduced_instructions'], axis=1)

    # Explode (unwind) the reduced instructions for train_df and rename column to 'instruction'
    # train_dataset_df = train_dataset_df.explode('reduced_instructions').rename(columns={
    #     'reduced_instructions': 'instruction'
    # })

    # Reproduce train_softprompts.py's per-task split so validation_instances match the
    # exact rows each soft prompt was RougeL-scored on (+ a few training-split instances).
    train_dataset_records = build_task_records(train_dataset_df, NUM_EXAMPLES, SEED, NUM_TRAIN_INSTANCES)
    test_dataset_records = build_task_records(test_dataset_df, NUM_EXAMPLES, SEED, NUM_TRAIN_INSTANCES)

    print(f"Found {len(train_dataset_records)} hard prompts in training dataset")
    print(f"Found {len(test_dataset_records)} hard prompts in testing dataset")

    # Get Compiled Data for train and test sets
    train_compiled_data = compile_data_list(train_dataset_records, TRAINED_SOFT_PROMPTS_DIR)
    test_compiled_data = compile_data_list(test_dataset_records, TRAINED_SOFT_PROMPTS_DIR)

    # Save the Compiled Data List to a Torch File
    print(f"\nCompilation Complete! Successfully paired: {len(train_compiled_data)} train datasets and {len(test_compiled_data)} test datasets.")
    # print(f"\nCompilation Complete! Successfully paired: {len(test_compiled_data)} test datasets.")

    # Create the Directory for saving the datasets
    os.makedirs(COMPILED_DATASET_DIR, exist_ok=True)
    
    # Save the Training and Validation Datasets
    train_dataset_path = os.path.join(COMPILED_DATASET_DIR, 'train_mapper_dataset.pt')
    val_dataset_path = os.path.join(COMPILED_DATASET_DIR, 'val_mapper_dataset.pt')
    
    torch.save(train_compiled_data, train_dataset_path)
    torch.save(test_compiled_data, val_dataset_path)
    
    print(f"Saved Train Split ({len(train_compiled_data)} samples) to: {train_dataset_path}")
    print(f"Saved Val Split ({len(test_compiled_data)} samples) to: {val_dataset_path}")
    



