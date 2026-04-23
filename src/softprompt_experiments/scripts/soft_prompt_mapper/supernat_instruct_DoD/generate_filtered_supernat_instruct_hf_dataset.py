import os
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import pandas as pd

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n",
        f"\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="Suryanshg/SUPER-NATURALINSTRUCTIONS-english")
    parser.add_argument("--teacher_model", type=str, default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--tokenizer_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--token_threshold", type=int, default=400, help="Max tokens allowed for input + output sentences; if higher, we filter them out")
    parser.add_argument("--min_instances_per_task", type=int, default=500, help="Min number of instances allowed per task for further experiments")
    args, _ = parser.parse_known_args(args_list)

    # Parse all arguments into Global Variables
    TOKENIZER_MODEL = args.tokenizer_model
    DATASET_PATH = args.dataset_path
    TOKEN_THRESHOLD = args.token_threshold
    MIN_INSTANCES_PER_TASK = args.min_instances_per_task
    FILTERED_DATASET_NAME = f"{DATASET_PATH}-filtered"

    print(f"Loading Tokenizer: {TOKENIZER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    
    print(f"Loading Dataset: {DATASET_PATH}...")
    hf_ds = load_dataset(DATASET_PATH)

    # Compute the token lengths for all instances
    def compute_length(example):
        text = f"Input: {example['input']}\nOutput: {example['output']}"
        # Setting truncation to False so we get the true length
        tokens = tokenizer.encode(text, truncation=False)
        return {"total_tokens": len(tokens)}

    print("Mapping token lengths across dataset (this may take a minute)...")
    hf_ds = hf_ds.map(compute_length, num_proc=8, desc="Counting Tokens")

    # Filter out any sequence longer than our threshold
    print(f"Filtering dataset to keep instances <= {TOKEN_THRESHOLD} tokens...")
    filtered_ds = hf_ds.filter(lambda x: x["total_tokens"] <= TOKEN_THRESHOLD, num_proc=8, desc="Filtering Lengths")

    # Convert all remaining validation/train splits to Pandas to calculate task distributions easily
    splits_to_concat = [filtered_ds[split].to_pandas() for split in filtered_ds.keys()]
    full_df = pd.concat(splits_to_concat, ignore_index=True)

    # Calculate remaining instances per task
    task_counts = full_df['task_name'].value_counts()
    
    # Find active tasks with >= MIN_INSTANCES
    valid_tasks = set(task_counts[task_counts >= MIN_INSTANCES_PER_TASK].index.tolist())
    print(f"Found {len(valid_tasks)} tasks that still have at least {MIN_INSTANCES_PER_TASK} instances.")

    # Filter the HF DatasetDict directly to retain existing splits
    final_hf_dataset = filtered_ds.filter(lambda x: x["task_name"] in valid_tasks, num_proc=8, desc="Filtering Valid Tasks")
    
    # Remove the temporary 'total_tokens' column
    final_hf_dataset = final_hf_dataset.remove_columns(["total_tokens"])
    
    total_instances = sum(len(final_hf_dataset[split]) for split in final_hf_dataset.keys())
    print(f"Final filtered dataset contains {total_instances} total instances preserving splits: {list(final_hf_dataset.keys())}")

    print(f"Attempting to push to Hugging Face Hub as: {FILTERED_DATASET_NAME}...")
    try:
        final_hf_dataset.push_to_hub(FILTERED_DATASET_NAME, private=False)
        print("Successfully pushed to Hub!")
    except Exception as e:
        print(f"\nCould not push to Hub. (Make sure you are logged in via `huggingface-cli login`). Error: {e}")

    print("\nDone!")