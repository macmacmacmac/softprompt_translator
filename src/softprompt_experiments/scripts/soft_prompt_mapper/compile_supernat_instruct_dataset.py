import os
import argparse
from datasets import Dataset, DatasetDict
from typing import List, Dict
import json
from tqdm import tqdm

def task_data_generator(tasks_dir, task_names: List[str]):
    
    # For every task name
    for task_name in tqdm(task_names):

        # Construct a path for the respective JSON file
        task_file_path = os.path.join(tasks_dir, f"{task_name}.json")
        
        # Skip if file doesn't exist just in case
        if not os.path.exists(task_file_path):
            print(f"Warning: {task_file_path} not found.")
            continue
            
        # Load the JSON file
        with open(task_file_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)
            
        # Extract useful fields from the JSON file
        # Singular Fields (Strings)
        definition = task_data.get("Definition", [""])[0]

        # Multi-Value Fields (Lists) that can be represented as strings
        input_language = convert_list_field_to_str(task_data.get("Input_language", []))
        output_language = convert_list_field_to_str(task_data.get("Output_language", []))
        instruction_language = convert_list_field_to_str(task_data.get("Instruction_language", []))
        categories = convert_list_field_to_str(task_data.get("Categories", []))

        # Multi-Value Fields (Lists) which are examples for learning
        instances = task_data.get("Instances", [])
        

        # For each instance, extract its output list
        for instance in instances:
            output_list = instance.get("output", [])
            
            # For each output per instance, create a new row
            for output in output_list:
                yield {
                    "task_name": task_name,
                    "instruction": definition,
                    "input_language": input_language,
                    "output_language": output_language,
                    "instruction_language": instruction_language,
                    "categories": categories,
                    "input": instance.get("input", ""),
                    "output": output,
                }


def convert_list_field_to_str(field: List[str]):
    if len(field) > 1:
        return ", ".join(field)
    elif len(field) == 1:
        return field[0]
    else:
        return ""


def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n",
        f"\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_dir_path", type=str, default="/home/sgoyal/natural-instructions")
    parser.add_argument("--english_only", action="store_true", help="Parse English only tasks")
    args, _ = parser.parse_known_args(args_list)


    # Parse all arguments into Global Variables
    LOCAL_DATASET_DIR_PATH = args.local_dataset_dir_path
    ENGLISH_ONLY = args.english_only

    # Derived Variables
    if ENGLISH_ONLY:
        SPLITS_DIR = os.path.join(LOCAL_DATASET_DIR_PATH, "splits", "default")
    else:
        SPLITS_DIR = os.path.join(LOCAL_DATASET_DIR_PATH, "splits", "xlingual")
    TASKS_DIR = os.path.join(LOCAL_DATASET_DIR_PATH, "tasks")

    # Load Train / Test Task Names
    train_tasks_path = os.path.join(SPLITS_DIR, "train_tasks.txt")
    test_tasks_path = os.path.join(SPLITS_DIR, "test_tasks.txt")

    # Read the files and create lists of paths, removing any extra whitespace/newlines
    with open(train_tasks_path, "r", encoding="utf-8") as f:
        train_tasks = f.read().splitlines()
        
    with open(test_tasks_path, "r", encoding="utf-8") as f:
        test_tasks = f.read().splitlines()

    print(f"Loaded {len(train_tasks)} Train Task Names")
    print(f"Loaded {len(test_tasks)} Test Task Names")

    # Compile Training Data
    print("Compiling train data...")
    train_dataset = Dataset.from_generator(
        lambda: task_data_generator(TASKS_DIR, train_tasks)
    )
    print(f"Compiled {len(train_dataset)} examples for training")

    # Compile Testing Data
    print("Compiling testing data...")
    test_dataset = Dataset.from_generator(
        lambda: task_data_generator(TASKS_DIR, test_tasks)
    )
    print(f"Compiled {len(test_dataset)} examples for testing")

    # Convert to HF Dataset
    hf_dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Push to HF Hub
    suffix = "english" if ENGLISH_ONLY else "xlingual"
    hf_dataset_dict.push_to_hub(f"Suryanshg/SUPER-NATURALINSTRUCTIONS-{suffix}")