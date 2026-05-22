import torch
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm
from scipy.stats import pearsonr
import numpy as np

from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.models.squishyprompt import SquishyPrompt
from softprompt_experiments.utils import (
    get_train_test_from_tokenized, 
    log_json
)

import json
import evaluate
from dotenv import load_dotenv
from tqdm import tqdm
import re
from openai import OpenAI

load_dotenv()  
ROUGE_METRIC = evaluate.load("rouge")
torch.manual_seed(42)


fewshot_prompt_template = """
Based on the following examples, predict the answer for the given input
{example_str}

Here is the input, predict the answer. Do NOT output preamble or explanations or anything else. 
{input_text}
"""
combined_template = """
You will be given a clue about the task and some examples. 
Based on these, predict the answer for the given input

Clue:
{mapper_prompt}

Examples:
{example_str}

Here is the input, now predict the answer. Do NOT output preamble or explanations or anything else. 
{input_text}
"""

# -----------------------------
# LLM call
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found.")

client = OpenAI(api_key=api_key)

def get_llm_prediction(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful math assistant. Follow the task exactly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,  # greedy
    )

    return response.choices[0].message.content.strip()

def parse_first_number(text: str):
    """
    Extract the first number (possibly negative) from a generated string.
    Returns None if no number is found.
    """
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group())
    return None    

def get_metrics_from_preds_target(preds, targets):
    # Convert to numpy
    preds = np.array(preds)
    targets = np.array(targets)

    # Compute metrics
    mse = np.mean((preds - targets)**2)
    mae = np.mean(np.abs(preds - targets))
    if len(preds) > 1:
        r, p = pearsonr(preds, targets)
    else:
        r, p = float("nan"), float("nan")

    metrics = {
        "mse": mse,
        "mae": mae,
        "pearson_r": r,
        "pearson_p": p,
    }
    return metrics

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--save_directory", type=str, default="./datasets/simple_math_vision")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--show_target", type=bool, default=False)
    parser.add_argument("--no_auto_split",dest="auto_split",action="store_false")
    parser.set_defaults(auto_split=True)

    args, _ = parser.parse_known_args(args_list)

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    SAVE_DIR = args.save_directory
    BATCH_SIZE = args.batch_size
    AUTO_SPLIT = args.auto_split

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token    

    # Get dataset sub directories
    dataset_dirs = []
    for entry in os.scandir(SAVE_DIR):
        if entry.is_dir():  # Check if the entry is a directory
            if "dataset_" in entry.name:
                dataset_dirs.append(entry.path)

    num_datasets = len(dataset_dirs)
    if num_datasets > 0:
        print(f"\nFound ({num_datasets}) datasets in directory")
    else:
        raise ValueError("path to directory has no datasets")

    for dataset_dir in tqdm(dataset_dirs):
        train_dataset, test_dataset, train_loader, test_loader = get_train_test_from_tokenized(
            dataset_dir,
            BATCH_SIZE,
            train_portion = 0.8,
            auto_split=AUTO_SPLIT
        )
        with open(os.path.join(dataset_dir,'mapper_preds.json')) as f:
            task_prompts = json.load(f)

        # Build few shot examples
        random_idxs = torch.randint(0, len(test_dataset), (3,))
        example_str = ""
        for i, idx in enumerate(random_idxs):
            full_ids = test_dataset[idx][0]
            decoded_example = tokenizer.decode(full_ids, skip_special_tokens=True)
            example_str += f"\nExample {i}:\n{decoded_example}\n"

        # Inference
        preds_fewshot = []
        preds_mapper = []
        preds_combined = []
        targets = []
        for full_ids, labels in tqdm(test_dataset):
            target_mask = (labels != -100)

            # Extract input-only ids for generation
            only_input_ids = full_ids[~target_mask] 
            target_ids = full_ids[target_mask]

            # Decode true target text and parse integer
            input_text = tokenizer.decode(only_input_ids, skip_special_tokens=True)
            true_text = tokenizer.decode(target_ids, skip_special_tokens=True)

            # true val-------------------------------------------
            true_val = parse_first_number(true_text)
            targets.append(true_val)
            if true_val is None:
                # If dataset is correct this should never happen
                continue
            
            # few shot-------------------------------------------
            fewshot_input = fewshot_prompt_template.format(
                example_str=example_str,
                input_text=input_text
            )
            fewshot_text = get_llm_prediction(fewshot_input)
            pred_fewshot = parse_first_number(fewshot_text)
            # If model outputs no number, skip or set to 0; we choose skip
            if pred_fewshot is None:
                continue
            preds_fewshot.append(pred_fewshot)

            # mapper---------------------------------------------
            mapper_input = mapper_prompt_template.format(
                mapper_prompt=task_prompts['mapper_verbalization_prefilled'],
                input_text=input_text
            )
            mapper_text = get_llm_prediction(mapper_input)
            pred_mapper = parse_first_number(mapper_text)
            # If model outputs no number, skip or set to 0; we choose skip
            if pred_mapper is None:
                continue
            preds_mapper.append(pred_mapper)

            # combined---------------------------------------------
            combined_input = combined_template.format(
                mapper_prompt=task_prompts['mapper_verbalization_prefilled'],
                example_str=example_str,
                input_text=input_text
            )
            combined_text = get_llm_prediction(combined_input)
            pred_combined = parse_first_number(combined_text)
            # If model outputs no number, skip or set to 0; we choose skip
            if pred_combined is None:
                continue
            preds_combined.append(pred_combined)



        # metrics here
        metrics_fewshot = get_metrics_from_preds_target(preds_fewshot, targets)
        metrics_mapper = get_metrics_from_preds_target(preds_mapper, targets)
        metrics_combined = get_metrics_from_preds_target(preds_combined, targets)

        # print(f"fewshot MSE: {metrics_fewshot['mse']}")
        # print(f"mapper MSE: {metrics_mapper['mse']}")

        task_prompts["metrics_using_fewshot"] = metrics_fewshot 
        task_prompts["metrics_using_mapper"] = metrics_mapper
        task_prompts["metrics_using_combined"] = metrics_combined
        log_json(os.path.join(dataset_dir, "mapper_preds.json"), task_prompts)


    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









