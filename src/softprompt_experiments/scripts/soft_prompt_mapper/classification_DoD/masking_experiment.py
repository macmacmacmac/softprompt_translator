import os
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
import string
from tqdm import tqdm
import json


def operate_on_soft_prompt(soft_prompt, 
                           true_keywords, 
                           model, 
                           tokenizer,
                           method = "Not Provided"):
    # Format for the model: Add batch dimension
    inputs_embeds = soft_prompt.unsqueeze(0).to(model.device, dtype=model.dtype)    # (1, seq_len, embed_dim)
    
    # Create an attention mask of 1s for the seq_len tokens
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=model.device) # (1, seq_len)
    
    # Generate the discrete text
    # Using greedy decoding (temperature=0.0)
    outputs = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=20,
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated token IDs back into an English string
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Clean and split target keywords, stripping punctuation from the ground truth
    raw_target_words = [w.strip().lower() for w in true_keywords.split(",") if w.strip()]
    target_set = set([w.translate(str.maketrans('', '', string.punctuation)) for w in raw_target_words])
    clean_pred = pred_text.translate(str.maketrans('', '', string.punctuation)).lower()
    pred_set = set(clean_pred.split())

    # Calculate Overlap
    overlap = target_set.intersection(pred_set)
    
    # Calculate Recall, Precision, and F1
    recall = len(overlap) / len(target_set) if len(target_set) > 0 else 0
    precision = len(overlap) / len(pred_set) if len(pred_set) > 0 else 0
    
    # Calculate F1 score (if applicable)
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    tqdm.write('-' * 50)
    tqdm.write(f"Method         : {method}")
    tqdm.write(f"Verbalization  : {pred_text}")
    tqdm.write(f"Metrics        : Recall: {recall:.2f} | Precision: {precision:.2f} | F1: {f1_score:.2f}")
    tqdm.write('-' * 50)

    return {
        "verbalization": pred_text,
        "f1": f1_score,
        "recall": recall,
        "precision": precision
    }


def calculate_avg_metrics(method_summaries, method):
    total_f1 = sum(summary["f1"] for summary in method_summaries)
    total_recall = sum(summary["recall"] for summary in method_summaries)
    print('-' * 50)
    print(f"Method: {method}, Avg Recall: {total_recall/len(method_summaries)}, Avg F1: {total_f1/len(method_summaries)}")
    

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
    parser.add_argument("--val_dataset_path", type=str, default="./datasets/mapper_training_dataset/DoD_3_5k/val_mapper_dataset.pt")
    parser.add_argument("--lora_dir", type=str, default="./mapper_lora_weights/DoD_3_5k")
    parser.add_argument("--training_stats_path", type=str, default="./trained_soft_prompts/DoD_3_5k/accuracy_stats.csv")
    parser.add_argument("--json_results_path", type=str, default="./DoD_3_5k_masking_results.json")
    parser.add_argument("--sample", action='store_true', help="Use a sample of val dataset instead of the full val dataset")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    VAL_DATASET_PATH = args.val_dataset_path
    LORA_DIR = args.lora_dir
    TRAINING_STATS_PATH = args.training_stats_path
    JSON_RESULTS_PATH = args.json_results_path
    NUM_SAMPLES = args.num_samples
    NUM_TOKENS = args.num_tokens
    SEED = args.seed
    DATASET_NAME = LORA_DIR.split('/')[-1]

    # Set the Seed for this experiment
    random.seed(SEED)

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Training Stats
    TRAINING_STATS_DF = pd.read_csv(TRAINING_STATS_PATH, index_col='dataset_id')

    # ┌───────────────────────────────────────────────┐
    # │                   DATASET PREP                │
    # └───────────────────────────────────────────────┘
    print(f"Loading Validation dataset from {VAL_DATASET_PATH}...")
    val_dataset = torch.load(VAL_DATASET_PATH, map_location="cpu", weights_only=True)
    
    print(f"Validation Dataset size: {len(val_dataset)}")

    # If inference is requested on a sample of the validation dataset
    if args.sample:
        # Pick a random subset to evaluate
        test_samples = random.sample(val_dataset, min(NUM_SAMPLES, len(val_dataset)))

    else:
        test_samples = val_dataset

    # ┌───────────────────────────────────────────────┐
    # │                 LORA MODEL PREP               │
    # └───────────────────────────────────────────────┘
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)

    print(f"Loading LoRA adapters from {LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()

    # ┌───────────────────────────────────────────────┐
    # │                 INFERENCE LOOP                │
    # └───────────────────────────────────────────────┘
    print("\n" + "="*80)
    print("\t\t\tGENERATION RESULTS")
    print("="*80 + "\n")

    json_results = []
    all_masked_summaries = {i: [] for i in range(NUM_TOKENS, (NUM_TOKENS // 2) - 1 , -1)}

    with torch.no_grad():
        for sample in tqdm(test_samples, desc = "Mapping Soft Prompts"):
            # Extract the data
            dataset_id = sample["dataset_id"]
            soft_prompt = sample["soft_prompt"]    # (seq_len, embed_dim)
            true_keywords = sample["hard_prompt"]  
            val_accuracy = TRAINING_STATS_DF.at[dataset_id, 'val_accuracy']

            tqdm.write(f"Dataset ID: {dataset_id}")
            tqdm.write(f"Soft Prompt Validation Accuracy: {val_accuracy}")
            tqdm.write(f"Target Keywords                : {true_keywords}")

            masked_summaries = {}
            seq_len = soft_prompt.shape[0]
            
            for keep_len in range(seq_len, 9, -1):
                truncated_prompt = soft_prompt[:keep_len, :]
                method_name = f"truncated_{keep_len}"
                
                summary = operate_on_soft_prompt(
                    truncated_prompt,
                    true_keywords,
                    model,
                    tokenizer,
                    method = method_name
                )
                
                all_masked_summaries[keep_len].append(summary)
                masked_summaries[method_name] = summary

       
            # Accumulate results for summarization
            json_results.append({
                "dataset": dataset_id,
                "hard_prompt": true_keywords,
                "validation_accuracy": val_accuracy,
                "masked_summaries": masked_summaries,
            })

            tqdm.write("-" * 80)

    # Print out the averages
    for keep_len in range(20, 9, -1):
        if all_masked_summaries[keep_len]:
            calculate_avg_metrics(all_masked_summaries[keep_len], f"truncated_{keep_len}")



    # Save a JSON file for summary results
    if json_results:
        with open(JSON_RESULTS_PATH, 'w') as f:
            json.dump(json_results, f, indent=4)
