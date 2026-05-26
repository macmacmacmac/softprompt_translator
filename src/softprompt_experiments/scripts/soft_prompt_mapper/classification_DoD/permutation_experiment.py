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
    parser.add_argument("--json_results_path", type=str, default="./DoD_3_5k_permutation_results.json")
    parser.add_argument("--sample", action='store_true', help="Use a sample of val dataset instead of the full val dataset")
    parser.add_argument("--permute", action='store_true', help="Permute order of soft prompt tokens")
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
    SEED = args.seed
    DATASET_NAME = LORA_DIR.split('/')[-1]
    PERMUTE = args.permute

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
    original_summaries = []
    left_permuted_summaries = []
    right_permuted_summaries = []
    full_permuted_summaries = []
    reverse_full_summaries = []
    reverse_second_half_summaries = []

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


                
            # Permute the first half
            indices = torch.randperm(10)
            prefix = soft_prompt[indices]
            suffix = soft_prompt[10:]
            left_permuted_soft_prompt = torch.cat([prefix, suffix])

            # Permute the last half
            indices = torch.randperm(10) + 10
            prefix = soft_prompt[:10]
            suffix = soft_prompt[indices]
            right_permuted_soft_prompt = torch.cat([prefix, suffix])

            # Permute the whole
            indices = torch.randperm(soft_prompt.size(0))
            full_permuted_soft_prompt = soft_prompt[indices]

            # Reverse order
            # Reverse full
            reverse_full_soft_prompt = torch.flip(soft_prompt, dims=[0])

            # Reverse second half
            reverse_second_half_soft_prompt = torch.cat([
                soft_prompt[:10],
                torch.flip(soft_prompt[10:], dims=[0])
            ])


            original_summary = operate_on_soft_prompt(
                soft_prompt,
                true_keywords,
                model,
                tokenizer,
                method = "original"
            )

            left_permuted_summary = operate_on_soft_prompt(
                left_permuted_soft_prompt,
                true_keywords,
                model,
                tokenizer,
                method = "left_permuted"
            )

            right_permuted_summary = operate_on_soft_prompt(
                right_permuted_soft_prompt,
                true_keywords,
                model,
                tokenizer,
                method = "right_permuted"
            )

            full_permuted_summary = operate_on_soft_prompt(
                full_permuted_soft_prompt,
                true_keywords,
                model,
                tokenizer,
                method = "full_permuted"
            )


            reverse_full_summary = operate_on_soft_prompt(
                reverse_full_soft_prompt,
                true_keywords,
                model,
                tokenizer,
                method = "reverse_full"
            )

            
            reverse_second_half_summary = operate_on_soft_prompt(
                reverse_second_half_soft_prompt,
                true_keywords,
                model,
                tokenizer,
                method = "reverse_second_half"
            )


            # Accumulate individual summaries
            original_summaries.append(original_summary)
            left_permuted_summaries.append(left_permuted_summary)
            right_permuted_summaries.append(right_permuted_summary)
            full_permuted_summaries.append(full_permuted_summary)
            reverse_full_summaries.append(reverse_full_summary)
            reverse_second_half_summaries.append(reverse_second_half_summary)
       
            # Accumulate results for summarization
            json_results.append({
                "dataset": dataset_id,
                "hard_prompt": true_keywords,
                "validation_accuracy": val_accuracy,
                "original_summary": original_summary,
                "left_permuted_summary": left_permuted_summary,
                "right_permuted_summary": right_permuted_summary,
                "full_permuted_summary": full_permuted_summary,
                "revsere_full_summary": reverse_full_summary,
                "reverse_second_half_summary": reverse_second_half_summary
            })

            tqdm.write("-" * 80)

    # Print out the averages
    calculate_avg_metrics(original_summaries, "original")
    calculate_avg_metrics(left_permuted_summaries, "left_permuted")
    calculate_avg_metrics(right_permuted_summaries, "right_permuted")
    calculate_avg_metrics(full_permuted_summaries, "full_permuted")
    calculate_avg_metrics(reverse_full_summaries, "reverse_full")
    calculate_avg_metrics(reverse_second_half_summaries, "reverse_second_half")


    # Save a JSON file for summary results
    if json_results:
        with open(JSON_RESULTS_PATH, 'w') as f:
            json.dump(json_results, f, indent=4)
