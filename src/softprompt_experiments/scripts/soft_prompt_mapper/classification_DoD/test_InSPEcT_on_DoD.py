import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.softprompt_experiments.InSPEcT_utils import elicit_description_using_inspect_technique
import pandas as pd
import string
from tqdm import tqdm


def calculate_eval_metrics(elicited_text, hard_prompt):
    """Calculates Class Rate, and F1 Score as defined by the InSPEcT methodology."""
    classes = hard_prompt.split(",")

    # Calculate Class Rate (Recall)
    clean_text = elicited_text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = set(clean_text.split())
    
    classes_count = sum(1 for c in classes if c.lower() in words)
    class_rate = classes_count / len(classes) if classes else 0.0
    
    # Calculate Precision and F1 Score
    precision = classes_count / len(words) if words else 0.0
    f1_score = 2 * (precision * class_rate) / (precision + class_rate) if (precision + class_rate) > 0 else 0.0
            
    return class_rate, f1_score


def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft_prompts_dataset_path", type=str, default="./datasets/mapper_training_dataset/DoD_3_5k_peft")
    parser.add_argument("--soft_prompts_dir", type=str, default="./trained_soft_prompts/DoD_3_5k_peft")
    parser.add_argument("--results_save_dir", type=str, default="./inspect_results")
    parser.add_argument("--num_training_examples", type=int, default=50)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--peft", action="store_true", help="Use PEFT style way of loading soft prompts")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    NUM_TOKENS = args.num_tokens
    SOFT_PROMPTS_DATASET_PATH = args.soft_prompts_dataset_path
    DOD_NAME = ''.join(SOFT_PROMPTS_DATASET_PATH.split('/')[-1])

    SOFT_PROMPTS_DIR = args.soft_prompts_dir
    NUM_TRAINING_EXAMPLES = args.num_training_examples
    RESULTS_SAVE_DIR = args.results_save_dir + f"/{DOD_NAME}"

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Loading Training accuracy stats
    TRAINING_STATS_DF = pd.read_csv(os.path.join(SOFT_PROMPTS_DIR, "accuracy_stats.csv"))

    # ┌───────────────────────────────────────────────┐
    # │              INSPECT MODEL PREP               │
    # └───────────────────────────────────────────────┘
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {MODEL_NAME} for InSPEcT...")
    inspect_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    inspect_model.eval()


    # ┌───────────────────────────────────────────────┐
    # │                LOAD DATASETS                  │
    # └───────────────────────────────────────────────┘
    val_dataset = torch.load(os.path.join(SOFT_PROMPTS_DATASET_PATH, 'val_mapper_dataset.pt'), map_location="cpu", weights_only=True)
    print(f"Validation Dataset size: {len(val_dataset)}")


    BEST_LAYER_PAIR = [
        {"min_source": 18, "max_source": 18, "min_target": 28, "max_target": 28}
    ]

    # ┌───────────────────────────────────────────────┐
    # │      APPLY BEST LAYER PAIR TO TEST PROMPTS    │
    # └───────────────────────────────────────────────┘
    # List to hold the summary of best metrics across all datasets
    summary_results = []

    for example_idx, data in enumerate(tqdm(val_dataset, desc="Performing InSPEcT on Test Soft Prompts")):
        dataset_id = data["dataset_id"]
        soft_prompt = data["soft_prompt"] # shape (1, soft_prompt_len, embed_dim)
        hard_prompt = data["hard_prompt"]

        # Get Elicited Text using InSPEcT Technique
        inspect_elicited_results = elicit_description_using_inspect_technique(
            model=inspect_model,
            tokenizer=tokenizer,
            num_tokens=NUM_TOKENS,
            soft_prompt=soft_prompt,
            dataset_name="REPLACE_ME",
            layer_combinations=BEST_LAYER_PAIR,
            target_prompt_type='few_shot'
        )

        # Evaluate InSPEcT results
        for i in range(len(inspect_elicited_results)):
            output_text = str(inspect_elicited_results[i]['output'])

            # Get all scores for the output text by InSPEcT
            class_rate, f1_score = calculate_eval_metrics(output_text, hard_prompt)
            inspect_elicited_results[i]['class_rate'] = class_rate
            inspect_elicited_results[i]['f1_score'] = f1_score

        # Find the row with the highest class_rate
        max_class_rate_row = max(inspect_elicited_results, key=lambda x: x['class_rate'])

        # Find the row with the highest f1_score
        max_f1_score_row = max(inspect_elicited_results, key=lambda x: x['f1_score'])

        # Retrieve the training stats for this dataset
        training_stats_df = TRAINING_STATS_DF[TRAINING_STATS_DF["dataset_id"] == dataset_id]

        # Save Elicitations using for this dataset
        os.makedirs(f"{RESULTS_SAVE_DIR}/test", exist_ok=True)
        df = pd.DataFrame(inspect_elicited_results)
        df.to_csv(f'{RESULTS_SAVE_DIR}/test/{dataset_id}_elicitations.csv', index=False)

        result_entry = {
            "dataset": dataset_id,
            "val_accuracy": training_stats_df['val_accuracy'].iloc[0] if len(training_stats_df) > 0 else None,
            "max_class_rate": round(max_class_rate_row['class_rate'], 4),
            "max_class_rate_src_layer": max_class_rate_row['source_layer'],
            "max_class_rate_tgt_layer": max_class_rate_row['target_layer'],
            "max_f1_score": round(max_f1_score_row['class_rate'], 4),
            "max_f1_score_src_layer": max_f1_score_row['source_layer'],
            "max_f1_score_tgt_layer": max_f1_score_row['target_layer'],
        }

        summary_results.append(result_entry)

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = f"{RESULTS_SAVE_DIR}/inspect_val_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSaved master summary with best metrics to: {summary_csv_path}")






