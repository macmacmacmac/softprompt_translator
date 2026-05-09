import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.softprompt_experiments.InSPEcT_utils import elicit_description_using_inspect_technique, ALL_LAYER_COMBINATIONS, BEST_PATCHES
import pandas as pd
import nltk
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
import string

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))
ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)


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
    parser.add_argument("--soft_prompts_dataset_path", type=str, default="./datasets/mapper_training_dataset/DoD_3_5k")
    parser.add_argument("--training_stats_path", type=str, default="./trained_soft_prompts/DoD_3_5k/accuracy_stats.csv")
    parser.add_argument("--num_training_examples", type=int, default=100)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--peft", action="store_true", help="Use PEFT style way of loading soft prompts")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    NUM_TOKENS = args.num_tokens
    SOFT_PROMPTS_DATASET_PATH = args.soft_prompts_dataset_path
    TRAINING_STATS_PATH = args.training_stats_path
    NUM_TRAINING_EXAMPLES = args.num_training_examples
    DOD_NAME = ''.join(SOFT_PROMPTS_DATASET_PATH.split('/')[-1])

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Loading Training accuracy stats
    TRAINING_STATS_DF = pd.read_csv(TRAINING_STATS_PATH)

    # ┌───────────────────────────────────────────────┐
    # │              INSPECT MODEL PREP               │
    # └───────────────────────────────────────────────┘
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {MODEL_NAME} for InSPEcT...")
    inspect_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    inspect_model.eval()


    # ┌───────────────────────────────────────────────┐
    # │           SOFT PROMPT DATASET PREP            │
    # └───────────────────────────────────────────────┘
    train_dataset = torch.load(os.path.join(SOFT_PROMPTS_DATASET_PATH, 'train_mapper_dataset.pt'), map_location="cpu", weights_only=True)
    val_dataset = torch.load(os.path.join(SOFT_PROMPTS_DATASET_PATH, 'val_mapper_dataset.pt'), map_location="cpu", weights_only=True)

    print(f"Train Dataset size: {len(train_dataset)} | Validation Dataset size: {len(val_dataset)}")

    # ┌───────────────────────────────────────────────┐
    # │     PERFORM INSPECT ON TRAIN SOFT PROMPTS     │
    # └───────────────────────────────────────────────┘
    # List to hold the summary of best metrics across all inspect datasets
    summary_results = []

    for data in train_dataset[:NUM_TRAINING_EXAMPLES]:
        dataset_id = data["dataset_id"]
        soft_prompt = data["soft_prompt"] # shape (1, soft_prompt_len, embed_dim)
        hard_prompt = data["hard_prompt"]

        print("-" * 100)
        print(f"Performing InSPEcT using soft prompts trained on {dataset_id}")
        print("-" * 100 + "\n")

        # Get Elicited Text using InSPEcT Technique
        inspect_elicited_results = elicit_description_using_inspect_technique(
            model=inspect_model,
            tokenizer=tokenizer,
            num_tokens=NUM_TOKENS,
            soft_prompt=soft_prompt,
            dataset_name="REPLACE_ME",
            layer_combinations=BEST_PATCHES,
            # layer_combinations=ALL_LAYER_COMBINATIONS,
            target_prompt_type='few_shot'
        )

        # Evaluate InSPEcT results
        for i in range(len(inspect_elicited_results)):
            output_text = str(inspect_elicited_results[i]['output'])
            class_rate, f1_score = calculate_eval_metrics(output_text, hard_prompt)
            verb_score = (class_rate + f1_score) / 2
            inspect_elicited_results[i]['verb_score'] = verb_score

        # Find the row with the highest verb_score
        max_verb_score_row = max(inspect_elicited_results, key=lambda x: x['verb_score'])

        # Retrieve the training stats for this dataset
        training_stats_df = TRAINING_STATS_DF[TRAINING_STATS_DF["dataset_id"] == dataset_id]

        # Save Elicitations using InSPEcT for this dataset
        # elicitation_save_dir = os.path.join("inspect_results", DOD_NAME)
        # os.makedirs(elicitation_save_dir, exist_ok=True)
        # df = pd.DataFrame(inspect_elicited_results)
        # df.to_csv(f'{elicitation_save_dir}/{dataset_id}_elicitations.csv', index=False)

        result_entry = {
            "dataset": dataset_id,
            "val_accuracy": training_stats_df['val_accuracy'].iloc[0] if len(training_stats_df) > 0 else None,
            "max_verb_score_rate": round(max_verb_score_row['class_rate'], 4),
            "max_verb_src_layer": max_verb_score_row['source_layer'],
            "max_verb_tgt_layer": max_verb_score_row['target_layer'],
        }

        summary_results.append(result_entry)

    summary_df = pd.DataFrame(summary_results)


    summary_save_dir = os.path.join("inspect_results",DOD_NAME)
    os.makedirs(summary_save_dir, exist_ok=True)
    summary_csv_path = f"{summary_save_dir}/inspect_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved master summary with best metrics to: {summary_csv_path}")

