import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.softprompt_experiments.InSPEcT_utils import elicit_description_using_inspect_technique, ALL_LAYER_COMBINATIONS, BEST_PATCHES
import pandas as pd
import string
from tqdm import tqdm
import collections
from rouge_score import rouge_scorer

ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def calculate_eval_metrics(soft_prompt_verbalization, hard_prompt):
    """Calculates ROUGE-L and Token Level F1"""

    # Calculate ROUGE-L
    rouge_scores = ROUGE_SCORER.score(target=hard_prompt, prediction=soft_prompt_verbalization)
    rouge_l = rouge_scores['rougeL']
    
    # Calculate Token-level F1
    def normalize_text(text):
        """Helper to lowercase and remove punctuation for tokenization"""
        clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        return clean_text.split()
        
    pred_tokens = normalize_text(soft_prompt_verbalization)
    gold_tokens = normalize_text(hard_prompt)
    
    # Count the intersection of tokens
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    
    # Handle edge cases
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        token_f1 = 1.0 if pred_tokens == gold_tokens else 0.0
    elif num_same == 0:
        token_f1 = 0.0
    else:
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        token_f1 = (2 * precision * recall) / (precision + recall)
            
    return rouge_l, token_f1


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
    # │           SOFT PROMPT DATASET PREP            │
    # └───────────────────────────────────────────────┘
    train_dataset = torch.load(os.path.join(SOFT_PROMPTS_DATASET_PATH, 'train_mapper_dataset.pt'), map_location="cpu", weights_only=True)
    val_dataset = torch.load(os.path.join(SOFT_PROMPTS_DATASET_PATH, 'val_mapper_dataset.pt'), map_location="cpu", weights_only=True)

    print(f"Train Dataset size: {len(train_dataset)} | Validation Dataset size: {len(val_dataset)}")

    # ┌───────────────────────────────────────────────┐
    # │     PERFORM INSPECT ON TRAIN SOFT PROMPTS     │
    # └───────────────────────────────────────────────┘
    # List to hold the summary of best metrics across all datasets
    summary_results = []

    # Meta List to store the verb scores for all training examples for all src layers and all target layers
    verb_score_meta_list = [[[0 for _ in range(32)] for _ in range(32)] for _ in range(NUM_TRAINING_EXAMPLES)]

    for example_idx, data in enumerate(tqdm(train_dataset[:NUM_TRAINING_EXAMPLES], desc="Performing InSPEcT on Train Soft Prompts")):
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
            layer_combinations=BEST_PATCHES,
            # layer_combinations=ALL_LAYER_COMBINATIONS, # TODO: Uncomment this
            target_prompt_type='few_shot_supernat'
        )

        # Evaluate InSPEcT results
        for i in range(len(inspect_elicited_results)):
            output_text = str(inspect_elicited_results[i]['output'])
            src_layer_idx = inspect_elicited_results[i]['source_layer'] + 1
            tgt_layer_idx = inspect_elicited_results[i]['target_layer'] + 1

            # Get all scores for the output text by InSPEcT
            class_rate, f1_score = calculate_eval_metrics(output_text, hard_prompt)
            verb_score = (class_rate + f1_score) / 2
            inspect_elicited_results[i]['class_rate'] = class_rate
            inspect_elicited_results[i]['f1_score'] = f1_score
            inspect_elicited_results[i]['verb_score'] = verb_score

            # Update the meta list with the verb score value
            verb_score_meta_list[example_idx][src_layer_idx][tgt_layer_idx] = verb_score

        # Find the row with the highest verb score
        max_verb_score_row = max(inspect_elicited_results, key=lambda x: x['verb_score'])

        # Retrieve the training stats for this dataset
        training_stats_df = TRAINING_STATS_DF[TRAINING_STATS_DF["dataset_id"] == dataset_id]

        # Save Elicitations using for this dataset
        os.makedirs(f"{RESULTS_SAVE_DIR}/train", exist_ok=True)
        df = pd.DataFrame(inspect_elicited_results)
        df.to_csv(f'{RESULTS_SAVE_DIR}/train/{dataset_id}_elicitations.csv', index=False)

        result_entry = {
            "dataset": dataset_id,
            "val_accuracy": training_stats_df['val_accuracy'].iloc[0] if len(training_stats_df) > 0 else None,
            "max_verb_score": round(max_verb_score_row['verb_score'], 4),
            "max_verb_score_src_layer": max_verb_score_row['source_layer'],
            "max_verb_score_tgt_layer": max_verb_score_row['target_layer'],
        }

        summary_results.append(result_entry)

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = f"{RESULTS_SAVE_DIR}/inspect_train_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSaved master summary with best metrics to: {summary_csv_path}")

    # ┌───────────────────────────────────────────────┐
    # │          CALCULATE THE BEST LAYER PAIR        │
    # └───────────────────────────────────────────────┘
    verb_score_tensor = torch.tensor(verb_score_meta_list)

    # Average the scores across all examples (dim 0)
    mean_scores = torch.mean(verb_score_tensor.float(), dim=0)

    # print("Mean scores for layers 13-17 (src x tgt):")
    # print(mean_scores[13:18, 13:18])

    # Find the indices of the maximum value across both dimensions
    best_src_layer, best_tgt_layer = torch.where(mean_scores == mean_scores.max())

    # Get the first occurrence if there are multiple maximums
    best_src_layer = best_src_layer[0].item() - 1
    best_tgt_layer = best_tgt_layer[0].item() - 1
    
    print(f"Best Layer Pair from Training Subset: Source Layer {best_src_layer}, Target Layer {best_tgt_layer}")

    BEST_LAYER_PAIR = [
        {"min_source": best_src_layer, "max_source": best_src_layer, "min_target": best_tgt_layer, "max_target": best_tgt_layer}
    ]

    # ┌───────────────────────────────────────────────┐
    # │      APPLY BEST LAYER PAIR TO TEST PROMPTS    │
    # └───────────────────────────────────────────────┘
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
            verb_score = (class_rate + f1_score) / 2
            inspect_elicited_results[i]['class_rate'] = class_rate
            inspect_elicited_results[i]['f1_score'] = f1_score
            inspect_elicited_results[i]['verb_score'] = verb_score

        # Save Elicitations using for this dataset
        os.makedirs(f"{RESULTS_SAVE_DIR}/test", exist_ok=True)
        df = pd.DataFrame(inspect_elicited_results)
        df.to_csv(f'{RESULTS_SAVE_DIR}/test/{dataset_id}_elicitations.csv', index=False)






