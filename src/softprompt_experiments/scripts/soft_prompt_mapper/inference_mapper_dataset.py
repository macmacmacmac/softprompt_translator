import os
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.softprompt_experiments.InSPEcT_utils import elicit_description_using_inspect_technique, ALL_LAYER_COMBINATIONS
import pandas as pd
import string
from scipy.stats import pearsonr
from tqdm import tqdm

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
    parser.add_argument("--sample", action='store_true', help="Use a sample of val dataset instead of the full val dataset")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--inspect", action="store_true", help="Run InSPEcT technique for comparison")
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    VAL_DATASET_PATH = args.val_dataset_path
    LORA_DIR = args.lora_dir
    TRAINING_STATS_PATH = args.training_stats_path
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

    # Load InSPEcT Model only when InSPEcT technique is requested
    inspect_model = None
    if args.inspect:
        print(f"Loading inspect model {MODEL_NAME}...")
        inspect_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
        inspect_model.eval()

    
    print(f"Loading LoRA adapters from {LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()

    # ┌───────────────────────────────────────────────┐
    # │                 INFERENCE LOOP                │
    # └───────────────────────────────────────────────┘
    print("\n" + "="*80)
    print("\t\t\tGENERATION RESULTS")
    print("="*80 + "\n")

    # List to hold the summary of best metrics across all inspect datasets
    summary_results = []
    validation_accuracies = []
    f1_scores = []
    recalls = []
    precisions = []

    with torch.no_grad():
        for sample in tqdm(test_samples, desc = "Mapping Soft Prompts"):
            # Extract the data
            dataset_id = sample["dataset_id"]
            soft_prompt = sample["soft_prompt"]    # (seq_len, embed_dim)
            true_keywords = sample["hard_prompt"]  
            
            # Format for the model: Add batch dimension
            inputs_embeds = soft_prompt.unsqueeze(0).to(DEVICE, dtype=DTYPE)    # (1, seq_len, embed_dim)
            
            # Create an attention mask of 1s for the seq_len tokens
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=DEVICE) # (1, seq_len)
            
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

            # ┌───────────────────────────────────────────────┐
            # │                   EVALUATION                  │
            # └───────────────────────────────────────────────┘
            # Clean and split target keywords, stripping punctuation from the ground truth
            raw_target_words = [w.strip().lower() for w in true_keywords.split(",") if w.strip()]
            target_set = set([w.translate(str.maketrans('', '', string.punctuation)) for w in raw_target_words])
            clean_pred = pred_text.translate(str.maketrans('', '', string.punctuation)).lower()
            pred_set = set(clean_pred.split())

            # Calculate Overlap
            overlap = target_set.intersection(pred_set)
            
            # Calculate Recall, Precision, and F1
            mapper_recall = len(overlap) / len(target_set) if len(target_set) > 0 else 0
            precision = len(overlap) / len(pred_set) if len(pred_set) > 0 else 0
            
            # Calculate F1 score (if applicable)
            if precision + mapper_recall > 0:
                f1_score = 2 * (precision * mapper_recall) / (precision + mapper_recall)
            else:
                f1_score = 0.0

            # Accumulate validation accuracy and Mapper F1 Score for Pearson Correlation Coefficient calculation
            val_accuracy = TRAINING_STATS_DF.at[dataset_id, 'val_accuracy']
            validation_accuracies.append(val_accuracy)
            f1_scores.append(f1_score)
            precisions.append(precision)
            recalls.append(mapper_recall)

            
            # Print out the stats
            tqdm.write(f"Dataset ID: {dataset_id}")
            tqdm.write(f"Soft Prompt Validation Accuracy: {val_accuracy}")
            tqdm.write(f"Target Keywords : {true_keywords}")
            tqdm.write(f"Model Predicted : {pred_text}")
            tqdm.write(f"Metrics         : Recall (Class Rate): {mapper_recall:.2f} | Precision: {precision:.2f} | F1: {f1_score:.2f}")


            # Save the summary comparing Mapper vs InSPEcT Best
            result_entry = {
                "dataset": dataset_id,
                "mapper_class_rate": round(mapper_recall, 4),
                "mapper_elicitation": pred_text,
            }


            elicitation_save_dir = f"./inspect_results/{DATASET_NAME}/DoD_soft_prompts"

            if args.inspect:

                tqdm.write("-" * 100)
                tqdm.write(f"Performing InSPEcT using soft prompts trained on Dataset ID: {dataset_id}")
                tqdm.write("-" * 100 + '\n')

                # Get Elicited Text using InSPEcT Technique
                inspect_elicited_results = elicit_description_using_inspect_technique(
                    model=inspect_model,
                    tokenizer=tokenizer,
                    num_tokens=NUM_TOKENS,
                    soft_prompt=soft_prompt,
                    dataset_name="REPLACE_ME",
                    layer_combinations=ALL_LAYER_COMBINATIONS,
                    target_prompt_type='few_shot'
                )

                # Calculate Recall, Precision, F1 Score, and Class Rate
                for j in range(len(inspect_elicited_results)):
                    output = str(inspect_elicited_results[j]['output'])
                    
                    # INSPECT'S EXACT PRE-PROCESSING
                    # Remove punctuation and lowercase the text
                    clean_output = output.translate(str.maketrans('', '', string.punctuation)).lower()
                    elicited_words = set(clean_output.split())

                    # Calculate Recall, Precision and F1 Score
                    overlap = target_set.intersection(elicited_words)
                    
                    recall = len(overlap) / len(target_set) if len(target_set) > 0 else 0
                    precision = len(overlap) / len(elicited_words) if len(elicited_words) > 0 else 0
                    
                    if precision + recall > 0:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1_score = 0.0

                    # Add the scores to the row
                    inspect_elicited_results[j]['recall'] = recall
                    inspect_elicited_results[j]['precision'] = precision
                    inspect_elicited_results[j]['f1_score'] = f1_score

                # Find the row with the highest class rate
                best_classrate_row = max(inspect_elicited_results, key = lambda x: x['recall'])

                # Save the summary comparing Mapper vs InSPEcT Best
                result_entry.update({
                    "inspect_best_class_rate": round(best_classrate_row['recall'], 4),
                    "inspect_best_class_rate_src_layer": best_classrate_row['source_layer'],
                    "inspect_best_class_rate_tgt_layer": best_classrate_row['target_layer'],
                    "inspect_best_class_rate_elicitation": best_classrate_row['output']
                })


                # Save Elicitations using InSPEcT for this dataset
                os.makedirs(elicitation_save_dir, exist_ok=True)

                df = pd.DataFrame(inspect_elicited_results)
                df.to_csv(f'{elicitation_save_dir}/dataset_{dataset_id}_elicitations.csv', index=False)

            tqdm.write("-" * 80)

    # Calculate Avg F1 Score
    avg_f1_score = sum(f1_scores)/len(f1_scores)

    # Calculate Pearson Correlation between F1 score and soft prompt validation accuracy
    f1_pcc, f1_p_value = pearsonr(validation_accuracies, f1_scores)
    recall_pcc, recall_p_value = pearsonr(validation_accuracies, recalls)
    precision_pcc, precision_p_value = pearsonr(validation_accuracies, precisions)

    print(f"Avg Mapper F1-Score: {avg_f1_score: 2f}")
    print(f"Mapper F1-Score vs Soft Prompt Val Accuracy PCC: {f1_pcc: 2f} | p-value: {f1_p_value: 2f}")
    print(f"Mapper Recall vs Soft Prompt Val Accuracy PCC: {recall_pcc: 2f} | p-value: {recall_p_value: 2f}")
    print(f"Mapper Precision vs Soft Prompt Val Accuracy PCC: {precision_pcc: 2f} | p-value: {precision_p_value: 2f}")


    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = f"{elicitation_save_dir}/mapper_vs_inspect.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSaved master summary with best metrics to: {summary_csv_path}")