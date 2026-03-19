import os
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.softprompt_experiments.InSPEcT_utils import elicit_description_using_inspect_technique, ALL_LAYER_COMBINATIONS
import pandas as pd
import string

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
    parser.add_argument("--val_dataset_path", type=str, default="./datasets/mapper_training_dataset/val_mapper_dataset.pt")
    parser.add_argument("--lora_dir", type=str, default="./mapper_lora_weights")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    VAL_DATASET_PATH = args.val_dataset_path
    LORA_DIR = args.lora_dir
    NUM_SAMPLES = args.num_samples
    NUM_TOKENS = args.num_tokens
    SEED = args.seed

    random.seed(SEED)

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # ┌───────────────────────────────────────────────┐
    # │                   DATASET PREP                │
    # └───────────────────────────────────────────────┘
    print(f"Loading Validation dataset from {VAL_DATASET_PATH}...")
    val_dataset = torch.load(VAL_DATASET_PATH, map_location="cpu", weights_only=True)
    
    print(f"Validation Dataset size: {len(val_dataset)}")

    # Pick a random subset to evaluate
    test_samples = random.sample(val_dataset, min(NUM_SAMPLES, len(val_dataset)))

    # ┌───────────────────────────────────────────────┐
    # │                 LORA MODEL PREP               │
    # └───────────────────────────────────────────────┘
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    inspect_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    
    print(f"Loading LoRA adapters from {LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()
    inspect_model.eval()

    # ┌───────────────────────────────────────────────┐
    # │                 INFERENCE LOOP                │
    # └───────────────────────────────────────────────┘
    print("\n" + "="*80)
    print("\t\t\tGENERATION RESULTS")
    print("="*80 + "\n")

    # List to hold the summary of best metrics across all inspect datasets
    summary_results = []

    with torch.no_grad():
        for i, sample in enumerate(test_samples):
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
            
            # Calculate Class Rate (Mapper)
            classes_count = sum(1 for c in target_set if c in pred_set)
            mapper_class_rate = classes_count / len(target_set) if len(target_set) > 0 else 0.0

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
            
            # Print out the stats
            print(f"Dataset ID: {dataset_id}")
            print(f"Target Keywords : {true_keywords}")
            print(f"Model Predicted : {pred_text}")
            print(f"Metrics         : Class Rate: {mapper_class_rate:.2f} | Recall: {recall:.2f} | Precision: {precision:.2f} | F1: {f1_score:.2f}")
            
            if f1_score == 1.0:
                print(f"Result: Perfect Match")
            elif recall == 1.0 and precision < 1.0:
                print(f"Result: Runaway Generation (Got all targets, but babbled)")
            elif recall > 0:
                print(f"Result: Partial Match")
            else:
                print(f"Result: Failed")


            print("-" * 100)
            print(f"Performing InSPEcT using soft prompts trained on Dataset ID: {dataset_id}")
            print("-" * 100 + '\n')

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

                # Class Rate Calculation
                classes_count = sum(1 for c in target_set if c in elicited_words)
                class_rate = classes_count / len(target_set) if len(target_set) > 0 else 0.0

                # Calculate Recall, Precision and F1 Score
                overlap = target_set.intersection(elicited_words)
                
                recall = len(overlap) / len(target_set) if len(target_set) > 0 else 0
                precision = len(overlap) / len(elicited_words) if len(elicited_words) > 0 else 0
                
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0.0

                # Add the scores to the row
                inspect_elicited_results[j]['class_rate'] = class_rate
                inspect_elicited_results[j]['recall'] = recall
                inspect_elicited_results[j]['precision'] = precision
                inspect_elicited_results[j]['f1_score'] = f1_score

            # Find the row with the highest class rate
            best_classrate_row = max(inspect_elicited_results, key = lambda x: x['class_rate'])

            # Save the summary comparing Mapper vs InSPEcT Best
            summary_results.append({
                "dataset": dataset_id,
                "mapper_class_rate": round(mapper_class_rate, 4),
                "mapper_elicitation": pred_text,
                "inspect_best_class_rate": round(best_classrate_row['class_rate'], 4),
                "inspect_best_class_rate_src_layer": best_classrate_row['source_layer'],
                "inspect_best_class_rate_tgt_layer": best_classrate_row['target_layer'],
                "inspect_best_class_rate_elicitation": best_classrate_row['output']
            })


            # Save Elicitations using InSPEcT for this dataset
            elicitation_save_dir = f"./inspect_results/DoD_soft_prompts"
            os.makedirs(elicitation_save_dir, exist_ok=True)

            df = pd.DataFrame(inspect_elicited_results)
            df.to_csv(f'{elicitation_save_dir}/dataset_{dataset_id}_elicitations.csv', index=False)

            print("-" * 80)

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = f"{elicitation_save_dir}/mapper_vs_inspect.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSaved master summary with best metrics to: {summary_csv_path}")