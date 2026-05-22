import os
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.utils import log_json

from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import evaluate
import json

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
    parser.add_argument("--proportion_to_use", type=float, default=1.0)
    # parser.add_argument("--mapper_dataset_path", type=str, default="./datasets/mapper_training_dataset/supnat_eng_fil_aug_3/")
    parser.add_argument("--mapper_dataset_path", type=str, default="./datasets/mapper_training_dataset/supnat_eng_fil_augenriched/")
    parser.add_argument("--save_directory", type=str, default="./datasets/mapper_training_dataset/supnat_eng_fil_orig/")
    # parser.add_argument("--lora_dir", type=str, default="./mapper_lora_weights/supnat_eng_fil_orig")
    # parser.add_argument("--training_stats_path", type=str, default="./trained_soft_prompts/SUPER-NATURALINSTRUCTIONS-english-filtered_original_instructions/training_stats.csv")
    parser.add_argument("--sample", action='store_true', help="Use a sample of val dataset instead of the full val dataset")
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--prefill_text", type=str, default="First, I should")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling decoding")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    PROPORTION_FOLDER = f"{int(100*args.proportion_to_use)}_percent"
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MAPPER_DATASET_PATH = os.path.join(args.mapper_dataset_path)
    VAL_DATASET_PATH = os.path.join(MAPPER_DATASET_PATH, "val_mapper_dataset.pt")
    LORA_DIR = os.path.join(MAPPER_DATASET_PATH, "mapper_lora_weights", PROPORTION_FOLDER)
    # TRAINING_STATS_PATH = args.training_stats_path
    SEED = args.seed
    DO_SAMPLE = args.do_sample
    # OUTPUT_JSON = args.output_json

    # Set the Seed for this experiment
    torch.manual_seed(SEED)
    random.seed(SEED)

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Training Stats
    # TRAINING_STATS_DF = pd.read_csv(TRAINING_STATS_PATH, index_col='task_name')

    # ┌───────────────────────────────────────────────┐
    # │                 LORA MODEL PREP               │
    # └───────────────────────────────────────────────┘
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    word_embeddings = base_model.get_input_embeddings()

    print(f"Loading LoRA adapters from {LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()

    # ┌───────────────────────────────────────────────┐
    # │                 INFERENCE LOOP                │
    # └───────────────────────────────────────────────┘
    print("\n" + "="*80)
    print("\t\t\tGENERATION RESULTS")
    print("="*80 + "\n")

    results_data = []

    dataset_dirs = []
    for entry in os.scandir(args.save_directory):
        if entry.is_dir():  # Check if the entry is a directory
            if "dataset_" in entry.name:
                dataset_dirs.append(entry.path)

    num_datasets = len(dataset_dirs)
    if num_datasets > 0:
        print(f"\nFound ({num_datasets}) datasets in directory")
    else:
        raise ValueError("path to directory has no datasets")

    prefill_text = args.prefill_text

    for dataset_dir in tqdm(dataset_dirs):
        print(f"For softprompt in {dataset_dir}")
        hardprompt = torch.load(
            os.path.join(dataset_dir,'dataset.pt'),
            weights_only=False
        )['hardprompt']
        results = {}
        results['hardprompt'] = hardprompt
        softprompt = SoftPrompt(
            model=model, 
            tokenizer=tokenizer, 
            word_embeddings=None, 
            path_to_model=os.path.join(dataset_dir,'softprompt.pt')
        )
        with open(os.path.join(dataset_dir,'softprompt_performance.json')) as f:
            soft_perf = json.load(f)


        with torch.no_grad():
            # Just softprompt
            soft_prompts = softprompt.forward()
            soft_attn_mask = torch.ones(soft_prompts.shape[:2], dtype=torch.long, device=DEVICE)

            # Prefill
            tokenized_input = tokenizer(prefill_text, return_tensors='pt',add_special_tokens=False).to(DEVICE)
            input_embeds = word_embeddings(tokenized_input['input_ids']).to(DEVICE).to(DTYPE)
            input_attn_mask = tokenized_input['attention_mask']
            
            # Soft + prefill
            full_embeds = torch.cat([soft_prompts, input_embeds], dim=1)
            full_attn_mask = torch.cat([soft_attn_mask, input_attn_mask], dim=1)

            # Random control
            random_embeds = word_embeddings(torch.randint(
                0, word_embeddings.num_embeddings,(soft_prompts.shape[1],), dtype=torch.long
            ).to(model.device)).to(DEVICE).to(DTYPE).unsqueeze(0)

            # Random + prefill
            random_full_embeds = torch.cat([random_embeds, input_embeds], dim=1)

            # SOFTPROMPT TRIAL-----------------------------------------------------------------
            outputs = model.generate(
                inputs_embeds=soft_prompts,
                attention_mask=soft_attn_mask,
                max_new_tokens=300,
                do_sample=DO_SAMPLE,
                temperature=0.7 if DO_SAMPLE else None,
                top_p=0.9 if DO_SAMPLE else None,
                pad_token_id=tokenizer.eos_token_id
            )
            # Decode the batched outputs
            pred_texts_vanilla = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_texts_vanilla = [text.strip() for text in pred_texts_vanilla]

            print(f"Verbalization using: soft embeds:\n<v begin>{pred_texts_vanilla[0]}</v end>\n")
            
            # SOFT + PREFILL TRIAL-----------------------------------------------------------------
            outputs = model.generate(
                inputs_embeds=full_embeds,
                attention_mask=full_attn_mask,
                max_new_tokens=300,
                do_sample=DO_SAMPLE,
                temperature=0.7 if DO_SAMPLE else None,
                top_p=0.9 if DO_SAMPLE else None,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the batched outputs
            pred_texts_prefilled = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_texts_prefilled = [text.strip() for text in pred_texts_prefilled]

            print(f"Verbalization using: soft embeds + prefill(\"{prefill_text}\"):\n{prefill_text}<v begin>{pred_texts_prefilled[0]}</v end>\n")

            # # CONTROL TRIAL-----------------------------------------------------------------
            # outputs = model.generate(
            #     inputs_embeds=random_embeds,
            #     attention_mask=soft_attn_mask,
            #     max_new_tokens=300,
            #     do_sample=DO_SAMPLE,
            #     temperature=0.7 if DO_SAMPLE else None,
            #     top_p=0.9 if DO_SAMPLE else None,
            #     pad_token_id=tokenizer.eos_token_id
            # )
            # # Decode the batched outputs
            # pred_texts_vanilla = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # pred_texts_vanilla = [text.strip() for text in pred_texts_vanilla]

            # print(f"Verbalization using: random embeds:\n<v begin>{pred_texts_vanilla[0]}</v end>\n")

            # # RANDOM + PREFILL TRIAL-----------------------------------------------------------------
            # outputs = model.generate(
            #     inputs_embeds=random_full_embeds,
            #     attention_mask=full_attn_mask,
            #     max_new_tokens=300,
            #     do_sample=DO_SAMPLE,
            #     temperature=0.7 if DO_SAMPLE else None,
            #     top_p=0.9 if DO_SAMPLE else None,
            #     pad_token_id=tokenizer.eos_token_id
            # )
            
            # # Decode the batched outputs
            # pred_texts_prefill = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # pred_texts_prefill = [text.strip() for text in pred_texts_prefill]

            # print(f"Verbalization using: random embeds + prefill(\"{prefill_text}\"):\n{prefill_text}<v begin>{pred_texts[0]}</v end>\n")

            results['mapper_verbalization_vanilla'] = pred_texts_vanilla[0]
            results['mapper_verbalization_prefilled'] = f"{prefill_text} {pred_texts_prefilled[0]}"


            log_json(os.path.join(dataset_dir, "mapper_preds_raw.json"), results)

            for key in results:
                soft_perf[key] = results[key]
            log_json(os.path.join(dataset_dir, "softprompt_performance.json"), soft_perf)

        print("Done!\n")



                