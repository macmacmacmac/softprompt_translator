import os
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import evaluate
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

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
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--mapper_dataset_path", type=str, default="./datasets/mapper_training_dataset/General-DoD")
    parser.add_argument("--sample", action='store_true', help="Use a sample of val dataset instead of the full val dataset")
    parser.add_argument("--lora_dir", type=str, default="./mapper_lora_weights/General-DoD")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling decoding")
    parser.add_argument("--embed_model_name", type=str, default="all-MiniLM-L6-v2")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = args.model_name
    VAL_DATASET_PATH = os.path.join(args.mapper_dataset_path, "val_mapper_dataset.pt")
    LORA_DIR = os.path.join(args.lora_dir, args.model_name)
    BATCH_SIZE = args.batch_size
    NUM_SAMPLES = args.num_samples
    SEED = args.seed
    DO_SAMPLE = args.do_sample
    EMBED_MODEL_NAME = args.embed_model_name
    OUTPUT_JSON_PATH = os.path.join(LORA_DIR, "verbalizations.json")

    # Set the Seed for this experiment
    random.seed(SEED)

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Rouge Metric
    ROUGE_METRIC = evaluate.load("rouge")

    # Load Embedding Model
    print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

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

    results_data = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_samples), BATCH_SIZE), desc="Mapping Soft Prompt Batches"):
            batch_samples = test_samples[i : i + BATCH_SIZE]
            
            # Extract data for the batch
            task_names = [s["task_name"] for s in batch_samples]
            hard_prompts = [s["hard_prompt"] for s in batch_samples]
            train_instances_list = [s.get("train_instances", []) for s in batch_samples]
            val_instances_list = [s.get("val_instances", []) for s in batch_samples]
            
            # Stack soft prompts: (batch_size, seq_len, embed_dim)
            soft_prompts = torch.stack([s["soft_prompt"] for s in batch_samples]).to(DEVICE, dtype=DTYPE)
            
            # Create an attention mask for the batch: (batch_size, seq_len)
            attention_mask = torch.ones(soft_prompts.shape[:2], dtype=torch.long, device=DEVICE)
            
            # Generate the predicted tokens for the whole batch
            outputs = model.generate(
                inputs_embeds=soft_prompts,
                attention_mask=attention_mask,
                max_new_tokens=300,
                do_sample=DO_SAMPLE,
                temperature=0.7 if DO_SAMPLE else None,
                top_p=0.9 if DO_SAMPLE else None,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the batched outputs
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_texts = [text.strip() for text in pred_texts]

            # ┌───────────────────────────────────────────────┐
            # │                   EVALUATION                  │
            # └───────────────────────────────────────────────┘
            # Compute ROUGE-L for the entire batch
            rouge_results = ROUGE_METRIC.compute(
                predictions=pred_texts, 
                references=hard_prompts, 
                use_stemmer=True,
                use_aggregator=False
            )
            rouge_l_scores = rouge_results['rougeL']

            # Compute per-sample cosine similarity between generated verbalizations and hard prompts
            pred_embeddings = embed_model.encode(pred_texts, convert_to_tensor=True)
            hard_embeddings = embed_model.encode(hard_prompts, convert_to_tensor=True)
            cosine_scores = cos_sim(pred_embeddings, hard_embeddings).diagonal().cpu().tolist()

            # Process and store results for each sample in the batch
            for j in range(len(batch_samples)):
                results_data.append({
                    "task_name": task_names[j],
                    "hard_prompt": hard_prompts[j],
                    "mapper_hard_prompt": pred_texts[j],
                    "mapper_hard_prompt_rougeL": rouge_l_scores[j],
                    "mapper_hard_prompt_cos_sim": cosine_scores[j],
                    "train_instances": train_instances_list[j],
                    "val_instances": val_instances_list[j]
                })

    # Save to JSON
    print(f"\nSaving results to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results_data, f, indent=4)
    print("Done!\n")
