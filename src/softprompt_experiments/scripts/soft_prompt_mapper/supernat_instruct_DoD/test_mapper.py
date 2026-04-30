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
    parser.add_argument("--val_dataset_path", type=str, default="./datasets/mapper_training_dataset/SUPER-NATURALINSTRUCTIONS-english-filtered_original_instructions/val_mapper_dataset.pt")
    parser.add_argument("--lora_dir", type=str, default="./mapper_lora_weights/SUPER-NATURALINSTRUCTIONS-english-filtered_original_instructions")
    parser.add_argument("--training_stats_path", type=str, default="./trained_soft_prompts/SUPER-NATURALINSTRUCTIONS-english-filtered_original_instructions/training_stats.csv")
    parser.add_argument("--sample", action='store_true', help="Use a sample of val dataset instead of the full val dataset")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling decoding")
    parser.add_argument("--output_json", type=str, default="./SupNatInstruct_verbalizations_original_instructions.json")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    VAL_DATASET_PATH = args.val_dataset_path
    LORA_DIR = args.lora_dir
    TRAINING_STATS_PATH = args.training_stats_path
    BATCH_SIZE = args.batch_size
    NUM_SAMPLES = args.num_samples
    SEED = args.seed
    DO_SAMPLE = args.do_sample
    OUTPUT_JSON = args.output_json

    # Set the Seed for this experiment
    random.seed(SEED)

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Training Stats
    TRAINING_STATS_DF = pd.read_csv(TRAINING_STATS_PATH, index_col='task_name')

    # Load Rouge Metric
    ROUGE_METRIC = evaluate.load("rouge")

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
    llama_word_embeddings = model.get_base_model().get_input_embeddings()

    print(f"Loading LoRA adapters from {LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()

    with torch.no_grad():
        SOFT_MARKER = llama_word_embeddings(tokenizer("<SOFT:>", add_special_tokens=False, return_tensors='pt').to(DEVICE)['input_ids']).detach()
        HARD_MARKER = llama_word_embeddings(tokenizer("<HARD:>", add_special_tokens=False, return_tensors='pt').to(DEVICE)['input_ids']).detach()
        INIT_MARKER = llama_word_embeddings(tokenizer("<INIT:>", add_special_tokens=False, return_tensors='pt').to(DEVICE)['input_ids']).detach()

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
            instances_list = [s.get("instances", []) for s in batch_samples]
            
            # Stack soft prompts: (batch_size, seq_len, embed_dim)
            soft_prompts = torch.stack([s["soft_prompt"] for s in batch_samples]).to(DEVICE, dtype=DTYPE)
            
            # stack init embeddings
            init = torch.stack([s["soft_prompt_init_embeddings"] for s in batch_samples]).to(DEVICE, dtype=DTYPE)

            batchsize = soft_prompts.shape[0]
            soft_marker = SOFT_MARKER.expand(batchsize, -1, -1)
            hard_marker = HARD_MARKER.expand(batchsize, -1, -1)
            init_marker = INIT_MARKER.expand(batchsize, -1, -1)

            inputs_embeds = torch.cat([soft_marker, soft_prompts, hard_marker], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
                
            # get the sequence length of the prefix so we can use it to build attn_mask and labels
            prefix_len = init_marker.shape[1] + init.shape[1] + soft_marker.shape[1] + soft_prompts.shape[1] + hard_marker.shape[1]
            
            # Concatenate Attention Masks (Add `1`s for the soft prompt so Llama Model pays attention to it)
            attention_mask = torch.ones((batchsize, prefix_len), dtype=attention_mask.dtype, device=DEVICE)   # (batch_size, soft_prompt_len)
            
            # Generate the predicted tokens for the whole batch
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
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
            
            # Process and store results for each sample in the batch
            for j in range(len(batch_samples)):
                task_name = task_names[j]
                task_rouge_l = TRAINING_STATS_DF.loc[task_name].get('val_rougeL', 'N/A')
                results_data.append({
                    "task_name": task_name,
                    "hard_prompt": hard_prompts[j],
                    "verbalization": pred_texts[j],
                    "task_rouge_l": task_rouge_l,
                    "verbalization_rouge_l": rouge_l_scores[j],
                    "instances": instances_list[j]
                })

    # Save to JSON
    print(f"\nSaving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_data, f, indent=4)
    print("Done!\n")
