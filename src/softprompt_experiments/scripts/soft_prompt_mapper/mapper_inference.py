import os
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./datasets/mapper_training_dataset/compiled_mapper_dataset.pt")
    parser.add_argument("--lora_dir", type=str, default="./mapper_lora_weights")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    DATASET_PATH = args.dataset_path
    LORA_DIR = args.lora_dir
    NUM_SAMPLES = args.num_samples
    SEED = args.seed

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # ┌───────────────────────────────────────────────┐
    # │                   DATASET PREP                │
    # └───────────────────────────────────────────────┘
    print(f"Loading compiled dataset from {DATASET_PATH}...")
    full_dataset = torch.load(DATASET_PATH, map_location="cpu", weights_only=True)

    # Shuffle using the exact same seed as training so we get the exact same Val set
    random.seed(SEED)
    random.shuffle(full_dataset)
    
    split_idx = int(len(full_dataset) * 0.9)
    val_dataset = full_dataset[split_idx:]
    
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
    
    print(f"Loading LoRA adapters from {LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()

    # ┌───────────────────────────────────────────────┐
    # │                 INFERENCE LOOP                │
    # └───────────────────────────────────────────────┘
    print("\n" + "="*80)
    print("\t\t\tGENERATION RESULTS")
    print("="*80 + "\n")

    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            # Extract the data
            dataset_id = sample["dataset_id"]
            soft_prompt = sample["soft_prompt"]    # (seq_len, embed_dim)
            true_keywords = sample["hard_prompt"]  
            
            # Format for the model: Add batch dimension
            inputs_embeds = soft_prompt.unsqueeze(0).to(DEVICE, dtype=DTYPE)    # (1, seq_len, embed_dim)
            
            # Create an attention mask of 1s for the seq_len tokens
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=DEVICE)
            
            # Generate the discrete text
            # We use greedy decoding (temperature=0.0) because we want the exact learned mapping, not creative text.
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated token IDs back into an English string
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            print(f"Dataset ID: {dataset_id}")
            print(f"Target Keywords : {true_keywords}")
            print(f"Model Predicted : {pred_text}")
            
            # Quick visual check if it was an exact match
            if pred_text.lower() == true_keywords.lower():
                print("Result: EXACT MATCH")
            else:
                print("Result: MISMATCH")
            print("-" * 80)