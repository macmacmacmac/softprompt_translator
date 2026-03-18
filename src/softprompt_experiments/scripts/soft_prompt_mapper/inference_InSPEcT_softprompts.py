import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.softprompt_experiments.InSPEcT_utils import elicit_description_using_inspect_technique, BEST_PATCHES
import pandas as pd

def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect_soft_prompts_dir", type=str, default="./inspect_soft_prompts")
    parser.add_argument("--lora_dir", type=str, default="./mapper_lora_weights")
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    INSPECT_SOFT_PROMPTS_DIR = args.inspect_soft_prompts_dir
    LORA_DIR = args.lora_dir
    NUM_TOKENS = args.num_tokens

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


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
    # │ PERFORM INFERENCE USING INSPECT SOFT PROMPTS  │
    # └───────────────────────────────────────────────┘
    for root, dirs, files in os.walk(INSPECT_SOFT_PROMPTS_DIR):
        if len(dirs) > 0:
            for soft_prompt_dir in dirs:
                # Extract the Dataset Name
                dataset_name = ' '.join(soft_prompt_dir.split('_')[:-1])

                print("-" * 100)
                print(f"Performing Inference using soft prompts trained on {dataset_name}")
                print("-" * 100)

                # Load the Soft Prompts
                soft_prompt = torch.load(os.path.join(root, soft_prompt_dir, 'softprompt.pt'))  # (soft_prompt_len, embed_dim)

                # Add batch dimension to the soft prompt
                inputs_embeds = soft_prompt.unsqueeze(0).to(DEVICE, dtype = DTYPE)               # (1, soft_prompt_len, embed_dim)

                # Create an attention mask of 1s for the soft_prompt_len tokens
                attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=DEVICE) # (1, soft_prompt_len)

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

                # TODO: Get Elicited Text using InSPEcT Technique
                inspect_elicited_results = elicit_description_using_inspect_technique(
                    model_name=MODEL_NAME,
                    num_tokens=NUM_TOKENS,
                    soft_prompt=soft_prompt,
                    dataset_name="REPLACE_ME",
                    layer_combinations=BEST_PATCHES,
                    target_prompt_type='few_shot'
                )

                elicitation_save_dir = f"./inspect_results/{dataset_name}"
                os.makedirs(elicitation_save_dir, exist_ok=True)

                df = pd.DataFrame(inspect_elicited_results)
                df.to_csv(f'{elicitation_save_dir}/elicitations.csv', index=False)

                # TODO: Add Evaluation over here in terms of ROUGE1, Class Rate etc.

                # Print out the Stats
                print(f"Model Predictions: {pred_text}\n\n")

        else: 
            break
