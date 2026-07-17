import os
import argparse
import torch
import ipdb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    parser.add_argument("--mapper-dataset-path", type=str, default="./shared/datasets/mapper_training_dataset/General-DoD")
    parser.add_argument("--master-verbalizations-path", type=str, default="./shared/verbalizations/master_verbalizations_v2.json")
    parser.add_argument("--new-verbalizations-path", type=str, default="./shared/verbalizations/master_verbalizations_v2_new.json")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MAPPER_DATASET_PATH = args.mapper_dataset_path
    MASTER_VERBALIZATIONS_PATH = args.master_verbalizations_path
    NEW_VERBALIZATIONS_PATH = args.new_verbalizations_path


    # Global Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Rouge Metric
    ROUGE_METRIC = evaluate.load("rouge")

    # ┌───────────────────────────────────────────────┐
    # │                   DATASET PREP                │
    # └───────────────────────────────────────────────┘
    print("Loading Train and Validation datasets ...")
    train_dataset = torch.load(os.path.join(MAPPER_DATASET_PATH, 'train_mapper_dataset.pt'), 
                               map_location="cpu", 
                               weights_only=True)
    val_dataset = torch.load(os.path.join(MAPPER_DATASET_PATH, 'val_mapper_dataset.pt'), 
                             map_location="cpu", 
                             weights_only=True)
    
    print(f"Train Dataset size: {len(train_dataset)} | Validation Dataset size: {len(val_dataset)}")


    # ┌───────────────────────────────────────────────┐
    # │                 MODEL PREP                    │
    # └───────────────────────────────────────────────┘
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    llama_word_embeddings = model.get_input_embeddings()
    
    # Init list to accumulate results
    soft_prompt_results = []

    with torch.no_grad():
        for item in tqdm(val_dataset):
            # Extract data
            soft_prompt = item["soft_prompt"].to(DEVICE, dtype=DTYPE)                                           # (soft_tokens,)
            task_name = item["task_name"]
            val_instances = item["val_instances"]

            # Prep Input / Output pairs
            # No trailing space after "Output:" -- must match train_softprompts.py's
            # prompt format exactly (a trailing space tokenizes as a standalone token
            # the soft prompt never saw during training)
            inputs = [f"Input: {instance["input"]}\nOutput:" for instance in val_instances]
            outputs = [f"{instance["output"]}" for instance in val_instances]

            # Cap generation like train_softprompts.py: longest target for this task
            # (measured with the leading space, as in training) + slack for EOS
            task_max_new_tokens = max(
                len(tokenizer.encode(f" {instance['output']}", add_special_tokens=False))
                for instance in val_instances
            ) + 10

            # Tokenize Inputs, and generate embeddings using Llama Embeddings model
            tokenized = tokenizer(inputs, add_special_tokens=True, return_tensors='pt',padding='longest', padding_side='left').to(DEVICE)
            input_ids, attn_mask = tokenized['input_ids'], tokenized['attention_mask']
            input_embeds = llama_word_embeddings(input_ids).to(DEVICE, dtype=DTYPE)                             # (batch_size, seq_len, embed_dim)

            # Duplicate the soft prompt per batch
            soft_prompt_embeds = soft_prompt.unsqueeze(0).expand(input_embeds.shape[0], -1, -1)                 # (batch_size, soft_tokens, embed_dim)
            soft_prompt_attn_mask = torch.ones(soft_prompt_embeds.shape[:2], dtype=torch.long, device=DEVICE)   # (batch_size, soft_tokens)

            # Add Soft Prompts and Input Embeddings together
            full_embeds = torch.cat([soft_prompt_embeds, input_embeds],dim=1)
            full_attn_mask = torch.cat([soft_prompt_attn_mask, attn_mask], dim=1)

            # Generate the predicted tokens for the whole batch
            pred_ids = model.generate(
                inputs_embeds=full_embeds,
                attention_mask=full_attn_mask,
                max_new_tokens=task_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode the batched outputs
            pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            pred_texts = [text.strip() for text in pred_texts]

            # Add output of soft prompt for each instance
            for instance, pred_text in zip(val_instances, pred_texts):
                instance["soft_output"] = pred_text

            # ┌───────────────────────────────────────────────┐
            # │                   EVALUATION                  │
            # └───────────────────────────────────────────────┘
            # Compute ROUGE-L for the entire batch
            rouge_results = ROUGE_METRIC.compute(
                predictions=pred_texts, 
                references=outputs
            )

            # compute() already returns a single aggregated score for the batch
            soft_task_rougeL = rouge_results['rougeL']

            # ┌───────────────────────────────────────────────┐
            # │                AGGREGATE RESULTS              │
            # └───────────────────────────────────────────────┘
            soft_prompt_results.append({
                "task_name": task_name,
                "val_instances": val_instances,
                "soft_task_rougeL": soft_task_rougeL
            })

    # ┌───────────────────────────────────────────────┐
    # │          ADD RESULTS TO MASTER FILE           │
    # └───────────────────────────────────────────────┘
    # Load master verbalizations
    with open(MASTER_VERBALIZATIONS_PATH, "r") as f:
        master_verbalizations = json.load(f)
    
    # Ensure len of master_verbalizations and soft prompt results is same
    assert len(master_verbalizations) == len(soft_prompt_results)

    # Append results
    for master_verbalization, soft_prompt_result in zip(master_verbalizations, soft_prompt_results):
        assert master_verbalization["task_name"] == soft_prompt_result["task_name"]
        master_verbalization["soft_task_rougeL"] = soft_prompt_result["soft_task_rougeL"]

        assert len(master_verbalization["val_instances"]) == len(soft_prompt_result["val_instances"])
        for m, s in zip(master_verbalization["val_instances"], soft_prompt_result["val_instances"]):
            m["soft_output"] = s["soft_output"]



    with open(NEW_VERBALIZATIONS_PATH, "w") as f:
        json.dump(master_verbalizations, f, indent = 4)


    


