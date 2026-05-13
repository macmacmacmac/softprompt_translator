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
    parser.add_argument("--proportion_to_use", type=float, default=1.0)
    parser.add_argument("--mapper_dataset_path", type=str, default="./datasets/mapper_training_dataset/supnat_eng_fil_orig_1/")
    # parser.add_argument("--lora_dir", type=str, default="./mapper_lora_weights/supnat_eng_fil_orig")
    # parser.add_argument("--training_stats_path", type=str, default="./trained_soft_prompts/SUPER-NATURALINSTRUCTIONS-english-filtered_original_instructions/training_stats.csv")
    parser.add_argument("--sample", action='store_true', help="Use a sample of val dataset instead of the full val dataset")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling decoding")
    parser.add_argument("--input", type=str, default="./LLM_verbalization.json")
    # parser.add_argument("--output", type=str, default="./LLM_verbalization_w_rouge_l")

    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MAPPER_DATASET_PATH = args.mapper_dataset_path
    VAL_DATASET_PATH = os.path.join(MAPPER_DATASET_PATH, "val_mapper_dataset.pt")
    # TRAINING_STATS_PATH = args.training_stats_path
    BATCH_SIZE = args.batch_size
    NUM_SAMPLES = args.num_samples
    SEED = args.seed
    DO_SAMPLE = args.do_sample
    # OUTPUT_JSON = args.output_json

    # Set the Seed for this experiment
    random.seed(SEED)

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Training Stats
    # TRAINING_STATS_DF = pd.read_csv(TRAINING_STATS_PATH, index_col='task_name')

    # Load Rouge Metric
    ROUGE_METRIC = evaluate.load("rouge")
    BLEU_METRIC = evaluate.load("sacrebleu")
    
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
    # │                 MODEL PREP               │
    # └───────────────────────────────────────────────┘
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    llama_word_embeddings = model.get_input_embeddings()

    # ┌───────────────────────────────────────────────┐
    # │                 INFERENCE LOOP                │
    # └───────────────────────────────────────────────┘
    print("\n" + "="*80)
    print("\t\t\tGENERATION RESULTS")
    print("="*80 + "\n")

    print(f"Loading data from {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)
    out_df = pd.DataFrame(data)
    out_df['softprompt_rougel'] = -100

    with torch.no_grad():
        for i in tqdm(range(0, len(test_samples)), desc="Mapping Soft Prompt Batches"):
            test_sample = test_samples[i]
            
            # Extract data for the batch
            soft_prompt = test_sample["soft_prompt"].to(DEVICE, dtype=DTYPE)
            task_name = test_sample["task_name"]
            instances_list = test_sample["instances"]

            inputs = [f"Input: {instance["input"]}\nOutput: " for instance in instances_list]
            outputs = [f"{instance["output"]}" for instance in instances_list]
            tokenized = tokenizer(inputs, add_special_tokens=True, return_tensors='pt',padding='longest').to(DEVICE)
            input_ids, attn_mask = tokenized['input_ids'], tokenized['attention_mask']
            input_embeds = llama_word_embeddings(input_ids).to(DEVICE, dtype=DTYPE)

            batchsize = input_embeds.shape[0]
            # Stack soft prompts: (batch_size, seq_len, embed_dim)
            soft_prompts = soft_prompt.unsqueeze(0).expand(batchsize, -1, -1)  
            full_embeds = torch.cat([soft_prompts, input_embeds],dim=1)

            # Create an attention mask for the batch: (batch_size, seq_len)
            soft_attn_mask = torch.ones(soft_prompts.shape[:2], dtype=torch.long, device=DEVICE)
            full_attn_mask = torch.cat([soft_attn_mask, attn_mask], dim=1)

            # Generate the predicted tokens for the whole batch
            pred_ids = model.generate(
                inputs_embeds=full_embeds,
                attention_mask=full_attn_mask,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode the batched outputs
            pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            pred_texts = [text.strip() for text in pred_texts]

            # ┌───────────────────────────────────────────────┐
            # │                   EVALUATION                  │
            # └───────────────────────────────────────────────┘
            # Compute ROUGE-L for the entire batch
            rouge_results = ROUGE_METRIC.compute(
                predictions=pred_texts, 
                references=outputs, 
                use_stemmer=True,
                use_aggregator=False
            )
            bleu_results = BLEU_METRIC.compute(
                predictions=pred_texts,
                references=[[x] for x in outputs]
            )

            for instance, pred_soft in zip(instances_list,pred_texts):
                instance['pred_soft'] = pred_soft

            mean_rouge_l = torch.mean(torch.tensor(rouge_results['rougeL']))
            out_df.loc[out_df['task_name'] == task_name, 'softprompt_rougel'] = mean_rouge_l.item()
            out_df.loc[out_df['task_name'] == task_name, 'instances_list'] = instances_list
            # task_rouge_l = TRAINING_STATS_DF.loc[task_name].get('val_rougeL', 'N/A')
            

    out_df.to_csv(args.output+".csv") #"./LLM_verbalizatoin_w_rouge_l.csv"!\n")
