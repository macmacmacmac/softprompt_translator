import os
import argparse
import torch
import ipdb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Dict

from softprompt_experiments.scripts.dpo.scoring_utils import (
    init_scoring,
    get_scores,
    get_logprob_of_translation_given_soft_prompt,
)

# Load all Environment Variables
load_dotenv()


# ┌───────────────────────────────────────────────┐
# │                 HELPER METHODS                │
# └───────────────────────────────────────────────┘
def generate_preference_dataset(dataset: List[Dict], translator_model: PeftModel, translator_tokenizer: AutoTokenizer) -> List[Dict]:
    preference_dataset = []
    skipped_tasks = []
    for k in range(K):
        for task in tqdm(dataset, desc=f"Round {k + 1}/{K}"):
            # Extract soft prompt
            soft_prompt = task["soft_prompt"].to(DEVICE, dtype=DTYPE)

            # ── Step 1: Greedy-decode z_G (deterministic anchor) ──
            greedy_embeds = soft_prompt.unsqueeze(0)                               # (1, soft_tokens, embed_dim)
            greedy_mask = torch.ones(greedy_embeds.shape[:2], dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                greedy_ids = translator_model.generate(
                    inputs_embeds = greedy_embeds,
                    attention_mask = greedy_mask,
                    max_new_tokens = MAX_NEW_TOKENS,
                    do_sample = False,
                    pad_token_id = translator_tokenizer.eos_token_id,
                )

            z_G = translator_tokenizer.decode(greedy_ids[0], skip_special_tokens=True).strip()

            if not z_G:
                skipped_tasks.append(task["task_name"])
                tqdm.write(f'Greedy decode produced empty translation! Skipping task: {task["task_name"]}')
                continue

            # Score the greedy translation
            score_G = get_scores([z_G], task["train_instances"]).item()

            # ── Step 2: Sample translations until one beats z_G ──
            # Pool of unique translations -> score (insertion-ordered), seeded with z_G
            unique_scores: Dict[str, float] = {z_G: score_G}

            sample_embeds = soft_prompt.unsqueeze(0).expand(N, -1, -1)             # (N, soft_tokens, embed_dim)
            sample_mask = torch.ones(sample_embeds.shape[:2], dtype=torch.long, device=DEVICE)

            temperature = TEMPERATURE
            for attempt in range(MAX_RETRIES + 1):
                # Produce N sampled translations
                with torch.no_grad():
                    gen_ids = translator_model.generate(
                        inputs_embeds = sample_embeds,
                        attention_mask = sample_mask,
                        max_new_tokens = MAX_NEW_TOKENS,
                        do_sample = True,
                        temperature = temperature,
                        top_p = TOP_P,
                        pad_token_id = translator_tokenizer.eos_token_id,
                    )

                translations = translator_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                translations = [txt.strip() for txt in translations]

                # Score ALL genuinely new, non-empty texts in the full batch
                new_texts = [txt for txt in translations if txt and txt not in unique_scores]
                if new_texts:
                    new_scores = get_scores(new_texts, task["train_instances"])
                    for txt, score in zip(new_texts, new_scores.tolist()):
                        unique_scores[txt] = score

                    # Check if any new translation beat z_G
                    if new_scores.max().item() > score_G:
                        break

                if attempt < MAX_RETRIES:
                    temperature = min(temperature + RETRY_TEMP_INCREMENT, MAX_RETRY_TEMPERATURE)
                    tqdm.write(
                        f"No sampled translation beat z_G for task {task['task_name']} "
                        f"(attempt {attempt + 1}/{MAX_RETRIES + 1}), retrying with temperature={temperature}..."
                    )

            # ── Step 3: Determine z_W and z_L ──
            z_W = max(unique_scores, key=unique_scores.get)
            z_L = min(unique_scores, key=unique_scores.get)

            # Degenerate: z_W and z_L are the same text → skip
            if z_W == z_L:
                skipped_tasks.append(task["task_name"])
                tqdm.write(f'All scores identical! Skipping task: {task["task_name"]}')
                continue

            # Calculate log prob of producing z_W and z_L using the translator, conditioned on the soft prompt
            logp_ref_z_W = get_logprob_of_translation_given_soft_prompt(translator_model, translator_tokenizer, z_W, soft_prompt)
            logp_ref_z_L = get_logprob_of_translation_given_soft_prompt(translator_model, translator_tokenizer, z_L, soft_prompt)

            # Add to dataset
            preference_dataset.append({
                "task_name": task["task_name"],
                "z_prime": task["soft_prompt"],
                "z_W": z_W,
                "z_L": z_L,
                "logp_ref_z_W": logp_ref_z_W,
                "logp_ref_z_L": logp_ref_z_L,
            })

    if skipped_tasks:
        print(f"Skipped {len(skipped_tasks)} task(s) due to degenerate preference pairs: {skipped_tasks}")

    return preference_dataset



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

    # Dataset Paths
    parser.add_argument("--mapper-dataset-path", type=str, default="./shared/datasets/mapper_training_dataset/General-DoD-DPO")
    parser.add_argument("--save-dataset-path", type=str, default="./shared/datasets/dpo_preference_datasets")

    # Score Model
    parser.add_argument("--score-fn", type=str, default="LOGPROB", help="Can be either: ROUGE-L | LOGPROB")
    parser.add_argument("--score-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HF model ids (contain `/`) run locally; bare OpenAI model names use the API (ROUGE-L only)")

    # vLLM Config
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM backend for local HF score models (speeds up ROUGE-L scoring)")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.55, help="GPU memory fraction for vLLM (keep it <0.5 to leave room for the translator model)")

    # Translator Model
    parser.add_argument("--lora-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lora-weights-path", type=str, default="./shared/mapper_lora_weights/General-DoD-10x/meta-llama/Llama-3.1-8B-Instruct")

    # HyperParams
    parser.add_argument("-n", "--num-samples-to-generate", type=int, default=10)
    parser.add_argument("-k", "--scaling-factor", type=int, default=1)
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--score-batch-size", type=int, default=16, help="Micro-batch size used when scoring train instances with a local HF score model (both score functions)")
    parser.add_argument("--score-max-new-tokens", type=int, default=128, help="Cap on generated output length per train instance when scoring ROUGE-L with a local HF score model")
    parser.add_argument("--max-retries", type=int, default=5, help="Max resample attempts for a task before giving up on a non-degenerate preference pair")
    parser.add_argument("--retry-temp-increment", type=float, default=0.25, help="Amount to raise sampling temperature by on each retry")
    parser.add_argument("--max-retry-temperature", type=float, default=2.0, help="Cap on the escalated retry temperature")
    parser.add_argument("--openai-concurrency", type=int, default=32, help="Number of worker threads for concurrent OpenAI scoring calls (ROUGE-L score function only)")

    args, _ = parser.parse_known_args(args_list)

    # Define Global Variables
    global DEVICE, DTYPE, K, TOP_P, N, TEMPERATURE, MAX_NEW_TOKENS, MAX_RETRIES, RETRY_TEMP_INCREMENT, MAX_RETRY_TEMPERATURE

    # Parse all the arguments into Variables
    MAPPER_DATASET_PATH = args.mapper_dataset_path
    SCORE_FN = args.score_fn
    SCORE_MODEL_NAME = args.score_model_name
    LORA_MODEL_NAME = args.lora_model_name
    LORA_WEIGHTS_PATH = args.lora_weights_path
    N = args.num_samples_to_generate
    K = args.scaling_factor
    TEMPERATURE = args.temperature
    MAX_NEW_TOKENS = args.max_new_tokens
    TOP_P = args.top_p
    SCORE_BATCH_SIZE = args.score_batch_size
    SCORE_MAX_NEW_TOKENS = args.score_max_new_tokens
    MAX_RETRIES = args.max_retries
    RETRY_TEMP_INCREMENT = args.retry_temp_increment
    MAX_RETRY_TEMPERATURE = args.max_retry_temperature
    OPENAI_CONCURRENCY = args.openai_concurrency
    USE_VLLM = args.use_vllm
    VLLM_GPU_MEMORY_UTILIZATION = args.vllm_gpu_memory_utilization
    SAVE_DATASET_PATH = args.save_dataset_path + f"/{SCORE_FN}score_{N}n_{K}k_{TEMPERATURE}temp_{TOP_P}top_p"

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


    # ┌───────────────────────────────────────────────┐
    # │                   DATASET PREP                │
    # └───────────────────────────────────────────────┘
    print("Loading Train and Validation datasets...")
    train_dataset = torch.load(os.path.join(MAPPER_DATASET_PATH, 'train_mapper_dataset.pt'), 
                               map_location="cpu", 
                               weights_only=True)

    val_dataset = torch.load(os.path.join(MAPPER_DATASET_PATH, 'val_mapper_dataset.pt'), 
                             map_location="cpu", 
                             weights_only=True)
    
    print(f"Train Dataset size: {len(train_dataset)} | Validation Dataset size: {len(val_dataset)}")
    

    # ┌───────────────────────────────────────────────┐
    # │                 TRANSLATOR PREP               │
    # └───────────────────────────────────────────────┘
    translator_tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_NAME)
    translator_tokenizer.pad_token = translator_tokenizer.eos_token

    print(f"Loading base model {LORA_MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(LORA_MODEL_NAME, dtype=DTYPE, device_map=DEVICE)

    print(f"Loading LoRA adapters from {LORA_WEIGHTS_PATH}...")
    translator_model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
    translator_model.eval()


    # ┌───────────────────────────────────────────────┐
    # │                SCORE MODEL PREP               │
    # └───────────────────────────────────────────────┘
    init_scoring(
        score_fn=SCORE_FN,
        score_model_name=SCORE_MODEL_NAME,
        score_batch_size=SCORE_BATCH_SIZE,
        score_max_new_tokens=SCORE_MAX_NEW_TOKENS,
        openai_concurrency=OPENAI_CONCURRENCY,
        device=DEVICE,
        dtype=DTYPE,
        translator_model=translator_model,
        lora_model_name=LORA_MODEL_NAME,
        use_vllm=USE_VLLM,
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    )


    # ┌───────────────────────────────────────────────┐
    # │          PREFERENCE DATASET GENERATION        │
    # └───────────────────────────────────────────────┘
    train_preference_dataset = generate_preference_dataset(train_dataset, translator_model, translator_tokenizer)
    val_preference_dataset = generate_preference_dataset(val_dataset, translator_model, translator_tokenizer)


    # ┌───────────────────────────────────────────────┐
    # │            SAVE PREFERENCE DATASETS           │
    # └───────────────────────────────────────────────┘
    # Create the Directory for saving the datasets
    os.makedirs(SAVE_DATASET_PATH, exist_ok=True)
    
    # Save the Training and Validation Datasets
    train_dataset_path = os.path.join(SAVE_DATASET_PATH, 'train_dataset.pt')
    val_dataset_path = os.path.join(SAVE_DATASET_PATH, 'val_dataset.pt')
    
    torch.save(train_preference_dataset, train_dataset_path)
    torch.save(val_preference_dataset, val_dataset_path)
    
    print(f"Saved Train Split ({len(train_preference_dataset)} samples) to: {train_dataset_path}")
    print(f"Saved Val Split ({len(val_preference_dataset)} samples) to: {val_dataset_path}")
    
