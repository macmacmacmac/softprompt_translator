import os
import argparse
import contextlib
import torch
import torch.nn.functional as F
import ipdb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
import time
from tqdm import tqdm
from typing import List, Dict
import evaluate

# Load all Environment Variables
load_dotenv()

# ROUGE-L Scorer
ROUGE_METRIC = None

# ┌───────────────────────────────────────────────┐
# │                PROMPT TEMPLATES               │
# └───────────────────────────────────────────────┘
SYS_PROMPT_TEMPLATE = """# Task
{task_prompt}
"""

USR_PROMPT_TEMPLATE = """# Input
{input}

# Output
"""

FULL_PROMPT_TEMPLATE = SYS_PROMPT_TEMPLATE + USR_PROMPT_TEMPLATE


# ┌───────────────────────────────────────────────┐
# │                 HELPER METHODS                │
# └───────────────────────────────────────────────┘
def prompt_openai_model(
    model_name: str,
    system_prompt: str,
    user_prompt: str, 
    max_retries: int = 5,
    **kwargs
):
    defaults = {
        "model": model_name,
    }
    params = {**defaults, **kwargs}
    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                **params,
            )

            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            print(
                f"Rate limited. Retrying in {wait_time}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_time)
    return ""


def get_scores(
        translations: List[str],
        train_instances: List[Dict]
    ) -> torch.Tensor:
    if SCORE_FN == "ROUGE-L":
        return get_rougeL_scores(
            translations,
            train_instances
        )
    elif SCORE_FN == "LOGPROB":
        return get_logprob_scores(
            translations,
            train_instances
        )
    raise ValueError(f"Unsupported score function: {SCORE_FN}")


def get_rougeL_scores(
        translations: List[str],
        train_instances: List[Dict]
    ) -> torch.Tensor:
    global ROUGE_METRIC
    if ROUGE_METRIC is None:
        ROUGE_METRIC = evaluate.load("rouge")

    rougeL_scores = []
    y = [t["output"] for t in train_instances]
    for translation in translations:
        # Prep system prompt based on hard prompt (translation)
        system_prompt = SYS_PROMPT_TEMPLATE.format(task_prompt = translation)

        # Generate outputs (y_hat) using translated soft prompt
        y_hat = []
        for instance in train_instances:
            user_prompt = USR_PROMPT_TEMPLATE.format(input = instance["input"])
            y_hat.append(prompt_openai_model(SCORE_MODEL_NAME, system_prompt, user_prompt))
        
        # Compute ROUGE-L between y and y_hat for current translated soft prompt
        rougeL_scores.append(ROUGE_METRIC.compute(predictions=y_hat, references=y)["rougeL"])
    return torch.tensor(rougeL_scores)


def get_logprob_scores(
        translations: List[str],
        train_instances: List[Dict]
    ) -> torch.Tensor:
    logprob_scores = []
    y = [t["output"] for t in train_instances]
    for translation in translations:
        # Create Prompts for all instances within a task
        prompts = [FULL_PROMPT_TEMPLATE.format(task_prompt = translation, input = t["input"]) for t in train_instances]

        # Log prob of each true output given its (translation + input) prompt, using the score model
        # (batched into a single forward pass across all instances in the task)
        instance_logprobs = get_logprob_of_output_given_prompt(score_model, score_tokenizer, prompts, y)

        # Average over instances so translations are compared on the same per-instance scale
        logprob_scores.append(instance_logprobs.mean().item())

    return torch.tensor(logprob_scores)


def get_logprob_of_translation_given_soft_prompt(
        model: torch.nn.Module,
        tokenizer,
        translation: str,
        soft_prompt_embeds: torch.Tensor
    ) -> float:
    # Tokenize the translation (append EOS so the model learns/expects an explicit stop token)
    inputs = tokenizer(translation + tokenizer.eos_token, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Embed token ids via the model's word embedding layer (works through PeftModel too)
    translation_embeds = model.get_input_embeddings()(input_ids).to(DTYPE)

    # Prepare labels
    labels = input_ids.clone()

    # Move soft prompt embeds to DEVICE and DTYPE and add batch_dim if necessary
    soft_prompt_embeds = soft_prompt_embeds.to(device=DEVICE, dtype=DTYPE)
    if soft_prompt_embeds.dim() == 2:
        soft_prompt_embeds = soft_prompt_embeds.unsqueeze(0)

    # Get num of soft tokens
    soft_prompt_len = soft_prompt_embeds.shape[1]

    # Prepend the soft prompt embeddings to the sequence embeddings
    inputs_embeds = torch.cat([soft_prompt_embeds, translation_embeds], dim=1)

    # Pad attention mask with 1s for the soft prompt span
    soft_prompt_mask = torch.ones((1, soft_prompt_len), dtype=attention_mask.dtype, device=DEVICE)
    full_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)

    # Pad labels with -100 for the soft prompt span so no loss/log-prob is attributed to it
    soft_prompt_labels = torch.full((1, soft_prompt_len), -100, dtype=labels.dtype, device=DEVICE)
    full_labels = torch.cat([soft_prompt_labels, labels], dim=1)

    # Perform forward pass (no `labels=` kwarg — we need raw logits to hand-compute per-token log-probs)
    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask)
        logits = outputs.logits                                 # (batch_size, seq_len, vocab_size)

    # Sum log-probs over the valid (non -100) sequence positions -> log P(translation | soft_prompt)
    total_log_prob = _sum_sequence_logprob(logits, full_labels)
    return total_log_prob.item()


def get_logprob_of_output_given_prompt(
        model: torch.nn.Module,
        tokenizer,
        prompts: List[str],
        outputs: List[str]
    ) -> torch.Tensor:
    # Tokenize prompt and output separately and concatenate ids -- this makes the
    # prompt/output boundary exact by construction (no cross-junction BPE merging
    # like we'd risk by tokenizing the concatenated string, and no second
    # measurement pass needed to find where the prompt span ends)
    batch_input_ids = []
    batch_labels = []
    for prompt, output in zip(prompts, outputs):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

        row_ids = prompt_ids + output_ids
        row_labels = [-100] * len(prompt_ids) + output_ids   # mask the prompt span, score only the output (+ EOS)

        batch_input_ids.append(row_ids)
        batch_labels.append(row_labels)

    # Score in micro-batches of SCORE_BATCH_SIZE rows instead of one giant forward pass —
    # keeps peak memory (activations + fp32 logits over the full vocab) bounded regardless
    # of how many train instances a task has
    per_row_scores = []
    scoring_ctx = model.disable_adapter() if SHARED_SCORE_MODEL else contextlib.nullcontext()
    with scoring_ctx:
        for start in range(0, len(batch_input_ids), SCORE_BATCH_SIZE):
            chunk_ids = batch_input_ids[start:start + SCORE_BATCH_SIZE]
            chunk_labels = batch_labels[start:start + SCORE_BATCH_SIZE]

            # Manually right-pad this chunk to its own max sequence length
            max_len = max(len(ids) for ids in chunk_ids)
            input_ids, attention_mask, labels = [], [], []
            for ids, lbls in zip(chunk_ids, chunk_labels):
                pad_len = max_len - len(ids)
                input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
                labels.append(lbls + [-100] * pad_len)

            input_ids = torch.tensor(input_ids, device=DEVICE)              # (chunk_size, seq_len)
            attention_mask = torch.tensor(attention_mask, device=DEVICE)    # (chunk_size, seq_len)
            labels = torch.tensor(labels, device=DEVICE)                    # (chunk_size, seq_len)

            # Single forward pass over this chunk (no `labels=` kwarg — we hand-compute log-probs)
            with torch.no_grad():
                outputs_hf = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs_hf.logits                           # (chunk_size, seq_len, vocab_size)

            # Per row: sum of log P(token | preceding tokens) over the scored (non -100) positions
            summed_log_probs = _sum_sequence_logprob(logits, labels)                # (chunk_size,)

            # Count scored tokens per row; [:, 1:] mirrors the shift inside _sum_sequence_logprob
            # (position 0 is never scored — nothing precedes it)
            token_counts = (labels[:, 1:] != -100).sum(dim=-1)                      # (chunk_size,)

            # Average per token, so instances with longer outputs aren't systematically more negative
            per_row_scores.append(summed_log_probs / token_counts)

    # Concatenate chunk results back into a single (num_rows,) tensor, preserving input order
    return torch.cat(per_row_scores, dim=0)


def _sum_sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Shift so that logits at position i predict the token at position i+1
    shift_logits = logits[:, :-1, :]            # (batch_size, seq_len - 1, vocab_size)
    shift_labels = labels[:, 1:]                # (batch_size, seq_len - 1)

    # Fused cross-entropy = -log P(label); ignore_index zeroes the -100 positions
    # without materializing a full-vocab log-prob tensor like log_softmax would
    loss = F.cross_entropy(
        shift_logits.flatten(0, 1).float(),
        shift_labels.flatten(),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)

    # Sum log-probs over the valid (non -100) sequence positions
    return -loss.sum(dim=-1)   # (batch_size,)



def generate_preference_dataset(dataset: List[Dict], translator_model, translator_tokenizer) -> List[Dict]:
    preference_dataset = []
    for k in range(K):
        for task in tqdm(dataset, desc=f"Round {k + 1}/{K}"):
            # Extract soft prompt
            soft_prompt = task["soft_prompt"].to(DEVICE, dtype=DTYPE)

            # Duplicate it N times
            soft_prompt_embeds = soft_prompt.unsqueeze(0).expand(N, -1, -1)     # (N, soft_tokens, embed_dim)
            attention_mask = torch.ones(soft_prompt_embeds.shape[:2], dtype=torch.long, device=DEVICE)

            # Produce N translations
            with torch.no_grad():
                gen_ids = translator_model.generate(
                    inputs_embeds = soft_prompt_embeds,
                    attention_mask = attention_mask,
                    max_new_tokens = MAX_NEW_TOKENS,
                    do_sample = True,
                    temperature = TEMPERATURE,
                    top_p = TOP_P,
                    pad_token_id = translator_tokenizer.eos_token_id
                )

            # Decode the N gen_ids into N translations
            translations = translator_tokenizer.batch_decode(gen_ids, skip_special_tokens = True)
            translations = [txt.strip() for txt in translations]

            # Get Avg Score for each translation
            scores = get_scores(translations, task["train_instances"])

            # Find z_W and z_L
            w_idx, l_idx = torch.argmax(scores).item(), torch.argmin(scores).item()
            if w_idx == l_idx:
                print(f"Degenerate preference pairs generated for task {task['task_name']}")
            z_W, z_L = translations[w_idx], translations[l_idx]


            # Calculate log prob of producing z_W and z_L using the translator, conditioned on the soft prompt
            logp_ref_z_W = get_logprob_of_translation_given_soft_prompt(translator_model, translator_tokenizer, z_W, soft_prompt)
            logp_ref_z_L = get_logprob_of_translation_given_soft_prompt(translator_model, translator_tokenizer, z_L, soft_prompt)

            # Add to dataset
            preference_dataset.append({
                "z_prime": task["soft_prompt"],
                "z_W": z_W,
                "z_L": z_L,
                "logp_ref_z_W": logp_ref_z_W,
                "logp_ref_z_L": logp_ref_z_L,
            })

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
    parser.add_argument("--score-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Can be either: Any OpenAI LLM or HF Model")

    # Translator Model
    parser.add_argument("--lora-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lora-weights-path", type=str, default="./shared/mapper_lora_weights/General-DoD/meta-llama/Llama-3.1-8B-Instruct")

    # HyperParams
    parser.add_argument("-n", "--num-samples-to-generate", type=int, default=10)
    parser.add_argument("-k", "--scaling-factor", type=int, default=10)
    parser.add_argument("-t", "--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--score-batch-size", type=int, default=4, help="Micro-batch size used when scoring train instances with the LOGPROB score function")

    args, _ = parser.parse_known_args(args_list)

    # Define Global Variables
    global DEVICE, DTYPE, SCORE_FN, SCORE_MODEL_NAME, K, TOP_P, N, TEMPERATURE, MAX_NEW_TOKENS, SCORE_BATCH_SIZE

    # Parse all the arguments into Variables
    MAPPER_DATASET_PATH = args.mapper_dataset_path
    SCORE_FN = args.score_fn
    SAVE_DATASET_PATH = args.save_dataset_path + f"/preference_dataset{SCORE_FN}"
    SCORE_MODEL_NAME = args.score_model_name
    LORA_MODEL_NAME = args.lora_model_name
    LORA_WEIGHTS_PATH = args.lora_weights_path
    N = args.num_samples_to_generate
    K = args.scaling_factor
    TEMPERATURE = args.temperature
    MAX_NEW_TOKENS = args.max_new_tokens
    TOP_P = args.top_p
    SCORE_BATCH_SIZE = args.score_batch_size

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
    if SCORE_FN == "ROUGE-L":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please add `OPENAI_API_KEY` inside a .env file in project root")
        
        global OPENAI_CLIENT
        OPENAI_CLIENT = OpenAI(
            api_key=api_key, 
            max_retries=5,
        )

    elif SCORE_FN == "LOGPROB":
        global score_model, score_tokenizer, SHARED_SCORE_MODEL
        score_tokenizer = AutoTokenizer.from_pretrained(SCORE_MODEL_NAME)

        # Do this only if score model is from Llama family
        score_tokenizer.pad_token = score_tokenizer.eos_token

        if SCORE_MODEL_NAME == LORA_MODEL_NAME:
            # Avoid loading a second bf16 copy of the same base model — reuse the
            # translator's underlying weights, with the LoRA adapter disabled during scoring
            print("Score model matches translator base model — sharing weights (LoRA disabled during scoring).")
            score_model = translator_model
            SHARED_SCORE_MODEL = True
        else:
            print(f"Loading score model {SCORE_MODEL_NAME}...")
            score_model = AutoModelForCausalLM.from_pretrained(SCORE_MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
            score_model.eval()
            SHARED_SCORE_MODEL = False

    else:
        print(f"Unsupported Score Function: {SCORE_FN}!")
        exit(1)


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
    
