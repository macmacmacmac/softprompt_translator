import os
import argparse
import torch
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
ROUGE_METRIC = evaluate.load("rouge")

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
    return []


def get_rougeL_scores(
        translations: List[str],
        train_instances: List[Dict]
    ) -> torch.Tensor:
    rougeL_scores = []
    y = [t["output"] for t in train_instances]
    for translation in translations:
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
    ):
    pass


def get_logprob_for_sequence(
        model: torch.nn.Module,
        tokenizer,
        sequence: str
    ):
    # Tokenize the sequence
    inputs = tokenizer(sequence, return_tensors="pt")
    
    # Perform forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits                                 # (batch_size, seq_len, vocab_size)



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

    # Dataset Path
    parser.add_argument("--mapper-dataset-path", type=str, default="./shared/datasets/mapper_training_dataset/General-DoD-DPO")

    # Score Model
    parser.add_argument("--score-fn", type=str, default="ROUGE-L", help="Can be either: ROUGE-L | LOGPROB")
    parser.add_argument("--score-model-name", type=str, default="gpt-4o-mini", help="Can be either: Any OpenAI LLM or HF Model")

    # Translator Model
    parser.add_argument("--lora-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lora-weights-path", type=str, default="./shared/mapper_lora_weights/General-DoD")

    # HyperParams
    parser.add_argument("-n", "--num-samples-to-generate", type=int, default=10)
    parser.add_argument("-k", "--scaling-factor", type=int, default=10)
    parser.add_argument("-t", "--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=0.9)

    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MAPPER_DATASET_PATH = args.mapper_dataset_path

    global SCORE_FN
    SCORE_FN = args.score_fn

    global SCORE_MODEL_NAME
    SCORE_MODEL_NAME = args.score_model_name

    LORA_MODEL_NAME = args.lora_model_name
    LORA_WEIGHTS_PATH = args.lora_weights_path
    N = args.num_samples_to_generate
    K = args.scaling_factor
    TEMPERATURE = args.temperature
    MAX_NEW_TOKENS = args.max_new_tokens
    TOP_P = args.top_p

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
        score_tokenizer = AutoTokenizer.from_pretrained(SCORE_MODEL_NAME)

        # Do this only if score model is from Llama family
        score_tokenizer.pad_token = score_tokenizer.eos_token

        print(f"Loading score model {SCORE_MODEL_NAME}...")
        score_model = AutoModelForCausalLM.from_pretrained(SCORE_MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
        score_model.eval()

    else:
        print(f"Unsupported Score Function: {SCORE_FN}!")
        exit(1)


    # ┌───────────────────────────────────────────────┐
    # │      TRAIN PREFERENCE DATASET GENERATION      │
    # └───────────────────────────────────────────────┘
    preference_dataset = []
    # TODO: Add a loop for K
    for task in train_dataset:
        # Extract soft prompt
        soft_prompt = task["soft_prompt"].to(DEVICE, dtype=DTYPE)

        # Duplicate it N times
        soft_prompt_embeds = soft_prompt.unsqueeze(0).expand(N, -1, -1)     # (N, soft_tokens, embed_dim)
        attention_mask = torch.ones(soft_prompt_embeds.shape[:2], dtype=torch.long, device=DEVICE)

        # Produce N translations
        with torch.no_grad():
            gen_ids = translator_model.generate(
                input_embeds = soft_prompt_embeds,
                attention_mask = attention_mask,
                max_new_tokens = MAX_NEW_TOKENS,
                do_sample = True,
                temperature = TEMPERATURE,
                top_p = TOP_P,
                pad_token_id = translator_tokenizer.eos_token_id
            )

        # Decode the N gen_ids into N translations
        translations = translator_tokenizer.batch_decode(gen_ids, skip_special_tokens = True)
        translations = [txt.strip for txt in translations]

        # Get Avg Score for each translation
        scores = get_scores()

        # Find z_W and z_L
        w_idx, l_idx = torch.argmax(scores).item(), torch.argmin(scores).item()
        z_W, z_L = translations[w_idx], translations[l_idx]


        # TODO: Calculate log prob of producing z_W and z_L using the translator



        # Add to dataset
        preference_dataset.append({
            "z_prime": task["soft_orompt"],
            "z_W": z_W,
            "z_L": z_L,

            # "logp_ref_z_W": ...
            # "logp_ref_z_L": ...
        })



        



    

