import json
import argparse
import os
import time
from transformers import AutoTokenizer
from openai import OpenAI, RateLimitError
import evaluate
from dotenv import load_dotenv
from tqdm import tqdm
import ipdb 

from softprompt_experiments.scripts.comparisons.verbalizations_rougeL_visualizer import build_visual

load_dotenv()  
ROUGE_METRIC = evaluate.load("rouge")

# -----------------------------
# Prompt templates
# -----------------------------


SYSTEM_PROMPT = """You are a helpful assistant. Follow the task exactly."""

FS_INFER_PROMPT = """
Look at the following input, output examples.
{examples}
This concludes the examples.
Based on this pattern, generate a concise, short 2-3 sentence prompt clearly explaining 
to another LLM how to perform the task.
"""



# -----------------------------
# LLM call
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please add `OPENAI_API_KEY` inside a .env file in project root")

client = OpenAI(
    api_key=api_key, 
    max_retries=5,
)

def get_llm_prediction(
    user_prompt: str, 
    system_prompt: str,
    max_retries: int = 5,
    **kwargs
):
    defaults = {
        "model": "gpt-4o-mini",
    }
    params = {**defaults, **kwargs}
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
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

# -----------------------------
# Driver
# -----------------------------
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print("=" * 100)
    print(f"\t\t\tRunning script: {exp_name}")
    print("=" * 100)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./shared/verbalizations/master_verbalizations_v3.json")
    parser.add_argument("--output", type=str, default="./shared/verbalizations/fs_prompt_regression_master_verbalizations_v3.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    
    args, _ = parser.parse_known_args(args_list)


    MODEL = args.model

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found.")

    global client
    client = OpenAI(api_key=api_key)

    print(f"Loading data from {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)

    for dataset in tqdm(data):
        # ipdb.set_trace()
        instances = dataset["val_instances"]
        train_instances = dataset["train_instances"]

        fs_examples = "\n".join([f"Input: {t['input']}\nOutput: {t['output']}\n" for t in train_instances])
        fs_infer_prompt = FS_INFER_PROMPT.format(examples=fs_examples)

        fs_pred = get_llm_prediction(SYSTEM_PROMPT, fs_infer_prompt, model=MODEL)

        dataset['fs_hard_prompt'] = fs_pred

        # ipdb.set_trace()

        print(f"Saving results to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(data, f, indent=4)


