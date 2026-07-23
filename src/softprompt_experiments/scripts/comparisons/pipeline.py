import json
import argparse
import os
import time
from openai import OpenAI, RateLimitError
import evaluate
from dotenv import load_dotenv
from tqdm import tqdm
import ipdb 

load_dotenv()  
ROUGE_METRIC = evaluate.load("rouge")

# -----------------------------
# Prompt templates
# -----------------------------


SYSTEM_PROMPT = """You are a helpful assistant. Follow the task exactly."""

USR_PROMPT = """{task_prompt}

Input:
{input}

Output:
"""

FS_TASK_PROMPT = """
Look at the following input, output examples.
{examples}
This concludes the examples.
Based on this pattern, predict the output to this input.
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
    parser.add_argument("--output", type=str, default="./shared/verbalizations/test.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--methods", nargs="+", default=["mapper", "inspect", "gt", "fs"], help="List of method prefixes to evaluate (e.g. mapper inspect gt fs mapper10x)")

    args, _ = parser.parse_known_args(args_list)

    MODEL = args.model
    methods = args.methods

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found.")

    global client
    client = OpenAI(api_key=api_key)

    print(f"Loading data from {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)

    for dataset in tqdm(data):
        instances = dataset["val_instances"]
        train_instances = dataset.get("train_instances", [])

        # Build prompt string for each method
        prompts = {}
        for m in methods:
            if m.lower() in ("fs", "fewshot"):
                fs_examples = "\n".join([f"Input: {t['input']}\nOutput: {t['output']}\n" for t in train_instances])
                prompts[m] = FS_TASK_PROMPT.format(examples=fs_examples)
            elif f"{m}_hard_prompt" in dataset:
                prompts[m] = str(dataset[f"{m}_hard_prompt"])
            elif m == "gt" and "hard_prompt" in dataset:
                prompts[m] = str(dataset["hard_prompt"])
            else:
                print(f"Warning: Prompt for method '{m}' not found in dataset. Skipping '{m}'.")
                prompts[m] = None

        method_preds = {m: [] for m in methods if prompts[m] is not None}
        refs = []


        for instance in instances:
            input_text = instance["input"]
            gt_output = instance["output"]

            for m in list(method_preds.keys()):
                usr_prompt = USR_PROMPT.format(task_prompt=prompts[m], input=input_text)
                pred = get_llm_prediction(user_prompt=usr_prompt, system_prompt=SYSTEM_PROMPT, model=MODEL)
                instance[f"{m}_output"] = pred
                method_preds[m].append(pred)


            refs.append(gt_output)

        # compute ROUGE-L over dataset for each method
        for m, preds in method_preds.items():
            dataset[f"{m}_task_rougeL"] = ROUGE_METRIC.compute(
                predictions=preds,
                references=refs,
                use_stemmer=True
            )["rougeL"]

        print(f"Saving results to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(data, f, indent=4)
