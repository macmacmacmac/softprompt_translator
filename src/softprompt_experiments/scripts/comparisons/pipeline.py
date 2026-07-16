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

load_dotenv()  
ROUGE_METRIC = evaluate.load("rouge")

# -----------------------------
# Prompt templates
# -----------------------------
SYS_PROMPT_TEMPLATE = """# Task
{task_prompt}
"""

USR_PROMPT_TEMPLATE = """# Input
{input}

# Output
"""

FS_PROMPT_TEMPLATE = """
Look at the following input, output examples.
{examples}
This concludes the examples, now use this to give an output to the following user input query.
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
    parser.add_argument("--input", type=str, default="./shared/verbalizations/master_verbalizations_v2.json")
    parser.add_argument("--output", type=str, default="./shared/verbalizations/changed_master_verbalizations_v2.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args, _ = parser.parse_known_args(args_list)

    MODEL = args.model

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b-Instruct")

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

        mapper_prompt = str(dataset["mapper_hard_prompt"])
        inspect_prompt = str(dataset["inspect_hard_prompt"])

        fs_examples = "\n".join([f"Example Input: {t['input']}\nExample Output: {t['output']}" for t in train_instances])
        fs_prompt = FS_PROMPT_TEMPLATE.format(examples=fs_examples)

        gt_prompt = str(dataset["hard_prompt"])


        mapper_preds = []
        inspect_preds = []
        fs_preds = []
        gt_preds = []

        refs = []

        for instance in instances:
            input = instance["input"]
            gt_output = instance["output"]

            user_prompt = USR_PROMPT_TEMPLATE.format(input = input)

            mapper_sys_prompt = SYS_PROMPT_TEMPLATE.format(task_prompt = mapper_prompt)
            inspect_sys_prompt = SYS_PROMPT_TEMPLATE.format(task_prompt = inspect_prompt)
            fs_sys_prompt = SYS_PROMPT_TEMPLATE.format(task_prompt = fs_prompt)
            gt_sys_prompt = SYS_PROMPT_TEMPLATE.format(task_prompt = gt_prompt)

            mapper_pred = get_llm_prediction(mapper_sys_prompt, user_prompt, model=MODEL)
            inspect_pred = get_llm_prediction(inspect_sys_prompt, user_prompt, model=MODEL)
            fs_pred = get_llm_prediction(fs_sys_prompt, user_prompt, model=MODEL)
            gt_pred = get_llm_prediction(gt_sys_prompt, user_prompt, model=MODEL)

            instance["mapper_output"] = mapper_pred
            instance["inspect_output"] = inspect_pred
            instance["fs_output"] = fs_pred
            instance["gt_output"] = gt_pred

            mapper_preds.append(mapper_pred)
            inspect_preds.append(inspect_pred)
            fs_preds.append(fs_pred)
            gt_preds.append(gt_pred)

            refs.append(gt_output)

        # compute ROUGE-L over dataset
        dataset["mapper_task_rougeL"] = ROUGE_METRIC.compute(
            predictions=mapper_preds,
            references=refs,
            use_stemmer=True
        )["rougeL"]
        dataset["inspect_task_rougeL"] = ROUGE_METRIC.compute(
            predictions=inspect_preds,
            references=refs,
            use_stemmer=True
        )["rougeL"]
        dataset["fs_task_rougeL"] = ROUGE_METRIC.compute(
            predictions=fs_preds,
            references=refs,
            use_stemmer=True
        )["rougeL"]
        dataset["gt_task_rougeL"] = ROUGE_METRIC.compute(
            predictions=gt_preds,
            references=refs,
            use_stemmer=True
        )["rougeL"]

        # ipdb.set_trace()
    
        # break

    print(f"Saving results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)


