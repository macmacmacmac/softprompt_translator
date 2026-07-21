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

    parser.add_argument("--do_all", action="store_true")
    parser.add_argument("--do_fs", action="store_true")
    parser.add_argument("--do_mapper", action="store_true")
    parser.add_argument("--do_gt", action="store_true")
    parser.add_argument("--do_inspect", action="store_true")
    parser.add_argument("--do_mapper10x", action="store_true")

    args, _ = parser.parse_known_args(args_list)

    do_all = args.do_all
    do_fs = args.do_fs
    do_mapper = args.do_mapper
    do_gt = args.do_gt
    do_inspect = args.do_inspect
    do_mapper10x = args.do_mapper10x

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
        val_instances = dataset["val_instances"]
        train_instances = dataset["train_instances"]

        mapper_prompt = str(dataset["mapper_hard_prompt"])
        mapper10x_prompt = str(dataset["mapper10x_logprob_z_W_z_G"])
        inspect_prompt = str(dataset["inspect_hard_prompt"])

        fs_examples = "\n".join([f"Input: {t['input']}\nOutput: {t['output']}\n" for t in train_instances])
        fs_prompt = FS_TASK_PROMPT.format(examples=fs_examples)

        gt_prompt = str(dataset["hard_prompt"])

        

        mapper_preds = []
        mapper10x_preds = []
        inspect_preds = []
        fs_preds = []
        gt_preds = []
        

        refs = []

        for instance in val_instances:
            input = instance["input"]
            gt_output = instance["output"]

            mapper_usr_prompt = USR_PROMPT.format(task_prompt = mapper_prompt, input = input)
            mapper10x_usr_prompt = USR_PROMPT.format(task_prompt = mapper10x_prompt, input = input)
            inspect_usr_prompt = USR_PROMPT.format(task_prompt = inspect_prompt, input = input)
            fs_usr_prompt = USR_PROMPT.format(task_prompt = fs_prompt, input = input)
            gt_usr_prompt = USR_PROMPT.format(task_prompt = gt_prompt, input = input)
            

            

            if do_mapper or do_all:
                mapper_pred = get_llm_prediction(SYSTEM_PROMPT, mapper_usr_prompt, model=MODEL)
                instance["mapper_output"] = mapper_pred
                mapper_preds.append(mapper_pred)

            if do_mapper10x or do_all:
                mapper10x_pred = get_llm_prediction(SYSTEM_PROMPT, mapper10x_usr_prompt, model=MODEL)
                instance["mapper10x_logprob_z_W_z_G_output"] = mapper10x_pred
                mapper10x_preds.append(mapper10x_pred)

            if do_inspect or do_all:
                inspect_pred = get_llm_prediction(SYSTEM_PROMPT, inspect_usr_prompt, model=MODEL)
                instance["inspect_output"] = inspect_pred
                inspect_preds.append(inspect_pred)

            if do_fs or do_all:
                fs_pred = get_llm_prediction(SYSTEM_PROMPT, fs_usr_prompt, model=MODEL)
                instance["fs_output"] = fs_pred
                fs_preds.append(fs_pred)

            if do_gt or do_all:
                gt_pred = get_llm_prediction(SYSTEM_PROMPT, gt_usr_prompt, model=MODEL)
                instance["gt_output"] = gt_pred
                gt_preds.append(gt_pred)


            refs.append(gt_output)

        # compute ROUGE-L over dataset
        if do_mapper or do_all:
            dataset["mapper_task_rougeL"] = ROUGE_METRIC.compute(
                predictions=mapper_preds,
                references=refs,
                use_stemmer=True
            )["rougeL"]

        if do_mapper10x or do_all:
            dataset["mapper10x_logprob_z_W_z_G_task_rougeL"] = ROUGE_METRIC.compute(
                predictions=mapper10x_preds,
                references=refs,
                use_stemmer=True
            )["rougeL"]

        if do_inspect or do_all:
            dataset["inspect_task_rougeL"] = ROUGE_METRIC.compute(
                predictions=inspect_preds,
                references=refs,
                use_stemmer=True
            )["rougeL"]

        if do_fs or do_all:
            dataset["fs_task_rougeL"] = ROUGE_METRIC.compute(
                predictions=fs_preds,
                references=refs,
                use_stemmer=True
            )["rougeL"]

        if do_gt or do_all:
            dataset["gt_task_rougeL"] = ROUGE_METRIC.compute(
                predictions=gt_preds,
                references=refs,
                use_stemmer=True
            )["rougeL"]

        # ipdb.set_trace()
    
        # break

        tqdm.write(f"Saving results to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(data, f, indent=4)


