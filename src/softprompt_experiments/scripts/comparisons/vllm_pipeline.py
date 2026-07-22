import json
import argparse
import os
import time
from transformers import AutoTokenizer
import evaluate
from dotenv import load_dotenv
from tqdm import tqdm
from vllm import LLM, SamplingParams

from softprompt_experiments.scripts.comparisons import verbalizations_rougeL_visualizer

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
# Driver
# -----------------------------
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print("=" * 100)
    print(f"\t\t\tRunning script: {exp_name}")
    print("=" * 100)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./shared/verbalizations/master_verbalizations_8b.json")
    parser.add_argument("--output", type=str, default="./shared/verbalizations/master_verbalizations_8b_fsr_and_1x.json")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--methods", nargs="+", default=["mapper", "inspect", "gt", "fs"], help="List of method prefixes to evaluate (e.g. mapper inspect gt fs mapper10x)")

    args, _ = parser.parse_known_args(args_list)

    MODEL = args.model
    methods = args.methods

    print(f"Loading vLLM model: {MODEL}...")
    llm = LLM(model=MODEL)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    tokenizer = llm.get_tokenizer()

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

        flat_queries = []
        metadata = []

        for i, instance in enumerate(instances):
            input_text = instance["input"]
            gt_output = instance["output"]
            refs.append(gt_output)

            for m in list(method_preds.keys()):
                usr_prompt = USR_PROMPT.format(task_prompt=prompts[m], input=input_text)
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": usr_prompt},
                ]
                
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                flat_queries.append(prompt_text)
                metadata.append((i, m))

        if flat_queries:
            outputs = llm.generate(flat_queries, sampling_params, use_tqdm=False)
            
            for out, (i, m) in zip(outputs, metadata):
                pred = out.outputs[0].text.strip()
                instances[i][f"{m}_output"] = pred
                method_preds[m].append(pred)

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
