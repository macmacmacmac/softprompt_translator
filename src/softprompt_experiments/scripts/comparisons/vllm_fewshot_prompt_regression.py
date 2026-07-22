import json
import argparse
import os
import time
from transformers import AutoTokenizer
import evaluate
from dotenv import load_dotenv
from tqdm import tqdm
from vllm import LLM, SamplingParams

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
# Driver
# -----------------------------
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print("=" * 100)
    print(f"\t\t\tRunning script: {exp_name}")
    print("=" * 100)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./shared/verbalizations/master_verbalizations_8b.json")
    parser.add_argument("--output", type=str, default="./shared/verbalizations/master_verbalizations_8b_fsr_test.json")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    args, _ = parser.parse_known_args(args_list)


    MODEL = args.model

    print(f"Loading vLLM model: {MODEL}...")
    llm = LLM(model=MODEL)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    tokenizer = llm.get_tokenizer()

    print(f"Loading data from {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)

    flat_queries = []

    for dataset in data:
        train_instances = dataset["train_instances"]

        fs_examples = "\n".join([f"Input: {t['input']}\nOutput: {t['output']}\n" for t in train_instances])
        fs_infer_prompt = FS_INFER_PROMPT.format(examples=fs_examples)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": fs_infer_prompt},
        ]
        
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        flat_queries.append(prompt_text)

    print("Generating prompts via vLLM...")
    if flat_queries:
        outputs = llm.generate(flat_queries, sampling_params)
        
        for dataset, out in zip(data, outputs):
            dataset['fsr_hard_prompt'] = out.outputs[0].text.strip()

    print(f"Saving results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)

