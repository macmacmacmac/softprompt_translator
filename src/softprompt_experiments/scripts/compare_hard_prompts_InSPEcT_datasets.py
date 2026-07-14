import json
import argparse
import os
import re
from openai import OpenAI
import evaluate
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset
import ipdb

# Load all env variables
load_dotenv()  

# Init ROUGE METRIC calculator
ROUGE_METRIC = evaluate.load("rouge")

PROMPT_TEMPLATE = "{input} {prompt}"


def get_llm_prediction(model, prompt):
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Follow the task exactly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,  # greedy
    )

    return response.choices[0].message.content.strip()


def load_test_dataset_from_hf(dataset_name):
    if dataset_name == "sst2":
        dataset = load_dataset("stanfordnlp/sst2", split="validation")
    elif dataset_name == "sst5":
        dataset = load_dataset("SetFit/sst5", split="test")
    elif dataset_name == "ag_news":
        dataset = load_dataset("fancyzhx/ag_news", split="test")
    elif dataset_name == "subj":
        dataset = load_dataset("SetFit/subj", split="test")
    elif dataset_name == "TREC-QC":
        dataset = load_dataset("SetFit/TREC-QC", split="test")
    else:
        print(f"Unsupported dataset: {dataset_name}")
        exit(1)
    return dataset


# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print("=" * 100)
    print(f"\t\t\tRunning script: {exp_name}")
    print("=" * 100)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbalizations-path", type=str, default="./verbalizations/verbalizations_InSPEcT_datasets.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL = args.model
    VERBALIZATIONS_PATH = args.verbalizations_path

    # Parse env variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found.")

    # Init OpenAI Client
    global openai_client
    openai_client = OpenAI(api_key=api_key)

    print(f"Loading {VERBALIZATIONS_PATH} ..")
    with open(VERBALIZATIONS_PATH, "r") as f:
        datasets = json.load(f)

    for dataset in tqdm(datasets):

        # Load Testing dataset from HF
        test_dataset = load_test_dataset_from_hf(dataset["dataset"])

        # Fetch Hard Prompts
        translator_hard_prompt = dataset["mapper_verbalization"]
        rl_prompt_hard_prompt = dataset["rl_prompt_verbalization"]

        ipdb.set_trace()

        # Create Prompts to send to OpenAI





