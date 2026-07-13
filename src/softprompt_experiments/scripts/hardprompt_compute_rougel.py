import json
import argparse
import os
import re
from transformers import AutoTokenizer
from openai import OpenAI
import evaluate
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()  
ROUGE_METRIC = evaluate.load("rouge")

# -----------------------------
# LLM call
# -----------------------------
def get_llm_prediction(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Follow the task exactly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,  # greedy
    )

    return response.choices[0].message.content.strip()


# -----------------------------
# Driver
# -----------------------------
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print("=" * 100)
    print(f"\t\t\tRunning script: {exp_name}")
    print("=" * 100)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbalization", type=str, default="./verbalizations_enriched_w_instances.json")
    parser.add_argument("--output", type=str, default="./verbalizations_enriched_w_instances.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args, _ = parser.parse_known_args(args_list)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b-Instruct")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found.")

    client = OpenAI(api_key=api_key)

    print(f"Loading data from {args.verbalization}...")
    with open(args.verbalization, "r") as f:
        data = json.load(f)

    for item in tqdm(data):
        task_prompt = str(item.get("verbalization", "")).strip()
        instances = item.get("instances", [])

        preds = []
        refs = []

        for instance in instances:
            user_input = instance["input"]
            ground_truth = instance["output"]

            full_prompt = f"{task_prompt}\n\nInput:\n{user_input}\n\nOutput:\n"

            pred = get_llm_prediction(client, args.model, full_prompt)

            instance["pred_control"] = pred

            preds.append(pred)
            refs.append(ground_truth)

        # compute ROUGE-L over dataset
        rouge_results = ROUGE_METRIC.compute(
            predictions=preds,
            references=refs,
            use_stemmer=True
        )

        item["mapper_verbalization_rougel"] = rouge_results["rougeL"]

    print(f"Saving results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)


