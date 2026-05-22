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
    parser.add_argument("--verbalization", type=str, default="./prefill_verbalizations_w_instances.json")
    parser.add_argument("--output", type=str, default="./prefill_verbalizations_all.json")
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
        #without random
        training_instances = item.get("training_instances", [])
        prefill_prompt = str(item.get("prefill_verbalization", "")).strip()
        # task_prompt = f"{mapper_prompt}\n\nExamples"
        fewshot_prompt = f"\n\nExamples"
        for instance in training_instances[:3]:
            fewshot_prompt += f"\nInput:\n{instance['input']}\n\nOutput:\n{instance['output']}"
        fewshot_prompt += "\nPrediction:"
        item['verbalization_fewshot'] = fewshot_prompt

        # load actual instances
        instances = item.get("instances", [])

        fewshot_preds = []
        prefill_preds = []
        refs = []

        for instance in instances:
            user_input = instance["input"]
            ground_truth = instance["output"]

            full_fewshot_prompt = f"{fewshot_prompt}\n\nInput:\n{user_input}\n\nOutput:\n"
            full_prefill_prompt = f"{prefill_prompt}\n\nInput:\n{user_input}\n\nOutput:\n"

            pred_fewshot = get_llm_prediction(client, args.model, full_fewshot_prompt)
            pred_prefill = get_llm_prediction(client, args.model, full_prefill_prompt)

            instance["pred_fewshot"] = pred_fewshot
            instance["pred_prefill"] = pred_prefill

            fewshot_preds.append(pred_fewshot)
            prefill_preds.append(pred_prefill)
            refs.append(ground_truth)

        # compute ROUGE-L over dataset
        fewshot_rouge_results = ROUGE_METRIC.compute(
            predictions=fewshot_preds,
            references=refs,
            use_stemmer=True
        )
        item["fewshot_verbalization_rougel"] = fewshot_rouge_results["rougeL"]

        prefill_rouge_results = ROUGE_METRIC.compute(
            predictions=prefill_preds,
            references=refs,
            use_stemmer=True
        )
        item["prefill_verbalization_rougel"] = prefill_rouge_results["rougeL"]


    print(f"Saving results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)


