import torch
import copy
import numpy as np
import argparse
import os
from transformers import (
    AutoTokenizer,
    # AutoModelForCausalLM,
)
from tqdm.auto import tqdm

from softprompt_experiments.utils import tokenize_and_save, log_json

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_datasets", type=int, default=10)
    parser.add_argument("--num_samples_per_dataset", type=int, default=500)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_datasetv2")
    args, _ = parser.parse_known_args(args_list)
    
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    NUM_SAMPLES_PER = args.num_samples_per_dataset
    NUM_DATASETS = args.num_datasets
    SAVE_DIR = args.save_directory

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # generate dataset
    def sample_random_function():
        # Possible unary or binary ops
        basic_ops = ["+", "-", "*"]
        special_ops = ["min", "max"]#"median"]  # variadic
        
        # Choose how many variables the expression will use
        var_choices = ["x", "y", "z"]
        num_vars = np.random.choice([1, 2, 3], p=[0.2, 0.3, 0.5])
        vars_used = np.random.choice(var_choices, size=num_vars, replace=False)
        vars_used = sorted(vars_used, key=var_choices.index)
        
        # Create coefficient mapping for used vars
        coefs = {v: np.random.randint(1, 10) for v in vars_used}

        # Helper to refer to a term like "3*x"
        def term(v):
            return f"{coefs[v]}*{v}"

        # === Expression templates ===
        templates = []

        # 1 variable templates
        if num_vars == 1:
            v = vars_used[0]
            templates += [
                f"{term(v)}",
                f"{term(v)} {np.random.choice(basic_ops)} {np.random.randint(1,10)}",
                f"{np.random.choice(special_ops)}([{term(v)}, {np.random.randint(1,10)}])",
            ]

        # 2 variable templates
        if num_vars == 2:
            v1, v2 = vars_used
            op = np.random.choice(basic_ops)
            templates += [
                f"{term(v1)} {op} {term(v2)}",
                f"({term(v1)} {op} {term(v2)}) {np.random.choice(basic_ops)} {np.random.randint(1,10)}",
                f"{np.random.choice(special_ops)}([{term(v1)}, {term(v2)}])",
            ]

        # 3 variable templates
        if num_vars == 3:
            v1, v2, v3 = vars_used
            op1, op2 = np.random.choice(basic_ops, size=2)
            templates += [
                f"{term(v1)} {op1} {term(v2)} {op2} {term(v3)}",
                f"{np.random.choice(special_ops)}([{term(v1)}, {term(v2)}, {term(v3)}])",
                f"({term(v1)} {op1} {term(v2)}) {op2} {term(v3)}",
                f"{term(v1)} {op1} ({term(v2)} {op2} {term(v3)})",
            ]

        # Pick a template
        expr = np.random.choice(templates)

        # Define safe eval environment
        safe_env = {
            "min": lambda lst: np.min(lst),
            "max": lambda lst: np.max(lst),
            # "median": lambda lst: np.median(lst),
        }

        # Build a function of x,y,z regardless of usage
        def func(x, y, z):
            # Provide missing ones as 0 but they won't be used in expr
            local_env = {"x": x, "y": y, "z": z}
            local_env.update(safe_env)
            return eval(expr, {"__builtins__": {}}, local_env)

        return np.vectorize(func), expr, vars_used


    def get_sentences_from_func(func, vars_used, NUM_SAMPLES_PER):
        # Sample small integer input domain
        xs = np.random.randint(0, 10, size=NUM_SAMPLES_PER)
        ys = np.random.randint(0, 10, size=NUM_SAMPLES_PER)
        zs = np.random.randint(0, 10, size=NUM_SAMPLES_PER)

        outputs = func(xs, ys, zs)

        # Build input sentences only with included variables
        input_sentences = []
        for x, y, z in zip(xs, ys, zs):
            pieces = []
            if "x" in vars_used: pieces.append(f"x={x}")
            if "y" in vars_used: pieces.append(f"y={y}")
            if "z" in vars_used: pieces.append(f"z={z}")
            inp = "Input: " + ", ".join(pieces) + "\nOutput: "
            input_sentences.append(inp)

        target_sentences = [f"{out}" for out in outputs]    
        return input_sentences, target_sentences

    # pipeline
    for i in tqdm(range(NUM_DATASETS)):
        save_dir = os.path.join(SAVE_DIR, f"dataset_{i}")
        os.makedirs(save_dir, exist_ok=True)

        func, expr, vars_used = sample_random_function()
        input_sentences, target_sentences = get_sentences_from_func(func, vars_used, NUM_SAMPLES_PER)
        # print('expr:',expr)
        # print('input_sentences:',input_sentences)
        # print('target_sentences:',target_sentences)

        tokenized = tokenize_and_save(input_sentences, target_sentences, save_dir, expr, tokenizer)
        log_json(
            os.path.join(save_dir,f'raws.json'), 
            {
                'hardprompt':expr,
                'input_sentences':input_sentences,
                'target_sentences':target_sentences
            }
        )

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









