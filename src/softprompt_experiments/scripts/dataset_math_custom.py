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

from softprompt_experiments.utils import tokenize_and_save

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples_per_dataset", type=int, default=500)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_dataset_custom")
    args, _ = parser.parse_known_args(args_list)
    
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    NUM_SAMPLES_PER = args.num_samples_per_dataset
    # NUM_DATASETS = args.num_datasets
    SAVE_DIR = args.save_directory

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # generate dataset
    formulas = [
        {   #gravitational force between two objects
            'expr': "(6.673e-11) * (x*y)/(z**2)",
            'input_template': "Input: m1={x},m2={y},r={z}\n Answer: ",
            'output_template': "{out:.3e} N"
        },
        {
            #circular orbital velocity
            'expr': "((6.673e-11) * x / y)**0.5",
            'input_template': "Input: M={x}, r={y}\n Answer: ",
            'output_template': "{out:.3e} m/s"
        },
        {
            #scpacetime interval
            'expr': "(x * 2.998e8)**2 - (y)**2 - (z)**2",
            'input_template': "Input: dt={x}, dx={y}, dy={z}\n Answer: ",
            'output_template': "{out:.3e}"
        },
        {
            #relative energy
            'expr': "((1 / (1 - (y**2)/(2.998e8)**2)**0.5) - 1) * x * (2.998e8)**2",
            'input_template': "Input: m={x},v={y}\n Answer: ",
            'output_template': "{out:.3e} J"
        },
        {
            #attraction between two charges
            'expr': "(8.99e9) * (x*y)/(z**2)",
            'input_template': "Input: q1={x},q2={y},r={z}\n Answer: ",
            'output_template': "{out:.3e} N"
        },
        {
            #electric potential energy
            'expr': "(8.99e9) * (x*y)/z",
            'input_template': "Input: q1={x},q2={y},r={z}\n Answer: ",
            'output_template': "{out:.3e} J"
        },
        {
            #number of particles from mass
            'expr': "(6.022e23) * x/y",
            'input_template': "Input: m={x}, M={y}\n Answer: ",
            'output_template': "{out:.3e} particles"
        },
        {
            #faraday constant definition
            'expr': "(x * 1e24/ 6.022e23) * y",
            'input_template': "Input: N={x}, M={y}\n Answer: ",
            'output_template': "{out:.3e} g"
        },
        {
            #kinetic energy
            'expr': "0.5*x*(y**2)",
            'input_template': "Input: m={x},v={y}\n Answer: ",
            'output_template': "{out:.3e} J"
        },
        {
            # Wave speed
            'expr': "x*y",
            'input_template': "Input: f={x},λ={y}\n Answer: ",
            'output_template': "{out:.3e} m/s"
        },
        {
            # Ideal gas law
            'expr': "(8.3145) * y * z / x",
            'input_template': "Input: P={x},n={y},T={z}\n Answer: ",
            'output_template': "{out:.3e} m³"
        },
        {
            #volume of an ellipsoid ish
            'expr': "(4/3)*3.1415*x*y*z",
            'input_template': "Input: a={x},b={y},c={z}\n Answer: ",
            'output_template': "{out:.3f}"
        },
    ]
    def expr_to_func(expr):        
        func = np.vectorize(lambda x, y, z: eval(expr))
        return func, expr

    def get_sentences_from_func(func, input_template, output_template, num_samples, num_vars):
        # x = np.random.randint(low=1, high=1000, size=num_samples)
        # y = np.random.randint(low=1, high=1000, size=num_samples)
        # z = np.random.randint(low=1, high=1000, size=num_samples)

        triples = set()

        # high = 10 if num_vars > 2 else 100
        high = 100

        while len(triples) < num_samples:
            triple = (
                np.random.randint(1, high),
                np.random.randint(1, high),
                np.random.randint(1, high),
            )
            triples.add(triple)

        x, y, z = map(np.array, zip(*triples))

        outputs = func(x, y, z)

        input_sentences = [
            input_template.format(x=x,y=y,z=z)
            for x,y,z in zip(x,y,z)
        ]

        target_sentences = [output_template.format(out=out) for out in outputs]

        return input_sentences, target_sentences

    # pipeline
    for i, formula in enumerate(formulas):
        save_dir = os.path.join(SAVE_DIR, f"dataset_{i}")
        os.makedirs(save_dir, exist_ok=True)

        expr = formula['expr']
        input_template = formula['input_template']
        output_template = formula['output_template']
        func, expr = expr_to_func(expr)
        num_vars = ("x" in expr) + ("y" in expr) + ("z" in expr)
        input_sentences, target_sentences = get_sentences_from_func(
            func, input_template, output_template, NUM_SAMPLES_PER, num_vars
        )
        
        print(input_sentences[0], target_sentences[0])

        tokenized = tokenize_and_save(input_sentences, target_sentences, save_dir, expr, tokenizer)

        

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









