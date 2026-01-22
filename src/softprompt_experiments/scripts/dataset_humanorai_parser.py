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

from softprompt_experiments.utils import tokenize_and_save, batched_tokenize_and_save
import pandas as pd

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_directory", type=str, default="./datasets/human_or_ai_dataset")
    args, _ = parser.parse_known_args(args_list)
    
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    # NUM_DATASETS = args.num_datasets
    SAVE_DIR = args.save_directory

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token


    # pipeline
    save_dir = os.path.join(SAVE_DIR, "dataset_0")

    df = pd.read_csv("./datasets/human_or_ai_dataset/human_or_ai_dataset.csv")
    
    # input_sentences = [f"\nSample text: {input_sent}\nAnswer: " for input_sent in df.text.tolist()]
    prefix = "\nSample text:\n\""
    suffix = "...\"\nAnswer:\n"
    target_sentences = [('AI generated' if target==1 else 'Human written') for target in df.generated.tolist()]

    # print(input_sentences[:3])
    # print(target_sentences[:3])

    tokenized = batched_tokenize_and_save(
        df.text.tolist(),
        prefix,
        suffix, 
        target_sentences, 
        save_dir, 
        "human or ai", 
        tokenizer, 
        input_max_length=128, 
        target_max_length=4
    )

    print(tokenizer.decode(tokenized['tokenized_samples']['input_ids'][0]))

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









