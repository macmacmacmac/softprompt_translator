import json
import argparse
import os
import re
from transformers import AutoTokenizer
from openai import OpenAI
import evaluate
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROUGE_METRIC = evaluate.load("rouge")


# -----------------------------
# Driver
# -----------------------------
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print("=" * 100)
    print(f"\t\t\tRunning script: {exp_name}")
    print("=" * 100)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./verbalization_control_fewshot.json")
    args, _ = parser.parse_known_args(args_list)

    df = pd.read_json(args.input, orient='records')

    df= df.sort_values(by='verbalization_rouge_l', ascending=False)
    print(df.iloc[0]['hard_prompt'])
    print(df.iloc[0]['verbalization_mapper'])
    print("\n\n")
    print(df.iloc[1]['hard_prompt'])
    print(df.iloc[1]['verbalization_mapper'])
    print("\n\n")
    print(df.iloc[2]['hard_prompt'])
    print(df.iloc[2]['verbalization_mapper'])
    print("\n\n")
    print(df.iloc[3]['hard_prompt'])
    print(df.iloc[3]['verbalization_mapper'])

    # df_shuffled = df.sample(frac=1)

    # pairs = df_shuffled[['hard_prompt', 'verbalization_mapper']].head(25)

    # for i, (_, row) in enumerate(pairs.iterrows(), 1):
    #     print("=" * 100)
    #     print(f"PAIR {i}")
    #     print("-" * 100)
    #     print("Hard Prompt:\n")
    #     print(row['hard_prompt'])
    #     print("\n")
    #     print("Verbalization Mapper:\n")
    #     print(row['verbalization_mapper'])
    #     print("\n")

    # df['diff'] = df['llm_verbalization_rouge_l'] - df['inspect_llm_verbalization_rouge_l']
    # betters_bool_mask = df['diff'] > 0
    # better = df[betters_bool_mask]
    # better = better.sort_values(by='diff', ascending=False)

    # print(f"There are ({len(better)}) tasks where the mapper beat fewshot")
    # print(f"Example: {better['task_name'][:5]}")