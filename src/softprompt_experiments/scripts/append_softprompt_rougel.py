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
    parser.add_argument("--output", type=str, default="./LLM_verbalization_w_rouge_l")
    args, _ = parser.parse_known_args(args_list)

    out_df = pd.read_csv(args.output+".csv")
    print(out_df.head())
    # print(f"Saving results to {args.output}.json...")
    # with open(args.output+".json", "w") as f:
    #     json.dump(data, f, indent=2)

    # -----------------------------
    # Paired t-test (recommended)
    # -----------------------------
    llm_scores = out_df['llm_verbalization_rouge_l']
    soft_scores = out_df['softprompt_rougel']

    t_stat, p_value = stats.ttest_rel(llm_scores, soft_scores, nan_policy='omit')

    print("\nPaired t-test (llm_verbalization vs soft):")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")


    # -----------------------------
    # Box plot
    # -----------------------------
    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [llm_scores, soft_scores],
        tick_labels=["LLM verbalization ROUGE-L", "Soft prompt ROUGE-L"]
    )

    plt.title("ROUGE-L Comparison: LLM Verbalization vs Soft Prompt")
    plt.ylabel("ROUGE-L Score")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plot_path = args.output + "_rouge_l_boxplot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nSaved box plot to: {plot_path}")

