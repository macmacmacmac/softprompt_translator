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
    parser.add_argument("--output", type=str, default="./shared/verbalizations/master_verbalizations_v2")
    args, _ = parser.parse_known_args(args_list)

    out_df = pd.read_json(args.output+".json")
    print(out_df.head())
    # print(f"Saving results to {args.output}.json...")
    # with open(args.output+".json", "w") as f:
    #     json.dump(data, f, indent=2)

    # col_name0 = 'fs_task_rougeL'
    col_names = [
        'fs_task_rougeL',
        'soft_task_rougeL',
        'gt_task_rougeL',
        'mapper_task_rougeL',
        'inspect_task_rougeL'
    ]
    label_names = [
        'FSL (LLM)',
        'Softprompt (SLM)',
        'Groundtruth (LLM)',
        'Translator (LLM)',
        'InSPEcT (LLM)'
    ]


    scores = [out_df[col_name] for col_name in col_names]

    # -----------------------------
    # Paired t-test (recommended)
    # -----------------------------

    # # t_stat, p_value = stats.ttest_rel(llm_scores, control_scores, nan_policy='omit')

    # # print("\nPaired t-test (Mapper verbalization on LLM vs. Control on LLM):")
    # # print(f"Mean {col_name1}, {llm_scores.mean()}")
    # # print(f"Mean {col_name2}, {control_scores.mean()}")
    # # print(f"t-statistic: {t_stat:.4f}")
    # # print(f"p-value: {p_value:.6f}")


    # -----------------------------
    # Box plot
    # -----------------------------
    plt.figure(figsize=(8, 4))
    plt.boxplot(
        scores,
        tick_labels=label_names
    )

    plt.title("Comparison of average task performance (ROUGE-L)", fontsize=11)
    plt.ylabel("Average Task ROUGE-L Score")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plot_path = args.output + "_boxplot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nSaved box plot to: {plot_path}")

