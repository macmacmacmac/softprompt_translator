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
import numpy as np

ROUGE_METRIC = evaluate.load("rouge")


# -----------------------------
# Driver
# -----------------------------
def build_visual(output_path):
    out_df = pd.read_json(output_path+".json")
    print(out_df.head())
    # print(f"Saving results to {args.output}.json...")
    # with open(args.output+".json", "w") as f:
    #     json.dump(data, f, indent=2)

    # col_name0 = 'fs_task_rougeL'
    col_names = [
        'soft_task_rougeL',
        'fs_task_rougeL',
        'gt_task_rougeL',
        'fsr_task_rougeL',
        # 'mapper10x_logprob_z_W_n20_task_rougeL',
        # 'mapper10x_logprob_z_W_z_G_task_rougeL',
        'dpo_10x_round1_task_rougeL',
        'mapper10x_logprob_z_W_task_rougeL',
        'mapper10x_task_rougeL',
        'mapper_task_rougeL',
        'inspect_task_rougeL'
    ]
    label_names = [
        'Softprompt',
        'FS',
        'Groundtruth',
        'FS \nw/ Prompt ',
        # 'Tr 10x \nn=20 z_W logprob',
        # 'Translator 10x \nz_G logprob',
        'Translator 10x \nDPO',
        'Translator 10x \nz_W logprob',
        'Translator 10x',
        'Translator 1x',
        'InSPEcT'
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
    plt.figure(figsize=(14, 6))

    plt.boxplot(
        scores,
        tick_labels=label_names,
    )
    plt.xticks(fontsize=16)

    # Plot and label means
    means = [s.mean() for s in scores]
    x = np.arange(1, len(scores) + 1)

    plt.scatter(x, means, color="red", marker="D", s=35, zorder=3, label="Mean")

    for xi, mean in zip(x, means):
        plt.text(
            xi,
            mean,
            f"{mean:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )
    plt.title("Average task ROUGE-L's on validation split of GeneralDoD", fontsize=16)
    plt.ylabel("Average Task ROUGE-L", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()

    plot_path = output_path + "_boxplot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nSaved box plot to: {plot_path}")

def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print("=" * 100)
    print(f"\t\t\tRunning script: {exp_name}")
    print("=" * 100)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./shared/verbalizations/master_verbalizations_v3")
    args, _ = parser.parse_known_args(args_list)

    build_visual(output_path=args.output)

