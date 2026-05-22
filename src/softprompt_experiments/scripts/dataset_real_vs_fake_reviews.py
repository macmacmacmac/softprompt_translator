import torch
import copy
import numpy as np
import argparse
import os
import pandas as pd
from transformers import (
    AutoTokenizer,
)
from tqdm.auto import tqdm
from datasets import load_dataset

from softprompt_experiments.utils import tokenize_and_save, batched_tokenize_and_save


def run(args_list):
    exp_name = os.path.basename(__file__)

    print(
        "=" * 100,
        "\n",
        f"\t\t\t\tRunning script: {exp_name}",
        "\n",
        "=" * 100,
        "\n",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_directory",
        type=str,
        default="./datasets/deceptive_opinion_spam",
    )

    args, _ = parser.parse_known_args(args_list)

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    SAVE_DIR = args.save_directory

    os.makedirs(SAVE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------
    # Load deceptive opinion spam dataset
    # ------------------------------------------------------------
    #
    # Kaggle source:
    # https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus
    #
    # Expected columns:
    #   - text
    #   - deceptive
    #
    # deceptive values:
    #   truthful / deceptive
    #
    # ------------------------------------------------------------

    csv_path = os.path.join(SAVE_DIR, "deceptive-opinion.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find dataset csv at: {csv_path}\n"
            "Download the kaggle dataset and place the csv there."
        )

    df = pd.read_csv(csv_path)

    required_columns = ["text", "deceptive"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}'. "
                f"Found columns: {list(df.columns)}"
            )

    # remove NaNs
    df = df.dropna(subset=["text", "deceptive"])

    # convert to strings
    df["text"] = df["text"].astype(str)
    df["deceptive"] = df["deceptive"].astype(str)

    prefix = "\nSample text:\n\""
    suffix = "...\"\nAnswer:\n"

    # tokenize + save
    tokenized = batched_tokenize_and_save(
        df["text"].tolist(),
        prefix,
        suffix, 
        df["deceptive"].tolist(), 
        SAVE_DIR, 
        "real vs fake reviews", 
        tokenizer, 
        input_max_length=128, 
        target_max_length=4
    )
    print(tokenizer.decode(tokenized['tokenized_samples']['input_ids'][0]))


    print(
        "\n",
        "=" * 100,
        "\n",
        f"\t\t\t\tCompleted script: {exp_name}",
        "\n",
        "=" * 100,
    )

    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     raise ValueError("API key not found.")

    # client = OpenAI(api_key=api_key)
    # def get_llm_prediction(client, model, prompt):
    #     response = client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant. Follow the task exactly."},
    #             {"role": "user", "content": prompt},
    #         ],
    #         temperature=0.0,  # greedy
    #     )

    # return response.choices[0].message.content.strip()
    # pred = get_llm_prediction(client, args.model, full_prompt)








