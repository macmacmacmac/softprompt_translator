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


    class TeachersDataset:
        def __init__(self, dataset_folder_path):
            # Teacher keys parser utility function
            def get_teacher_from_key(key):
                # example: 2.09.1266_03.26.10_DV15
                idx = key.index('_')
                key = key[0:idx]
                idx = key.rindex('.')
                return int(key[idx+1:])

            # Loads in the dataset
            with open(os.path.join(dataset_folder_path, 'transcripts_NCRECE.pkl'), 'rb') as file:
                transcripts = pickle.load(file)
            targets = np.load(os.path.join(dataset_folder_path, "y_transcript_NCRECE.npy"))
            keys = np.load(os.path.join(dataset_folder_path, "keys_transcript_NCRECE.npy"))
            keys = [get_teacher_from_key(key) for key in keys]

            self.data = pd.DataFrame({
                "transcripts" : transcripts,
                "targets" : targets,
                "keys" : keys,
            })

        def get_keys_from_path(self, path_to_keys):
            with open(path_to_keys, 'r') as file:
                keys = file.read().replace(" ", "").split(",")
                keys = [int(key) for key in keys]
            return keys

        def get_dataset_by_keys(self, keys):
            bool_idx = [(key in keys) for key in self.data['keys']]
            samples = self.data.iloc[bool_idx].copy()
            
            return samples

    # pipeline
    save_dir = os.path.join(SAVE_DIR, "dataset_0")

    dataset_path = "./datasets/CLASS_dataset"
    dataset = TeachersDataset(dataset_path)
    print("Dataset loaded"+ "="*16 +"\n\n")

    train_keys = dataset.get_keys_from_path(os.path.join(dataset_path,"train_keys.txt"))
    test_keys = dataset.get_keys_from_path(os.path.join(dataset_path,"test_keys.txt"))

    traindf = dataset.get_dataset_by_keys(train_keys)
    testdf = dataset.get_dataset_by_keys(test_keys)
    print("Dataset splitted"+ "="*16+"\n\n")

    # if normalize:
    #     mu, std = np.mean(traindf['targets']), np.std(traindf['targets'])
    #     traindf['targets'] = (np.array(traindf['targets'])-mu)/std
    #     normalize = (mu, std)


    df = pd.read_csv("./datasets/human_or_ai_dataset/human_or_ai_dataset.csv")
    
    # input_sentences = [f"\nSample text: {input_sent}\nAnswer: " for input_sent in df.text.tolist()]
    prefix = "\nClass recording transcript:\n\""
    suffix = "...\"\nAnswer:\n"
    target_sentences = [('AI generated' if target==1 else 'Human written') for target in df.generated.tolist()]

    # print(input_sentences[:3])
    # print(target_sentences[:3])

    train_tokenized = batched_tokenize_and_save(
        df.text.tolist(),
        prefix,
        suffix, 
        target_sentences, 
        save_dir, 
        "human or ai", 
        tokenizer, 
        input_max_length=128, 
        target_max_length=4,
        file_name='train_dataset.pt'
    )

    test_tokenized = batched_tokenize_and_save(
        df.text.tolist(),
        prefix,
        suffix, 
        target_sentences, 
        save_dir, 
        "human or ai", 
        tokenizer, 
        input_max_length=128, 
        target_max_length=4,
        file_name='test_dataset.pt'
    )


    print(tokenizer.decode(tokenized['tokenized_samples']['input_ids'][0]))

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









