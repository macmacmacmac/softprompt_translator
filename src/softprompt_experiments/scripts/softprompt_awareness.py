import torch
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm

from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.models.squishyprompt import SquishyPrompt
from softprompt_experiments.utils import (
    get_train_test_from_tokenized, 
    log_json
)

import json

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_physics2")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--show_target", type=bool, default=False)
    parser.add_argument("--no_auto_split",dest="auto_split",action="store_false")
    parser.set_defaults(auto_split=True)

    args, _ = parser.parse_known_args(args_list)

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    SAVE_DIR = args.save_directory
    BATCH_SIZE = args.batch_size
    AUTO_SPLIT = args.auto_split

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype
    ).to(device)
    model.eval()
    word_embeddings = model.get_input_embeddings()

    # Get dataset sub directories
    dataset_dirs = []
    for entry in os.scandir(SAVE_DIR):
        if entry.is_dir():  # Check if the entry is a directory
            if "dataset_" in entry.name:
                dataset_dirs.append(entry.path)

    num_datasets = len(dataset_dirs)
    if num_datasets > 0:
        print(f"\nFound ({num_datasets}) datasets in directory")
    else:
        raise ValueError("path to directory has no datasets")

    for dataset_dir in tqdm(dataset_dirs):
        train_dataset, test_dataset, train_loader, test_loader = get_train_test_from_tokenized(
            dataset_dir,
            BATCH_SIZE,
            train_portion = 0.8,
            auto_split=AUTO_SPLIT
        )
        with open(os.path.join(dataset_dir,'softprompt_performance.json')) as f:
            soft_perf = json.load(f)

        entropy = soft_perf['entropy']

        pearson_r = None
        accuracy = None
        if "pearson_r" in soft_perf:
            pearson_r = soft_perf['outputs']['pearson_r']
        elif "accuracy" in soft_perf:
            accuracy = soft_perf['outputs']['accuracy']

        if AUTO_SPLIT:
            hardprompt = torch.load(
                os.path.join(dataset_dir,'dataset.pt'),
                weights_only=False
            )['hardprompt']
        else:
            hardprompt = torch.load(
                os.path.join(dataset_dir,'train_dataset.pt'),
                weights_only=False
            )['hardprompt']
        softprompt = SoftPrompt(
            model=model, 
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            path_to_model=os.path.join(dataset_dir,'softprompt.pt')
        )

        init_embeds = softprompt.initial_embeddings

        results = {}
        results['hardprompt'] = hardprompt
        print(f"\n--------------------------Actual hardprompt: {hardprompt}--------------------------\n")
        print(f"|=== Entropy: {entropy}")
        if pearson_r is not None:
            print(f"|=== Pearson R: {pearson_r}")
        if accuracy is not None:
            print(f"|=== Accuracy: {accuracy}")

        random_idxs = torch.randint(0, len(test_dataset), (args.num_samples,))

        soft_generations = ""
        base_generations = ""
        gen_prompt = "First, I should"
        gen_ids = tokenizer(gen_prompt,return_tensors="pt").input_ids.to(device)
        gen_embed = word_embeddings(gen_ids).to(dtype=dtype)
        for idx in random_idxs:
            labels = test_dataset[idx][1].to(model.device)
            full_ids = test_dataset[idx][0].to(model.device)
            mask = (labels==-100).to(model.device)
            antimask = (labels!=-100).to(model.device)

            tokenized_text = full_ids[mask].to(model.device)
            input_text = tokenizer.decode(tokenized_text, skip_special_tokens=True)
            input_embed = word_embeddings(tokenized_text).unsqueeze(0)

            soft_gen = softprompt.generate_from_embeds(embeds=input_embed, max_new_tokens=75, suffix_str=gen_prompt)[0]
            
            if args.show_target:
                tokenized_target = full_ids[antimask].to(model.device)
                target_text = tokenizer.decode(tokenized_target, skip_special_tokens=True)
                print(f"Actual: {target_text}")
            print(f"<soft generation start>{input_text}{gen_prompt}{soft_gen}<soft generation end>\n")
            soft_generations += (input_text + gen_prompt + soft_gen + "\n")

            base_embs = torch.cat([input_embed, gen_embed], dim=1)
            attention_mask = torch.ones(base_embs.size()[:-1], device=input_embed.device, dtype=torch.long)
            base_gen_ids = model.generate(
                inputs_embeds=base_embs,
                attention_mask=attention_mask,
                max_new_tokens=75,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            base_gen = tokenizer.decode(base_gen_ids[0], skip_special_tokens=True)
            print(f"<base generation start>{gen_prompt}{base_gen}<base generation end>\n")
            base_generations += (input_text + gen_prompt + base_gen + "\n")
        
        # log awareness rate to results
        results['awareness_rate'] = awareness_rate
        for key in results:
            soft_perf[key] = results[key]
        log_json(os.path.join(dataset_dir, "softprompt_performance.json"), soft_perf)


    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









