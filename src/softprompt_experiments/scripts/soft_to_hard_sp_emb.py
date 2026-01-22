import torch
import argparse
import random
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm

from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.utils import (
    get_train_test_from_softprompt_embeds, 
    train_softprompt_from_embeds,
    eval_softprompt,
    log_json
)

from itertools import chain

import torch.nn.functional as F
import torch.nn as nn

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_datasetv2_large")
    parser.add_argument("--num_samples_to_eval", type=int, default=25)
    parser.add_argument("--num_tokens", type=int, default=8)
    parser.add_argument("--verbose", type=bool, default=False)

    args, _ = parser.parse_known_args(args_list)

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    # MODEL_NAME = "meta-llama/Llama-3.1-8B"
    SAVE_DIR = args.save_directory
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_SAMPLES_TO_EVAL = args.num_samples_to_eval

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

    # loads in a dataset of trained softprompts
    train_dataset, test_dataset, train_loader, test_loader = get_train_test_from_softprompt_embeds(
        model,
        word_embeddings,
        tokenizer,
        dataset_dirs,
        BATCH_SIZE,
        0.8,
        False
    )

    # -----------------------
    # LOAD BASE MODEL
    # -----------------------
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # freeze base model

    # Suffix to mark end of input
    suffix = " Input: x=1, y=2, z=3\nOutput: "

    suffix_ids = tokenizer(
        suffix,
        add_special_tokens=False,
        return_tensors='pt'
    )['input_ids'].to(model.device)
    SUFFIX_LEN = suffix_ids.shape[1]
    suffix_emb = word_embeddings(suffix_ids).to(model.dtype).detach()

    softprompt = SoftPrompt(
        model=model,
        init="1 2 3 4 ",
        tokenizer=tokenizer,
        word_embeddings=word_embeddings,
        num_tokens=16
    )

    # projector = torch.nn.Linear(
    #     word_embeddings.weight.shape[1], word_embeddings.weight.shape[1],
    #     dtype=dtype,
    #     device=device
    # )
    class Projector(nn.Module):
        def __init__(self, dtype, device):
            super().__init__()

            self.bias = nn.Parameter(
                torch.zeros((1, 8, 4096), dtype=dtype, device=device)
            )

        def forward(self, x):
            return x + self.bias
    
    projector = Projector(dtype, device)

    optimizer = torch.optim.AdamW(chain(softprompt.parameters(), projector.parameters()), lr=LR, weight_decay=0.1)

    # -----------------------
    # TRAINING LOOP (with test loss logging)
    # -----------------------
    tr_losses = []
    te_losses = []

    for epoch in tqdm(range(EPOCHS)):
        # -------------------
        # Train
        # -------------------
        model.train()     # only LoRA params train; base is frozen
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            softembeds, hardprompt_embeds, tokenized_hardprompt = [b.to(device) for b in batch]
            batched_suffixemb = suffix_emb.expand(softembeds.size(0), -1, -1)
            batched_softprompt = softprompt.forward().expand(softembeds.size(0), -1, -1)
            full_inputs = torch.cat([
                projector(softembeds.to(model.dtype)),
                batched_suffixemb,
                batched_softprompt,
                hardprompt_embeds.to(model.dtype)
            ], dim=1)

            labels = torch.cat([
                torch.full((batched_softprompt.shape[0], batched_softprompt.shape[1]), -100).to(device),
                torch.full((softembeds.shape[0], softembeds.shape[1]), -100).to(device),
                torch.full((batched_suffixemb.shape[0], batched_suffixemb.shape[1]), -100).to(device),
                tokenized_hardprompt
            ], dim=1)

            outputs = model(inputs_embeds=full_inputs, labels=labels)

            # outputs = model(inputs_embeds=input_embeds, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        tr_loss = total_loss / len(train_loader)
        tr_losses.append(tr_loss)

        # -------------------
        # Test
        # -------------------
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                softembeds, hardprompt_embeds, tokenized_hardprompt = [b.to(device) for b in batch]
                
                batched_suffixemb = suffix_emb.expand(softembeds.size(0), -1, -1)
                batched_softprompt = softprompt.forward().expand(softembeds.size(0), -1, -1)

                full_inputs = torch.cat([
                    projector(softembeds.to(model.dtype)),
                    batched_suffixemb,
                    batched_softprompt,
                    hardprompt_embeds.to(model.dtype)
                ], dim=1)

                labels = torch.cat([
                    torch.full((batched_softprompt.shape[0], batched_softprompt.shape[1]), -100).to(device),
                    torch.full((softembeds.shape[0], softembeds.shape[1]), -100).to(device),
                    torch.full((batched_suffixemb.shape[0], batched_suffixemb.shape[1]), -100).to(device),
                    tokenized_hardprompt
                ], dim=1)

                outputs = model(inputs_embeds=full_inputs, labels=labels)
                # outputs = model(inputs_embeds=input_embeds, labels=labels)

                total_test_loss += outputs.loss.item()

        te_loss = total_test_loss / len(test_loader)
        te_losses.append(te_loss)

        print(f"Epoch {epoch} | Train Loss: {tr_loss:.4f} | Test Loss: {te_loss:.4f}")

    # -----------------------
    # SAMPLE PREDICTIONS
    # -----------------------
    model.eval()
    with torch.no_grad():
        # --- TRAIN SET ---
        train_samples = random.sample(
            list(train_dataset), 
            min(NUM_SAMPLES_TO_EVAL, len(train_dataset))
        )
        for softembeds, hardprompt_embeds, tokenized_hardprompt in train_samples:
            full_inputs = torch.cat([
                projector(softembeds.unsqueeze(0).to(model.dtype)),
                suffix_emb.to(model.dtype),
                softprompt.forward(),
            ], dim=1)
            
            max_new_tokens = len(tokenized_hardprompt)

            pred_ids = model.generate(inputs_embeds=full_inputs, max_new_tokens=max_new_tokens)
            pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            hardprompt = tokenizer.decode(tokenized_hardprompt, skip_special_tokens=True)

            print(f"Prediction (train): {pred_text}")
            print(f"hardprompt (train): {hardprompt}\n")
        # --- TEST SET ---
        test_samples = random.sample(
            list(test_dataset),
            min(NUM_SAMPLES_TO_EVAL, len(test_dataset))
        )
        for softembeds, hardprompt_embeds, tokenized_hardprompt in test_samples:
            full_inputs = torch.cat([
                projector(softembeds.unsqueeze(0).to(model.dtype)),
                suffix_emb.to(model.dtype),
                softprompt.forward(),
            ], dim=1)

            max_new_tokens = len(tokenized_hardprompt)

            pred_ids = model.generate(inputs_embeds=full_inputs, max_new_tokens=max_new_tokens)
            pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            hardprompt = tokenizer.decode(tokenized_hardprompt, skip_special_tokens=True)

            print(f"Prediction (test): {pred_text}")
            print(f"hardprompt (test): {hardprompt}\n")

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









