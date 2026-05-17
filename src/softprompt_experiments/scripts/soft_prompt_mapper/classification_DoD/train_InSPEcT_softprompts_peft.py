import os
import argparse
from datasets import load_dataset
from peft import PromptTuningInit, PromptTuningConfig, TaskType, get_peft_model
from transformers import default_data_collator, get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd


# ┌───────────────────────────────────────────────┐
# │                GLOBAL VARIABLES               │
# └───────────────────────────────────────────────┘
# Default max length for each data example
MAX_LENGTH = 80 

# Common Text Label
COMMON_TEXT_LABEL = "text_label"


# Inspect Dataset specific configs
INSPECT_DATASET_CONFIGS = {
    "stanfordnlp/sst2": {
        "epochs": 8,
        # "lr": 8e-4,
        "lr": 8e-3,
        "batch_size": 8,
        "eval_split": "validation",
        "text_column": "sentence",
        "label_column": "label",
        "classes": ["negative", "positive"]
    },
    "SetFit/sst5": {
        "epochs": 12,
        "lr": 6e-3,
        "batch_size": 8,
        "eval_split": "validation",
        "text_column": "text",
        "label_column": "label",
        "classes": ["terrible", "bad", "neutral", "good", "great"]
    },
    "fancyzhx/ag_news": {
        "epochs": 8,
        "lr": 8e-3,
        "batch_size": 8,
        "eval_split": "test",
        "text_column": "text",
        "label_column": "label",
        "classes": ["world", "sports", "business", "technology"]
    },
    "SetFit/subj": {
        "epochs": 8,
        "lr": 8e-3,
        "batch_size": 8,
        "eval_split": "test",
        "text_column": "text",
        "label_column": "label",
        "classes": ["objective", "subjective"]
    },
    "SetFit/TREC-QC": {
        "epochs": 20,
        # "lr": 8e-4,
        "lr": 8e-3,
        "batch_size": 8,
        "eval_split": "test",
        "text_column": "text",
        "label_column": "label_coarse",
        "classes": ["description", "entity", "abbreviation", "human", "number", "location"]
    }
}

# ┌───────────────────────────────────────────────┐
# │                 HELPER METHODS                │
# └───────────────────────────────────────────────┘
def load_inspect_dataset(dataset_name,
                         label_column,
                         eval_split,
                         classes,
                         max_training_examples = 50_000,
                         max_eval_examples = 2000):
    
    # Load the Dataset from HF
    dataset = load_dataset(dataset_name)
    
    # Limit Training Data (if applicable)
    dataset['train'] = dataset['train'].select(range(max_training_examples)) if \
        len(dataset['train']) > max_training_examples else dataset['train']
    
    # Limit Eval Data (if applicable)
    dataset[eval_split] = dataset[eval_split].shuffle().select(range(max_eval_examples)) if \
        len(dataset[eval_split]) > max_eval_examples else dataset[eval_split]

    # Create a new column for "text_label" using the classes
    return dataset.map(
        lambda x: {COMMON_TEXT_LABEL: [classes[label] for label in x[label_column]]},
        batched=True,
        num_proc=1,
    )


def tokenize_dataset(examples, 
                     tokenizer, 
                     text_column,
                     max_tokenized_label_len,
                     max_length=MAX_LENGTH):
    
    # Get the batch_size (different from training batch_size)
    batch_size = len(examples[text_column])

    # Construct input text for each example
    # TODO: This is different from the structure of how we trained DoD soft prompts
    inputs = [f"{text_column} : {x.strip()} Label : " for x in examples[text_column]]
    # inputs = [f"Sentence: {x.strip()} Label:" for x in examples[text_column]]

    # Construct the target text for each example
    # TODO: This is minor difference in how we trained DoD soft prompts in terms of extra spaces
    targets = [str(x) for x in examples[COMMON_TEXT_LABEL]]
    # targets = [f" {x}" for x in examples[COMMON_TEXT_LABEL]]

    # Tokenize the input text and labels.
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)

    # pad the labels with the tokenizer's pad_token_id.
    # Concatenate the input text and labels into the model_inputs.
    # Create a separate attention mask for labels and model_inputs.
    for i in range(batch_size):
        end_padding_length = max_tokenized_label_len - len(labels["input_ids"][i])
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        input_suffix = label_input_ids
        model_inputs["input_ids"][i] = sample_input_ids + input_suffix + \
            [tokenizer.pad_token_id] * end_padding_length
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids + \
            [-100] * end_padding_length
        model_inputs["attention_mask"][i] = [1] * (len(model_inputs["input_ids"][i]) - end_padding_length) + \
            [0] * end_padding_length

    # pad the input ids, labels and attention_mask to the max_length 
    # and convert them to PyTorch tensors
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * \
            (max_length - len(sample_input_ids)) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + \
            model_inputs["attention_mask"][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids

        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_tokenized_dataloaders(
        dataset,
        tokenizer,
        text_column,
        eval_split,
        batch_size,
        classes
    ):

    max_tokenized_label_len = max(len(tokenizer.encode(c)) for c in classes)

    # Tokenize Training Dataset
    train_dataset = dataset["train"].map(
        lambda d: tokenize_dataset(d, 
                                   tokenizer, 
                                   text_column, 
                                   max_tokenized_label_len),
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )

    # Tokenize Validation Dataset
    val_dataset = dataset[eval_split].map(
        lambda d: tokenize_dataset(d, 
                                   tokenizer, 
                                   text_column, 
                                   max_tokenized_label_len),
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on eval dataset",
    )

    # Create DataLoaders for Training and Validation datasets
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=default_data_collator, 
        batch_size=batch_size
    )

    val_dataloader = DataLoader(
        val_dataset, 
        collate_fn=default_data_collator, 
        batch_size=batch_size
    )
    
    return train_dataloader, val_dataloader

def construct_soft_prompt_save_dir_path(dataset_name, save_dir, num_tokens):
    dataset_name = dataset_name.split('/')[1]
    return f"{save_dir}/{dataset_name}_{num_tokens}tokens"


def train_soft_prompts(peft_model, 
                       train_dataloader, 
                       eval_dataloader, 
                       num_tokens, 
                       soft_prompt_save_dir,
                       num_epochs, 
                       lr, 
                       batch_size,
                       device
    ):
    
    # Init Optimizer
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)

    # Init Linear Scheduler for Learning Rate
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # Loop num_epochs times
    for epoch in range(num_epochs):

        # Set the model on training mode
        peft_model.train()

        # Calculate total loss
        total_loss = 0
        train_correct = 0

        # For each batch in the dataloader
        for batch in tqdm(train_dataloader):

            # Reset Gradients
            optimizer.zero_grad()

            # Move all items of data batch to the device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward Pass Thru the model
            outputs = peft_model(**batch)

            # Extract Loss and accumulate it to the total loss
            loss = outputs.loss
            total_loss += loss.detach()

            # Calculate batch accuracy
            with torch.no_grad():
                top_tokens = torch.argmax(outputs.logits, dim=-1)[:, num_tokens-1:-1]
                is_prediction_correct = (
                    (batch['labels'] == -100) |
                    (batch['labels'] == top_tokens)
                ).all(dim=1)
                train_correct += sum(is_prediction_correct).item()

            # Compute Gradients and Do backpropagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        # Eval Model at the end of this epoch in terms of loss and correctness
        eval_loss, eval_correct = eval_soft_prompts(peft_model, eval_dataloader, num_tokens, device)

        # Calculate Val and Train accuracy
        val_accuracy = eval_correct / len(eval_dataloader.dataset)
        train_accuracy = train_correct / len(train_dataloader.dataset)

        # Calculate Avg Val and Training Loss    
        avg_val_loss = eval_loss / len(eval_dataloader)
        avg_train_loss = total_loss / len(train_dataloader)

        tqdm.write(f"\nEpoch {epoch + 1} Summary:")
        tqdm.write(f"Train -> Loss: {avg_train_loss: .4f} | Accuracy: {train_accuracy * 100: .2f}%")
        tqdm.write(f"Val   -> Loss: {avg_val_loss: .4f} | Accuracy: {val_accuracy * 100: .2f}%")

    # Save the raw trained soft prompt
    os.makedirs(soft_prompt_save_dir, exist_ok=True)
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad][0]
    print(f"Extracted soft prompts from peft model dtype: {trainable_params.dtype}")
    torch.save(trainable_params, os.path.join(soft_prompt_save_dir, "softprompt.pt"))
    tqdm.write(f"\nTraining complete! Soft prompt saved to {soft_prompt_save_dir}/softprompt.pt")

    # Save the peft model
    # peft_model.save_pretrained(soft_prompt_save_dir)

    return {
        "train_accuracy": round(train_accuracy * 100, 4),
        "val_accuracy": round(val_accuracy * 100, 4),
        "avg_train_loss": avg_train_loss.item() if torch.is_tensor(avg_train_loss) else avg_train_loss,
        "avg_val_loss": avg_val_loss.item() if torch.is_tensor(avg_val_loss) else avg_val_loss
    }

# Performs Evaluation Based on Exact Token Match
def eval_soft_prompts(model, 
                      eval_dataloader, 
                      num_tokens,
                      device):
    
    # Set model to evaluation mode
    model.eval()                          
    eval_loss = 0
    eval_correct = 0
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU
        with torch.no_grad():             # Disable gradient computation
            outputs = model(**batch)      # Forward pass with input_ids, attention_mask, labels
        loss = outputs.loss               # Get cross-entropy loss
        eval_loss += loss.detach().float()
        
        # Get the most likely token at each position
        # [:,num_tokens-1:-1] skips the soft prompt tokens and last token
        top_tokens = torch.argmax(outputs.logits, dim=-1)[:, num_tokens-1:-1]
        
        # Check if prediction matches label (ignoring -100 padding positions)
        is_prediction_correct = (
            (batch['labels'] == -100) |   # Ignore padding tokens (always "correct")
            (batch['labels'] == top_tokens)  # Or actual match
        ).all(dim=1)                      # All tokens in sequence must match
        eval_correct += sum(is_prediction_correct).item()
    
    return eval_loss, eval_correct


# ┌───────────────────────────────────────────────┐
# │                 DRIVER CODE                   │
# └───────────────────────────────────────────────┘
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./inspect_soft_prompts_peft_random_16bit")
    parser.add_argument("--num_tokens", type=int, default=20)
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    # MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    SAVE_DIR = args.save_dir
    NUM_TOKENS = args.num_tokens

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Init Prompt Tuning Config
    prompt_tuning_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM, # Random Weight Init for Prompt Tuning Params
        # prompt_tuning_init=PromptTuningInit.SAMPLE_VOCAB, # Sampling from the Model's Vocab for Init
        num_virtual_tokens=NUM_TOKENS,
        tokenizer_name_or_path=MODEL_NAME
    )

    all_stats = []

    # For each InSPEcT dataset in the global config
    for dataset_name in INSPECT_DATASET_CONFIGS:

        print(f"Training Soft Prompts for dataset {dataset_name}")

        # Extract useful configs about the Dataset from the global config dict
        dataset_config = INSPECT_DATASET_CONFIGS[dataset_name]
        text_column = dataset_config["text_column"]
        label_column = dataset_config["label_column"]
        classes = dataset_config["classes"]
        eval_split = dataset_config["eval_split"]
        batch_size = dataset_config["batch_size"]
        epochs = dataset_config["epochs"]
        lr = dataset_config["lr"]


        # Load InSPEcT Dataset
        dataset = load_inspect_dataset(
            dataset_name = dataset_name,
            label_column = label_column,
            classes = classes,
            eval_split = eval_split
        )

        # Fetch Tokenized Trainig and Validation Dataloaders
        train_dataloader, val_dataloader = prepare_tokenized_dataloaders(
            dataset,
            tokenizer,
            text_column,
            eval_split,
            batch_size,
            classes
        )

        # Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)

        # Prepare Soft Prompt Model
        soft_prompt_model = get_peft_model(base_model, prompt_tuning_config).to(DTYPE)
        trainable_params = [p for p in soft_prompt_model.parameters() if p.requires_grad][0]
        print(f"Extracted soft prompts (before training) from peft model dtype: {trainable_params.dtype}")


        # Prepare Save Dir for the Trained Tokens
        soft_prompt_save_dir = construct_soft_prompt_save_dir_path(dataset_name, SAVE_DIR, NUM_TOKENS)

        # Train the Soft Prompts
        stats = train_soft_prompts(
            peft_model=soft_prompt_model,
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
            num_tokens=NUM_TOKENS,
            soft_prompt_save_dir=soft_prompt_save_dir,
            num_epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=DEVICE
        )
        
        stats["dataset_id"] = dataset_name
        all_stats.append(stats)

        # Clean up allocations that might be memory intensive
        del soft_prompt_model
        del base_model
        torch.cuda.empty_cache()

    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        # Reorder columns to ensure exact CSV format
        stats_df = stats_df[["dataset_id", "train_accuracy", "val_accuracy", "avg_train_loss", "avg_val_loss"]]
        
        stats_csv_path = os.path.join(SAVE_DIR, "accuracy_stats.csv")
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"\nSaved accuracy stats to {stats_csv_path}")



