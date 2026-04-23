import os
import argparse
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from softprompt_experiments.models.softprompt import SoftPrompt
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

# Inspect Dataset specific configs
INSPECT_DATASET_CONFIGS = {
    "stanfordnlp/sst2": {
        "epochs": 8,
        "lr": 8e-4,
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
        "lr": 8e-4,
        "batch_size": 8,
        "eval_split": "test",
        "text_column": "text",
        "label_column": "label_coarse",
        "classes": ["description", "entity", "abbreviation", "human", "number", "location"]
    }
}


class InSPEcTClassificationDataset(Dataset):
    def __init__(self, dataset_id, text_column, label_column, dataset_classes, split="train", max_examples = 50_000):
        """
        Fetches all sentences and their target keywords for a specific dataset_id from HuggingFace.
        """
        # Init Member Variables
        self.dataset_id = dataset_id
        self.split = split
        self.inputs = []
        self.targets = []

        # Load the Dataset from HF
        dataset = load_dataset(dataset_id)

        # Limit Data (if applicable)
        dataset[split] = dataset[split].select(range(max_examples)) \
            if len(dataset[split]) > max_examples else dataset[split]
        
        # Create a new column for input_text
        dataset[split] = dataset[split].map(
            lambda batch: {"input_text": [f"Sentence: {text} Label:" for text in batch[text_column]]},
            batched = True,
            num_proc = 1
        )

        # Create a new column for target_text
        dataset[split] = dataset[split].map(
            lambda batch: {"target_text": [f" {dataset_classes[label]}" for label in batch[label_column]]},
            batched = True,
            num_proc = 1
        )

        # Fetch the inputs and targets inputs
        self.inputs = list(dataset[split]["input_text"])
        self.targets = list(dataset[split]["target_text"])
    
        if len(self.inputs) == 0:
            raise ValueError(f"No data found for dataset_id {self.dataset_id} (split: {self.split})")


    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, idx):
        # Return the raw strings which we will tokenize later
        return self.inputs[idx], self.targets[idx]
    


class CausalLMBatchCollator:
    def __init__(self, tokenizer, soft_prompt_length=20):
        self.tokenizer = tokenizer
        self.soft_prompt_length = soft_prompt_length

    def __call__(self, batch):
        inputs, targets = zip(*batch)
        
        # Combine input and target into the full sequence the model needs to see
        full_texts = [f"{inp}{tgt}" for inp, tgt in zip(inputs, targets)]
        
        # Tokenize the full sequences (this gives us input_ids and attention_mask)
        tokenized = self.tokenizer(
            full_texts, 
            padding=True, 
            truncation=True,
            max_length=64, 
            return_tensors="pt",
            add_special_tokens=True
        )
        
        input_ids = tokenized["input_ids"]                                                                  # (batch_size, seq_len)
        attention_mask = tokenized["attention_mask"]
        
        # Create the labels tensor (start by copying input_ids)
        labels = input_ids.clone()                                                                          # (batch_size, seq_len)
        
        # Mask out the input text and the padding tokens with -100
        for i, (inp, tgt) in enumerate(zip(inputs, targets)):

            # Tokenize just the input text to find out how long it is
            inp_len = len(self.tokenizer(inp, add_special_tokens=True)["input_ids"])
            
            # Mask the input portion so loss is not calculated on it
            labels[i, :inp_len] = -100
            
            # Mask any padding tokens added to the end of the sequence
            labels[i, attention_mask[i] == 0] = -100

        # Account for the Soft Prompt!
        # Because we will prepend `soft_prompt_length` virtual embeddings to the front
        # of the inputs in the training loop, we must pad the front of our labels with -100s
        # so the matrix dimensions line up perfectly for the loss function.
        batch_size = labels.size(0)
        soft_prompt_labels = torch.full((batch_size, self.soft_prompt_length), -100, dtype=torch.long)      # (batch_size, soft_prompt_len)
        labels = torch.cat([soft_prompt_labels, labels], dim=1)                                             # (batch_size, soft_prompt_len + seq_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }



# Driver Code
def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="./inspect_soft_prompts")
    parser.add_argument("--use_custom_train_configs", action="store_true", help="Use Dataset specific training configs")
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)
    

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    DEFAULT_LR = args.lr
    DEFAULT_EPOCHS = args.epochs
    NUM_TOKENS = args.num_tokens
    DEFAULT_BATCH_SIZE = args.batch_size
    SAVE_DIR = args.save_dir
    PATIENCE = args.patience
    MIN_DELTA = args.min_delta
    USE_CUSTOM_TRAIN_CONFIGS = args.use_custom_train_configs


    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Llama doesn't have a default pad token, so we map it to EOS
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    # Load Llama Model and set it in eval mode
    llama_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        device_map = DEVICE
    )
    llama_model.eval()

    # Freeze the entire Llama model since we only want to train the Soft prompt
    for param in llama_model.parameters():
        param.requires_grad = False

    # Get the Llama Model's Word Embedding Mappings
    llama_word_embeddings = llama_model.get_input_embeddings()

    # Init Collator
    collator = CausalLMBatchCollator(
        tokenizer = llama_tokenizer,
        soft_prompt_length = NUM_TOKENS
    )

    # Determine file path for accuracy stats
    ACCURACY_STATS_FILE_PATH = f"{SAVE_DIR}/accuracy_stats.csv"

    # Preload existing accuracy_stats, if it exists already
    if os.path.isfile(ACCURACY_STATS_FILE_PATH):
        df = pd.read_csv(ACCURACY_STATS_FILE_PATH)
        training_stats = df.to_dict(orient='list')

    else:
        # Init a dict to save training stats
        training_stats = {
            'dataset_id': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'avg_train_loss': [],
            'avg_val_loss': []
        }


    # Loop over all dataset ids
    for dataset_id in INSPECT_DATASET_CONFIGS:

        # Init save dir
        dataset_name = dataset_id.split('/')[1]
        save_dir = f"{SAVE_DIR}/{dataset_name}_{NUM_TOKENS}tokens"

        # Retrieve the configs for current dataset_id
        configs = INSPECT_DATASET_CONFIGS[dataset_id]

        # Training Configs
        epochs = configs["epochs"] if USE_CUSTOM_TRAIN_CONFIGS else DEFAULT_EPOCHS
        lr = configs["lr"] if USE_CUSTOM_TRAIN_CONFIGS else DEFAULT_LR
        batch_size = configs["batch_size"] if USE_CUSTOM_TRAIN_CONFIGS else DEFAULT_BATCH_SIZE

        # General Configs
        eval_split = configs["eval_split"]
        text_column = configs["text_column"]
        label_column = configs["label_column"]
        classes = configs["classes"]

        # ┌───────────────────────────────────────────────┐
        # │                  DATASET PREP                 │
        # └───────────────────────────────────────────────┘

        # Init Training Dataset
        train_dataset = InSPEcTClassificationDataset(
            dataset_id = dataset_id,
            text_column = text_column,
            label_column = label_column,
            dataset_classes= classes,
            split = "train"
        )

        # Init Training DataLoader 
        train_dataloader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            collate_fn = collator
        )

        # Init Validation Dataset
        val_dataset = InSPEcTClassificationDataset(
            dataset_id = dataset_id,
            text_column = text_column,
            label_column = label_column,
            dataset_classes= classes,
            split = eval_split
        )

        # Init Validation DataLoader 
        val_dataloader = DataLoader(
            val_dataset,
            batch_size = batch_size,
            shuffle = False,
            collate_fn = collator
        )

        print(f"Successfully loaded dataset {dataset_name}!")

        # ┌───────────────────────────────────────────────┐
        # │               SOFT PROMPT INIT                │
        # └───────────────────────────────────────────────┘

        soft_prompt = SoftPrompt(
            model = llama_model,
            tokenizer = llama_tokenizer,
            word_embeddings = llama_word_embeddings,
            num_tokens = NUM_TOKENS
        ).to(DEVICE)

        # Init Optimizer with only params from Soft Prompt
        optimizer = torch.optim.AdamW(soft_prompt.parameters(), lr = lr)

        # ┌───────────────────────────────────────────────┐
        # │                 TRAINING LOOP                 │
        # └───────────────────────────────────────────────┘
        tqdm.write(f"\n--- Starting training for Dataset {dataset_id} ---")

        # Early Stopping Variables
        epochs_no_improve = 0
        best_soft_prompt_state = None
        best_val_loss = float('inf')
        best_val_accuracy = 0
        best_train_loss = float('inf')
        best_train_accuracy = 0

        # Loop EPOCHS times (or unitl Early Stopping triggers)
        for epoch in range(epochs):
            total_train_loss = 0
            train_correct_tokens = 0
            train_total_tokens = 0

            # Set the soft prompt in training mode
            soft_prompt.train()
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):

                # Reset gradients
                optimizer.zero_grad()
                
                # Move inputs to DEVICE
                input_ids = batch["input_ids"].to(DEVICE)                                       # (batch_size, seq_len)
                attention_mask = batch["attention_mask"].to(DEVICE)                             # (batch_size, seq_len)
                labels = batch["labels"].to(DEVICE)                                             # (batch_size, soft_prompt_len + seq_len)
                
                # Get the text embeddings from Llama
                with torch.no_grad():
                    text_embeds = llama_word_embeddings(input_ids).detach()
                    
                # Get the continuous Soft Prompt embeddings and duplicate for the batch
                soft_prompt_embeds = soft_prompt()                                              # (1, soft_prompt_len, embed_dim)
                soft_prompt_embeds = soft_prompt_embeds.expand(text_embeds.shape[0], -1, -1)    # (batch_size, soft_prompt_len, embed_dim)
                
                # Concatenate Embeddings: [Soft Prompt + Input Text]
                inputs_embeds = torch.cat([soft_prompt_embeds, text_embeds], dim=1)             # (batch_size, soft_prompt_len + seq_len, embed_dim)
                
                # Concatenate Attention Masks: Add `1`s so Llama pays attention to the soft prompt
                soft_prompt_mask = torch.ones((attention_mask.shape[0], NUM_TOKENS), dtype=attention_mask.dtype, device=DEVICE)     # (batch_size, soft_prompt_len)
                full_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)      # (batch_size, soft_prompt_len + seq_len)

                # Forward Pass
                outputs = llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_attention_mask,
                    labels=labels
                )
                
                # Extract the loss
                loss = outputs.loss
                
                # Backpropagate loss and update the parameters
                loss.backward()
                optimizer.step()

                logits = outputs.logits

                # Shift logits and labels so token i predicts i + 1
                shifted_logits = logits[..., :-1, :].contiguous()
                shifted_labels = labels[..., 1:].contiguous()

                # Get the predicted token ids
                preds = torch.argmax(shifted_logits, dim = -1)

                # Create a mask to ignore the -100 padding tokens
                valid_mask = (shifted_labels != -100)

                # Count correct predictions
                correct = (preds == shifted_labels) & valid_mask

                # Accumulate numbers for Accuracy and Avg Loss calculation per Epoch
                train_correct_tokens += correct.sum().item()
                train_total_tokens += valid_mask.sum().item()        
                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_dataloader)
            train_accuracy = (train_correct_tokens / train_total_tokens) * 100 if train_total_tokens > 0 else 0

            # Set soft_prompt in eval mode
            soft_prompt.eval()
            total_val_loss = 0
            val_correct_tokens = 0
            val_total_tokens = 0

            # Freeze all weights
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):

                    # Move inputs to DEVICE
                    input_ids = batch["input_ids"].to(DEVICE)                                       # (batch_size, seq_len)
                    attention_mask = batch["attention_mask"].to(DEVICE)                             # (batch_size, seq_len)
                    labels = batch["labels"].to(DEVICE)                                             # (batch_size, soft_prompt_len + seq_len)

                    # Get the text embeddings from Llama
                    text_embeds = llama_word_embeddings(input_ids).detach()
                    
                    # Get the continuous Soft Prompt embeddings and duplicate for the batch
                    soft_prompt_embeds = soft_prompt().expand(text_embeds.shape[0], -1, -1)         # (batch_size, soft_prompt_len, embed_dim)
                    
                    # Concatenate Embeddings: [Soft Prompt + Input Text]
                    inputs_embeds = torch.cat([soft_prompt_embeds, text_embeds], dim=1)             # (batch_size, soft_prompt_len + seq_len, embed_dim)
                    
                    # Concatenate Attention Masks: Add `1`s so Llama pays attention to the soft prompt
                    soft_prompt_mask = torch.ones((attention_mask.shape[0], NUM_TOKENS), dtype=attention_mask.dtype, device=DEVICE)     # (batch_size, soft_prompt_len)
                    full_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)      # (batch_size, soft_prompt_len + seq_len)

                    # Forward Pass
                    outputs = llama_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=full_attention_mask,
                        labels=labels
                    )

                    # Accumulate validation loss
                    total_val_loss += outputs.loss.item()

                    # Calculate Validation Accuracy
                    logits = outputs.logits
                    shifted_logits = logits[..., :-1, :].contiguous()
                    shifted_labels = labels[..., 1:].contiguous()

                    # Get the predicted token ids
                    preds = torch.argmax(shifted_logits, dim = -1)

                    # Create a mask to ignore the -100 padding tokens
                    valid_mask = (shifted_labels != -100)

                    # Count correct predictions
                    correct = (preds == shifted_labels) & valid_mask

                    # Accumulate numbers for Accuracy and Avg Loss calculation
                    val_correct_tokens += correct.sum().item()
                    val_total_tokens += valid_mask.sum().item()
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            val_accuracy = (val_correct_tokens / val_total_tokens) * 100 if val_total_tokens > 0 else 0

            tqdm.write(f"\nEpoch {epoch + 1} Summary:")
            tqdm.write(f"Train -> Loss: {avg_train_loss: .4f} | Accuracy: {train_accuracy: .2f}%")
            tqdm.write(f"Val   -> Loss: {avg_val_loss: .4f} | Accuracy: {val_accuracy: .2f}%")

            # TODO: Improve the logic here by stopping early when testing accuracy reaches 90%
            # ┌───────────────────────────────────────────────┐
            # │               EARLY STOPPING LOGIC            │
            # └───────────────────────────────────────────────┘
            # Check if the current validation loss is better than our best so far
            if avg_val_loss < (best_val_loss - MIN_DELTA):
                best_val_loss = avg_val_loss
                best_val_accuracy = val_accuracy
                best_train_loss = avg_train_loss
                best_train_accuracy = train_accuracy
                epochs_no_improve = 0
                
                # Save a copy of the best weights in memory
                best_soft_prompt_state = {k: v.cpu().clone() for k, v in soft_prompt.state_dict().items()}
                tqdm.write(f"  --> Validation loss improved! Saving current state.")
            else:
                epochs_no_improve += 1
                tqdm.write(f"  --> No improvement. Patience: {epochs_no_improve}/{PATIENCE}")

            # Trigger Early Stopping
            if epochs_no_improve >= PATIENCE:
                tqdm.write(f"\n[Early Stopping Triggered] Convergence reached at epoch {epoch + 1}.")
                break
            
        # ┌───────────────────────────────────────────────┐
        # │              SAVE SOFT PROMPTS                │
        # └───────────────────────────────────────────────┘
        os.makedirs(save_dir, exist_ok=True)

        # Load the best state back into the model before saving
        if best_soft_prompt_state is not None:
            soft_prompt.load_state_dict(best_soft_prompt_state)

        soft_prompt.save_softprompt(save_dir)
        tqdm.write(f"\nTraining complete! Soft prompt saved to {save_dir}/softprompt.pt")


        # ┌───────────────────────────────────────────────┐
        # │            WRITE TRAINING STATS               │
        # └───────────────────────────────────────────────┘
        # Check if current dataset id exists:
        if dataset_id in training_stats['dataset_id']:
            idx = training_stats['dataset_id'].index(dataset_id)
            training_stats['train_accuracy'][idx] = round(best_train_accuracy, 4)
            training_stats['val_accuracy'][idx] = round(best_val_accuracy, 4)
            training_stats['avg_train_loss'][idx] = round(best_train_loss, 4)
            training_stats['avg_val_loss'][idx] = round(best_val_loss, 4)
        else:
            training_stats['dataset_id'].append(dataset_id)
            training_stats['train_accuracy'].append(round(best_train_accuracy, 4))
            training_stats['val_accuracy'].append(round(best_val_accuracy, 4))
            training_stats['avg_train_loss'].append(round(best_train_loss, 4))
            training_stats['avg_val_loss'].append(round(best_val_loss, 4))

        # Save the CSV file with training stats
        df = pd.DataFrame(training_stats)
        df.to_csv(ACCURACY_STATS_FILE_PATH, index=False)


        # Free up some allocations
        del soft_prompt
        del optimizer
        del df
        torch.cuda.empty_cache()
