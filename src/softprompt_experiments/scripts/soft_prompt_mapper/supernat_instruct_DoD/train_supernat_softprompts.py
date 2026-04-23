import os
import argparse
import torch
import math
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from softprompt_experiments.models.softprompt import SoftPrompt
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from datasets import concatenate_datasets


"""
NOTES:
1. There is variability in the number of instances (Input-output pairs) per task.
--- TRAIN SPLIT INSTANCES PER TASK STATS ---
Total unique tasks: 756
Min instances in a task: 29
Max instances in a task: 82995
Mean instances per task: 4068.23
Median instances per task: 2972.50

--- TEST SPLIT INSTANCES PER TASK STATS ---
Total unique tasks: 119
Min instances in a task: 26
Max instances in a task: 16897
Mean instances per task: 4067.28
Median instances per task: 2855.00

2. There is variability in the length (tokens) of input - output sequences:
--- TRAIN SPLIT INPUT+OUTPUT LENGTH STATS ---
Total instances analyzed: 3075585
Min length: 2
Max length: 222451
Mean length: 143.68
Median length: 47.00
90th percentile: 337.00
95th percentile: 566.00
99th percentile: 1312.00

--- TEST SPLIT INPUT+OUTPUT LENGTH STATS ---
Total instances analyzed: 484006
Min length: 3
Max length: 12092
Mean length: 108.16
Median length: 56.00
90th percentile: 277.00
95th percentile: 365.00
99th percentile: 681.00

3. We need to have tasks selected which have min 500 training instances and train only using those? TODO: Confirm this

"""

class TaskDataset(Dataset):
    def __init__(self, data_rows):
        """
        Fetches all sentences and their targets for a specific task.
        """
        self.inputs = []
        self.targets = []
        
        for row in data_rows:
            # Format the input text so the LLM knows what to do
            # (The soft prompt will be prepended to this later)
            input_text = f"Input: {row['input']}\nOutput:"

            # Format the target text
            target_text = f" {row['output']}"
            
            self.inputs.append(input_text)
            self.targets.append(target_text)
        
        if len(self.inputs) == 0:
            raise ValueError("No data found for task")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Return the raw strings, tokenized later by collator
        return self.inputs[idx], self.targets[idx]
    

class CausalLMBatchCollator:
    def __init__(self, tokenizer, soft_prompt_length=20, max_length=512):
        self.tokenizer = tokenizer
        self.soft_prompt_length = soft_prompt_length
        self.max_length = max_length

    def __call__(self, batch):
        inputs, targets = zip(*batch)
        
        # Combine input and target into the full sequence the model needs to see
        full_texts = [f"{inp}{tgt}" for inp, tgt in zip(inputs, targets)]
        
        # Tokenize the full sequences (this gives us input_ids and attention_mask)
        tokenized = self.tokenizer(
            full_texts, 
            padding=True, 
            truncation=True,
            max_length=self.max_length, 
            return_tensors="pt",
            add_special_tokens=True
        )
        
        input_ids = tokenized["input_ids"]                              # (batch_size, seq_len)
        attention_mask = tokenized["attention_mask"]                    # (batch_size,)
        
        # Create the labels tensor
        labels = input_ids.clone()                                      # (batch_size, seq_len)
        
        # Mask out the input text and the padding tokens with -100
        for i, (inp, tgt) in enumerate(zip(inputs, targets)):

            # Tokenize just the input text to find out how long it is
            inp_len = len(self.tokenizer.encode(inp, add_special_tokens=True))

            # Mask the input portion so loss is not calculated on it
            labels[i, :inp_len] = -100

            # Mask any padding tokens added to the end of the sequence
            labels[i, attention_mask[i] == 0] = -100

        # Account for Soft Prompt in labels
        # Because we will prepend `soft_prompt_length` virtual embeddings to the front
        # of the inputs in the training loop, we must pad the front of our labels with -100s
        # so the matrix dimensions line up perfectly for the loss function.
        batch_size = labels.size(0)
        soft_prompt_labels = torch.full((batch_size, self.soft_prompt_length), -100, dtype=torch.long)  # (batch_size, soft_prompt_len)
        labels = torch.cat([soft_prompt_labels, labels], dim=1)                                         # (batch_size, soft_prompt_len + seq_len)

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--num_tokens", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--dataset_path", type=str, default="Suryanshg/SUPER-NATURALINSTRUCTIONS-english")
    parser.add_argument("--save_dir", type=str, default="./trained_soft_prompts/SUPER-NATURALINSTRUCTIONS-english")
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    DATASET_PATH = args.dataset_path
    LR = args.lr
    EPOCHS = args.epochs
    NUM_TOKENS = args.num_tokens
    MAX_LENGTH = args.max_length
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    SAVE_DIR = args.save_dir
    PATIENCE = args.patience
    MIN_DELTA = args.min_delta

    # Create Directory to save all soft prompts for this Dataset
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token # Llama doesn't have a default pad token, so we map it to EOS

    # Load Llama Model and set it in eval mode
    llama_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        device_map = DEVICE
    )
    llama_model.eval()

    # Freeze the entire Llama model since we only want to train Soft Prompts
    for param in llama_model.parameters():
        param.requires_grad = False

    # Get the Llama Model's Word Embedding Mappings
    llama_word_embeddings = llama_model.get_input_embeddings()

    # Init Collator
    collator = CausalLMBatchCollator(
        tokenizer=llama_tokenizer,
        soft_prompt_length=NUM_TOKENS,
        max_length=MAX_LENGTH
    )

    # Load HF Dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    hf_ds = load_dataset(DATASET_PATH)

    # Combine splits to get all tasks and their respective instances across the dataset
    splits_to_concat = [hf_ds[split] for split in hf_ds.keys()]
    full_ds = concatenate_datasets(splits_to_concat)
    
    # Convert to pandas for grouping instances by task efficiently without .filter() iteration
    full_df = full_ds.to_pandas()
    grouped_tasks = full_df.groupby('task_name')

    # Get all unique tasks
    unique_tasks = list(grouped_tasks.groups.keys())
    print(f"Found {len(unique_tasks)} unique training tasks to process.")
    
    # Init Dataset Progress Bar
    dataset_pbar = tqdm(unique_tasks, desc="Master Task Progress")

    # Init file path for training stats
    TRAINING_STATS_FILE_PATH = f"{SAVE_DIR}/training_stats.csv"

    # Preload existing training_stats, if it exists already
    if os.path.isfile(TRAINING_STATS_FILE_PATH):
        df = pd.read_csv(TRAINING_STATS_FILE_PATH)
        training_stats = df.to_dict(orient='list')
    else:
        # Init a dict to save training stats
        training_stats = {
            'task_name': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'avg_train_loss': [],
            'avg_val_loss': []
        }

    # Loop over all tasks
    for task_name in dataset_pbar:
        save_dir = f"{SAVE_DIR}/{task_name}"

        # If there exists an already trained soft prompt for this dataset id, then skip this
        if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "softprompt.pt")):
            tqdm.write(f"Skipping training for task: {task_name}")
            continue
        
        # Update the outer progress bar so you know exactly which dataset is training
        dataset_pbar.set_postfix({"Current Task": task_name})

        # ┌───────────────────────────────────────────────┐
        # │                  DATASET PREP                 │
        # └───────────────────────────────────────────────┘

        # Fetch all rows for this task across the entire dataset
        task_df = grouped_tasks.get_group(task_name)
        
        # Perform a 90/10 split on the input/output pairs
        split_idx = int(len(task_df) * 0.9)
        if split_idx == 0 or split_idx == len(task_df):
            raise ValueError(f"Not enough dataset rows for a 90/10 split (Total Rows: {len(task_df)})")
        
        # Extract rows for training and testing split
        train_rows = task_df.iloc[:split_idx].to_dict('records')
        val_rows = task_df.iloc[split_idx:].to_dict('records')
        
        # Use train/val rows to create datasets
        train_dataset = TaskDataset(train_rows)
        val_dataset = TaskDataset(val_rows)

        # Init train/val dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

        # ┌───────────────────────────────────────────────┐
        # │               SOFT PROMPT INIT                │
        # └───────────────────────────────────────────────┘

        # Init Soft Prompt
        soft_prompt = SoftPrompt(
            model=llama_model,
            tokenizer=llama_tokenizer,
            word_embeddings=llama_word_embeddings,
            num_tokens=NUM_TOKENS
        ).to(DEVICE)

        # Init Optimizer with only params from Soft Prompt
        optimizer = torch.optim.AdamW(soft_prompt.parameters(), lr=LR)

        # ┌───────────────────────────────────────────────┐
        # │                 TRAINING LOOP                 │
        # └───────────────────────────────────────────────┘
        tqdm.write(f"\n--- Starting training for Task {task_name} ---")

        # Early Stopping Variables
        epochs_no_improve = 0
        best_soft_prompt_state = None
        best_val_loss = float('inf')
        best_val_perplexity = float('inf')
        best_train_loss = float('inf')
        best_train_perplexity = float('inf')

        # Loop EPOCHS times (or until Early Stopping triggers)
        for epoch in range(EPOCHS):
            total_train_loss = 0

            # Set the soft prompt in training mode
            soft_prompt.train()
            optimizer.zero_grad()
            
            for step, batch in enumerate(train_dataloader):
                
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
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

                # Accumulate Gradients
                loss.backward()
                
                # Backpropagate losses after every GRADIENT_ACCUMULATION_STEPS
                if ((step + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or (step + 1 == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()

                total_train_loss += (loss.item() * GRADIENT_ACCUMULATION_STEPS)

                # Free up memory explicitly
                del outputs, loss, inputs_embeds, text_embeds, soft_prompt_embeds, soft_prompt_mask, full_attention_mask
                torch.cuda.empty_cache()
                
            avg_train_loss = total_train_loss / len(train_dataloader)
            train_perplexity = math.exp(avg_train_loss) if avg_train_loss < 50 else float('inf')

            # Set soft_prompt in eval mode
            soft_prompt.eval()
            total_val_loss = 0

            # Freeze all weights
            with torch.no_grad():
                for batch in val_dataloader:

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

                    # Free up memory explicitly
                    del outputs, inputs_embeds, text_embeds, soft_prompt_embeds, soft_prompt_mask, full_attention_mask
                    torch.cuda.empty_cache()
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 50 else float('inf')

            tqdm.write(f"\nEpoch {epoch + 1} Summary:")
            tqdm.write(f"Train -> Loss: {avg_train_loss: .4f} | Perplexity: {train_perplexity: .4f}")
            tqdm.write(f"Val   -> Loss: {avg_val_loss: .4f} | Perplexity: {val_perplexity: .4f}")

            # ┌───────────────────────────────────────────────┐
            # │               EARLY STOPPING LOGIC            │
            # └───────────────────────────────────────────────┘
            # Check if the current validation loss is better than our best so far
            if avg_val_loss < (best_val_loss - MIN_DELTA):
                best_val_loss = avg_val_loss
                best_val_perplexity = val_perplexity
                best_train_loss = avg_train_loss
                best_train_perplexity = train_perplexity
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
        # Check if current task_name exists:
        if task_name in training_stats['task_name']:
            idx = training_stats['task_name'].index(task_name)
            training_stats['train_perplexity'][idx] = round(best_train_perplexity, 4)
            training_stats['val_perplexity'][idx] = round(best_val_perplexity, 4)
            training_stats['avg_train_loss'][idx] = round(best_train_loss, 4)
            training_stats['avg_val_loss'][idx] = round(best_val_loss, 4)
        else:
            training_stats['task_name'].append(task_name)
            training_stats['train_perplexity'].append(round(best_train_perplexity, 4))
            training_stats['val_perplexity'].append(round(best_val_perplexity, 4))
            training_stats['avg_train_loss'].append(round(best_train_loss, 4))
            training_stats['avg_val_loss'].append(round(best_val_loss, 4))

        # Save the CSV file with training stats
        df = pd.DataFrame(training_stats)
        df.to_csv(TRAINING_STATS_FILE_PATH, index=False)

        # Free up some allocations
        del soft_prompt
        del optimizer
        del df
        torch.cuda.empty_cache()