import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, concatenate_datasets
import evaluate


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
        
        # Because SUPER-NATURALINSTRUCTIONS DoD requires open ended generation, we need to 
        # also prep just the input sequence for doing evaluations after training
        input_tokenized = self.tokenizer(
            list(inputs),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True
        )

        return {

            # Full Sequence related data
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,

            # Input only related data
            "input_only_ids": input_tokenized["input_ids"],
            "input_only_attention_mask": input_tokenized["attention_mask"],
            "targets": list(targets)
        }



def train_soft_prompts(model, 
                       train_dataloader, 
                       eval_dataloader, 
                       tokenizer,
                       max_length,
                       rouge_metric,
                       num_tokens, 
                       soft_prompt_save_dir,
                       num_epochs, 
                       lr, 
                       device
    ):
    
    # Init Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Init Linear Scheduler for Learning Rate
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    best_val_loss = float('inf')
    best_val_rougeL = 0.0
    best_train_loss = float('inf')

    # Loop num_epochs times
    for epoch in range(num_epochs):

        # Set the model on training mode
        model.train()

        # Calculate total loss
        total_loss = 0

        # For each batch in the dataloader
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):

            # Reset Gradients
            optimizer.zero_grad()

            # Move relevant items to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward Pass Thru the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Extract Loss and accumulate it to the total loss
            loss = outputs.loss
            total_loss += loss.detach()

            # Compute Gradients and Do backpropagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

             # Free up memory explicitly
            del outputs, loss, input_ids, attention_mask, labels
        
        # Free up memory after epoch
        torch.cuda.empty_cache()
        
        # Eval Model at the end of this epoch
        eval_loss, val_rougeL = eval_soft_prompts(model, eval_dataloader, tokenizer, max_length, rouge_metric, device)

        # Calculate Avg Val and Training Loss    
        avg_val_loss = eval_loss / len(eval_dataloader)
        avg_train_loss = total_loss / len(train_dataloader)

        tqdm.write(f"\nEpoch {epoch + 1} Summary:")
        tqdm.write(f"Train -> Loss: {avg_train_loss: .4f}")
        tqdm.write(f"Val   -> Loss: {avg_val_loss: .4f} | RougeL: {val_rougeL: .4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_rougeL = val_rougeL
            best_train_loss = avg_train_loss

    # Save the trained soft prompt
    os.makedirs(soft_prompt_save_dir, exist_ok=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad][0]
    torch.save(trainable_params, os.path.join(soft_prompt_save_dir, "softprompt.pt"))
    tqdm.write(f"\nTraining complete! Soft prompt saved to {soft_prompt_save_dir}/softprompt.pt")

    return {
        "val_rougeL": best_val_rougeL,
        "avg_train_loss": best_train_loss.item() if torch.is_tensor(best_train_loss) else best_train_loss,
        "avg_val_loss": best_val_loss.item() if torch.is_tensor(best_val_loss) else best_val_loss
    }


def eval_soft_prompts(model, 
                      eval_dataloader, 
                      tokenizer,
                      max_length,
                      rouge_metric,
                      device):
    
    # Set model to evaluation mode
    model.eval()                          
    eval_loss = 0
    
    val_preds = []
    val_targets = []
    
    for batch in tqdm(eval_dataloader, desc="Eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        input_only_ids = batch["input_only_ids"].to(device)
        input_only_attention_mask = batch["input_only_attention_mask"].to(device)

        with torch.no_grad():             # Disable gradient computation
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )      
            
            loss = outputs.loss               # Get cross-entropy loss
            eval_loss += loss.detach().float()
            
            # Generate Outputs for RougeL using PEFT's text generation
            # PEFT will automatically prepend the learned virtual tokens to input_only_ids.
            generated_ids = model.generate(
                input_ids=input_only_ids,
                attention_mask=input_only_attention_mask,
                max_new_tokens=max_length,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            val_preds.extend(decoded_preds)
            val_targets.extend(batch["targets"])
            
        del outputs, generated_ids, input_ids, attention_mask, labels, input_only_ids, input_only_attention_mask

    # Free up memory after evaluation
    torch.cuda.empty_cache()
    
    # compute rougeL
    rouge_result = rouge_metric.compute(predictions=val_preds, references=val_targets)
    val_rougeL = rouge_result['rougeL']
    
    return eval_loss, val_rougeL


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
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset_path", type=str, default="SoftPromptTranslator/SUPER-NATURALINSTRUCTIONS-english-filtered")
    parser.add_argument("--save_dir", type=str, default="./trained_soft_prompts/SUPER-NATURALINSTRUCTIONS-english-filtered_peft")
    parser.add_argument("--num_examples", type=int, default=500, help = "num of examples to use per task for training and eval of soft prompts")
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
    SAVE_DIR = args.save_dir
    PATIENCE = args.patience
    MIN_DELTA = args.min_delta
    NUM_EXAMPLES = args.num_examples
    SEED = args.seed

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

    # Freeze the entire Llama model since we only want to train the Soft prompt
    for param in llama_model.parameters():
        param.requires_grad = False

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
            'val_rougeL': [],
            'avg_train_loss': [],
            'avg_val_loss': []
        }
        
    # Init ROUGE metric
    rouge_metric = evaluate.load('rouge')

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

        # Cap the task size if it exceeds our num_examples threshold
        if len(task_df) > NUM_EXAMPLES:
            # randomly sample from the task dataframe
            task_df = task_df.sample(n=NUM_EXAMPLES, random_state=SEED)
        
        # Perform a 90/10 split on the input/output pairs
        split_idx = int(len(task_df) * 0.9)
        
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

        # Get task-specific max length dynamically from the dataframe (fallback to MAX_LENGTH if missing)
        task_max_length = int(task_df['total_tokens'].max()) if 'total_tokens' in task_df.columns else MAX_LENGTH

        # Init Collator
        collator = CausalLMBatchCollator(
            tokenizer=llama_tokenizer,
            soft_prompt_length=NUM_TOKENS,
            max_length=task_max_length
        )

        # Init train/val dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)


        # ┌───────────────────────────────────────────────┐
        # │               SOFT PROMPT INIT                │
        # └───────────────────────────────────────────────┘
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=NUM_TOKENS,
            tokenizer_name_or_path=MODEL_NAME
        )
        peft_model = get_peft_model(llama_model, peft_config)

        # ┌───────────────────────────────────────────────┐
        # │                  TRAINING                     │
        # └───────────────────────────────────────────────┘
        tqdm.write(f"\n--- Starting training for Task {task_name}, Max Tokens {task_max_length} ---")

        stats = train_soft_prompts(
            model=peft_model,
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
            tokenizer=llama_tokenizer,
            max_length=task_max_length,
            rouge_metric=rouge_metric,
            num_tokens=NUM_TOKENS,
            soft_prompt_save_dir=save_dir,
            num_epochs=EPOCHS,
            lr=LR,
            device=DEVICE
        )

        # Unwrap PEFT model to prepare for the next dataset loop
        
        # We wrapped a LlamaForCausalLM object (not standard PeftModel format), 
        # so calling base_model returns a PeftModelForCausalLM.
        # We need the underlying PyTorch LlamaForCausalLM model directly.
        llama_model = peft_model.base_model

        # ┌───────────────────────────────────────────────┐
        # │            WRITE TRAINING STATS               │
        # └───────────────────────────────────────────────┘
        # Check if current task_name exists:
        if task_name in training_stats['task_name']:
            idx = training_stats['task_name'].index(task_name)
            training_stats['val_rougeL'][idx] = round(stats["val_rougeL"], 4)
            training_stats['avg_train_loss'][idx] = round(stats["avg_train_loss"], 4)
            training_stats['avg_val_loss'][idx] = round(stats["avg_val_loss"], 4)
        else:
            training_stats['task_name'].append(task_name)
            training_stats['val_rougeL'].append(round(stats["val_rougeL"], 4))
            training_stats['avg_train_loss'].append(round(stats["avg_train_loss"], 4))
            training_stats['avg_val_loss'].append(round(stats["avg_val_loss"], 4))

        # Save the CSV file with training stats
        df = pd.DataFrame(training_stats)
        df.to_csv(TRAINING_STATS_FILE_PATH, index=False)


        # Free up some allocations
        del peft_model
        del df
        torch.cuda.empty_cache()
