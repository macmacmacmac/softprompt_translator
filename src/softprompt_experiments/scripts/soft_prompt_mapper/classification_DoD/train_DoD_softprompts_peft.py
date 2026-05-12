import os
import argparse
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from tqdm import tqdm
import pandas as pd


class SQLiteClassificationDataset(Dataset):
    def __init__(self, db_path, dataset_id, split="train"):
        """
        Fetches all sentences and their target keywords for a specific dataset_id.
        """
        self.db_path = db_path
        self.dataset_id = dataset_id
        self.split = split
        
        self.inputs = []
        self.targets = []
        
        # Load the data into RAM upon initialization
        self._load_data_from_sqlite()


    def _load_data_from_sqlite(self):
        # Open a temporary connection just to fetch the data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # We use an INNER JOIN to grab the sentence AND the actual keyword text in one jump.
        # Thanks to the composite B-Tree index, this executes in < 1 millisecond.
        query = """
            SELECT s.sentence, k.keyword
            FROM sentences s
            JOIN keywords k ON s.keyword_id = k.keyword_id
            WHERE s.dataset_id = ? AND s.split = ?
        """
        
        cursor.execute(query, (self.dataset_id, self.split))
        rows = cursor.fetchall()
        
        for sentence, keyword in rows:
            # Format the input text so the LLM knows what to do
            # (The soft prompt will be prepended to this later)
            input_text = f"Sentence: {sentence} Label:"
            
            # Format the target text
            target_text = f" {keyword}"
            
            self.inputs.append(input_text)
            self.targets.append(target_text)
            
        conn.close()
        
        if len(self.inputs) == 0:
            raise ValueError(f"No data found for dataset_id {self.dataset_id} (split: {self.split})")


    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, idx):
        # Return the raw strings
        # we will tokenize later
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
            inp_len = len(self.tokenizer.encode(inp, add_special_tokens=True))
            
            # Mask the input portion so loss is not calculated on it
            labels[i, :inp_len] = -100
            
            # Mask any padding tokens added to the end of the sequence
            labels[i, attention_mask[i] == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }



def train_soft_prompts(model, 
                       train_dataloader, 
                       eval_dataloader, 
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

    # Loop num_epochs times
    for epoch in range(num_epochs):

        # Set the model on training mode
        model.train()

        # Calculate total loss
        total_loss = 0
        train_correct_tokens = 0
        train_total_tokens = 0

        # For each batch in the dataloader
        for batch in tqdm(train_dataloader):

            # Reset Gradients
            optimizer.zero_grad()

            # Move all items of data batch to the device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward Pass Thru the model
            outputs = model(**batch)

            # Extract Loss and accumulate it to the total loss
            loss = outputs.loss
            total_loss += loss.detach()

            # Calculate batch accuracy
            with torch.no_grad():
                top_tokens = torch.argmax(outputs.logits, dim=-1)[:, num_tokens-1:-1]
                
                # Create a mask to ignore the -100 padding positions
                valid_mask = (batch['labels'] != -100)
                
                # Check where predictions match the labels exactly
                correct_tokens = (batch['labels'] == top_tokens) & valid_mask
                
                train_correct_tokens += correct_tokens.sum().item()
                train_total_tokens += valid_mask.sum().item()

            # Compute Gradients and Do backpropagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        # Eval Model at the end of this epoch in terms of loss and correctness
        eval_loss, eval_correct_tokens, eval_total_tokens = eval_soft_prompts(model, eval_dataloader, num_tokens, device)

        # Calculate Val and Train accuracy based on individual tokens
        val_accuracy = (eval_correct_tokens / eval_total_tokens) if eval_total_tokens > 0 else 0
        train_accuracy = (train_correct_tokens / train_total_tokens) if train_total_tokens > 0 else 0

        # Calculate Avg Val and Training Loss    
        avg_val_loss = eval_loss / len(eval_dataloader)
        avg_train_loss = total_loss / len(train_dataloader)

        tqdm.write(f"\nEpoch {epoch + 1} Summary:")
        tqdm.write(f"Train -> Loss: {avg_train_loss: .4f} | Accuracy: {train_accuracy * 100: .2f}%")
        tqdm.write(f"Val   -> Loss: {avg_val_loss: .4f} | Accuracy: {val_accuracy * 100: .2f}%")

    # Save the trained soft prompt
    os.makedirs(soft_prompt_save_dir, exist_ok=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad][0]
    torch.save(trainable_params, os.path.join(soft_prompt_save_dir, "softprompt.pt"))
    tqdm.write(f"\nTraining complete! Soft prompt saved to {soft_prompt_save_dir}/softprompt.pt")

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
    eval_correct_tokens = 0
    eval_total_tokens = 0
    
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU
        with torch.no_grad():             # Disable gradient computation
            outputs = model(**batch)      # Forward pass with input_ids, attention_mask, labels
        loss = outputs.loss               # Get cross-entropy loss
        eval_loss += loss.detach().float()
        
        # Get the most likely token at each position
        # [:,num_tokens-1:-1] skips the soft prompt tokens and last token
        top_tokens = torch.argmax(outputs.logits, dim=-1)[:, num_tokens-1:-1]
        
        # Create a mask to ignore the -100 padding positions
        valid_mask = (batch['labels'] != -100)
        
        # Check where predictions match the labels exactly
        correct_tokens = (batch['labels'] == top_tokens) & valid_mask
        
        eval_correct_tokens += correct_tokens.sum().item()
        eval_total_tokens += valid_mask.sum().item()
    
    return eval_loss, eval_correct_tokens, eval_total_tokens


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
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--db_path", type=str, default="./datasets/mapper_classification_datasets/DoD_3_5k.sqlite")
    parser.add_argument("--save_dir", type=str, default="./trained_soft_prompts")
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)
    

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    DB_PATH = args.db_path
    LR = args.lr
    EPOCHS = args.epochs
    NUM_TOKENS = args.num_tokens
    BATCH_SIZE = args.batch_size
    SAVE_DIR = args.save_dir
    PATIENCE = args.patience
    MIN_DELTA = args.min_delta


    # Create Parent Directory to save all soft prompts for this Dataset
    DB_NAME = DB_PATH.split("/")[-1].split(".")[0]
    PARENT_DIR = f"{SAVE_DIR}/{DB_NAME}_peft"
    os.makedirs(PARENT_DIR, exist_ok=True)


    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

    # Init Collator
    collator = CausalLMBatchCollator(
        tokenizer = llama_tokenizer,
        soft_prompt_length = NUM_TOKENS
    )

    # Get all Dataset ids
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT dataset_id FROM datasets")
    dataset_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Init Dataset Progress Bar
    dataset_pbar = tqdm(dataset_ids, desc = "Master Dataset Progress")


    # Determine file path for accuracy stats
    ACCURACY_STATS_FILE_PATH = f"{PARENT_DIR}/accuracy_stats.csv"

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
    for dataset_id in dataset_pbar:
        save_dir = f"{PARENT_DIR}/dataset_{dataset_id}"

        # If there exists an already trained soft prompt for this dataset id, then skip this
        if os.path.exists(save_dir):
            tqdm.write(f"Skipping training for dataset id: {dataset_id}")
            continue
            
        # Update the outer progress bar so you know exactly which dataset is training
        dataset_pbar.set_postfix({"Current Dataset id": dataset_id})

        # ┌───────────────────────────────────────────────┐
        # │                  DATASET PREP                 │
        # └───────────────────────────────────────────────┘

        # Init Training Dataset
        train_dataset = SQLiteClassificationDataset(
            db_path = DB_PATH,
            dataset_id = dataset_id,
            split = "train"
        )
        # Init Training DataLoader 
        train_dataloader = DataLoader(
            train_dataset,
            batch_size = BATCH_SIZE,
            shuffle = True,
            collate_fn = collator
        )

        # Init Validation Dataset
        val_dataset = SQLiteClassificationDataset(
            db_path = DB_PATH,
            dataset_id = dataset_id,
            split = "test"
        )

        # Init Validation DataLoader 
        val_dataloader = DataLoader(
            val_dataset,
            batch_size = BATCH_SIZE,
            shuffle = False,
            collate_fn = collator
        )


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
        tqdm.write(f"\n--- Starting training for Dataset {dataset_id} ---")

        stats = train_soft_prompts(
            model=peft_model,
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
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
        # Check if current dataset id exists:
        if dataset_id in training_stats['dataset_id']:
            idx = training_stats['dataset_id'].index(dataset_id)
            training_stats['train_accuracy'][idx] = stats["train_accuracy"]
            training_stats['val_accuracy'][idx] = stats["val_accuracy"]
            training_stats['avg_train_loss'][idx] = stats["avg_train_loss"]
            training_stats['avg_val_loss'][idx] = stats["avg_val_loss"]
        else:
            training_stats['dataset_id'].append(dataset_id)
            training_stats['train_accuracy'].append(stats["train_accuracy"])
            training_stats['val_accuracy'].append(stats["val_accuracy"])
            training_stats['avg_train_loss'].append(stats["avg_train_loss"])
            training_stats['avg_val_loss'].append(stats["avg_val_loss"])

        # Save the CSV file with training stats
        df = pd.DataFrame(training_stats)
        df.to_csv(ACCURACY_STATS_FILE_PATH, index=False)


        # Free up some allocations
        del peft_model
        del df
        torch.cuda.empty_cache()
