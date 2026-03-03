import os
import argparse
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from softprompt_experiments.models.softprompt import SoftPrompt
from tqdm import tqdm


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
        
        input_ids = tokenized["input_ids"]
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--db_path", type=str, default="./datasets/mapper_classification_datasets/classification_5k.sqlite")
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)
    

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    DB_PATH = args.db_path
    LR = args.lr
    EPOCHS = args.epochs
    NUM_TOKENS = args.num_tokens
    BATCH_SIZE = args.batch_size
    SEED = args.seed


    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # # Llama doesn't have a default pad token, so we map it to EOS
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

    # Get all Dataset ids
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT dataset_id FROM datasets")
    dataset_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Dataset Progress Bar
    dataset_pbar = tqdm(dataset_ids, desc = "Master Dataset Progress")

    # Loop over all dataset ids
    for dataset_id in dataset_pbar:
        save_dir = f"./trained_soft_prompts/dataset_{dataset_id}"

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

        soft_prompt = SoftPrompt(
            model = llama_model,
            tokenizer = llama_tokenizer,
            word_embeddings = llama_word_embeddings,
            num_tokens = NUM_TOKENS
        ).to(DEVICE)

        # Init Optimizer with only params from Soft Prompt
        optimizer = torch.optim.AdamW(soft_prompt.parameters(), lr = LR)

        # ┌───────────────────────────────────────────────┐
        # │                 TRAINING LOOP                 │
        # └───────────────────────────────────────────────┘
        tqdm.write(f"\n--- Starting training for Dataset {dataset_id} ---")

        # Loop EPOCH times
        for epoch in range(EPOCHS):
            total_train_loss = 0
            train_correct_tokens = 0
            train_total_tokens = 0

            # Set the soft prompt in training mode
            soft_prompt.train()
            
            for batch in train_dataloader:

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
            
        # ┌───────────────────────────────────────────────┐
        # │              SAVE SOFT PROMPTS                │
        # └───────────────────────────────────────────────┘
        save_dir = f"./trained_soft_prompts/dataset_{dataset_id}"
        os.makedirs(save_dir, exist_ok=True)
        soft_prompt.save_softprompt(save_dir)
        tqdm.write(f"\nTraining complete! Soft prompt saved to {save_dir}/softprompt.pt")

        # Free up some allocations
        del soft_prompt
        del optimizer
        torch.cuda.empty_cache()