import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# PyTorch Dataset wrapper on the Mapper Dataset from Soft Prompts to Hard Prompts
class MapperDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Return the (20, 4096) tensor and the target string
        return self.data[idx]["soft_prompt"], self.data[idx]["hard_prompt"]


# Custom Data Collator for the Mapper Dataset
class MapperCollator:
    def __init__(self, tokenizer, soft_prompt_length=20):
        self.tokenizer = tokenizer
        self.soft_prompt_length = soft_prompt_length

    def __call__(self, batch):
        soft_prompts, hard_prompts = zip(*batch)
        
        # Stack the frozen soft prompts into a batch: (batch_size, soft_prompt_len, embed_dim)
        soft_prompts = torch.stack(soft_prompts)        # (batch_size, soft_prompt_len, embed_dim)
        
        # Tokenize the target hard prompts
        tokenized = self.tokenizer(
            hard_prompts, 
            padding=True, 
            truncation=True, 
            max_length=64,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        input_ids = tokenized["input_ids"]              # (batch_size, seq_len)
        attention_mask = tokenized["attention_mask"]    # (batch_size, seq_len)
        
        # Create labels and mask the padding tokens with -100
        labels = input_ids.clone()                      # (batch_size, seq_len)
        labels[attention_mask == 0] = -100              # (batch_size, seq_len)

        return {
            "soft_prompts": soft_prompts,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--mapper_dataset_path", type=str, default="./datasets/mapper_training_dataset/DoD_3_5k")
    parser.add_argument("--save_dir", type=str, default="./mapper_lora_weights")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--optim_weight_decay", type=float, default=0.1) 
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MAPPER_DATASET_PATH = args.mapper_dataset_path
    DB_NAME = MAPPER_DATASET_PATH.split('/')[-1]
    SAVE_DIR = args.save_dir
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_TOKENS = args.num_tokens
    LORA_RANK = args.lora_rank
    LORA_DROPOUT = args.lora_dropout
    OPTIM_WEIGHT_DECAY = args.optim_weight_decay

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token       # Llama doesn't have a default pad token, so we map it to EOS


    # ┌───────────────────────────────────────────────┐
    # │                 LORA MODEL PREP               │
    # └───────────────────────────────────────────────┘
    print(f"Loading base model {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    
    # Configure LoRA Config to target the key linear layers of attention and feed-forward networks
    lora_config = LoraConfig(
        r = LORA_RANK, 
        lora_alpha = 2 * LORA_RANK,
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout = LORA_DROPOUT,
        bias = "none",
        task_type = TaskType.CAUSAL_LM
    )
    
    # Attach the LoRA adapters to the base model
    model = get_peft_model(base_model, lora_config)
    
    # Print out exactly how many params are trainable
    model.print_trainable_parameters() 
    
    # Get the Llama Model's Word Embedding Mappings
    llama_word_embeddings = model.get_base_model().get_input_embeddings()

    # Init Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = LR, 
                                  weight_decay = OPTIM_WEIGHT_DECAY)


    # ┌───────────────────────────────────────────────┐
    # │                   DATASET PREP                │
    # └───────────────────────────────────────────────┘
    print("Loading Train and Validation datasets ...")
    train_dataset = torch.load(os.path.join(MAPPER_DATASET_PATH, 'train_mapper_dataset.pt'), map_location="cpu", weights_only=True)
    val_dataset = torch.load(os.path.join(MAPPER_DATASET_PATH, 'val_mapper_dataset.pt'), map_location="cpu", weights_only=True)
    
    print(f"Train Dataset size: {len(train_dataset)} | Validation Dataset size: {len(val_dataset)}")

    # Init Collator
    collator = MapperCollator(
        tokenizer = tokenizer, 
        soft_prompt_length = NUM_TOKENS
    )

    # Init Training Dataloader
    train_dataloader = DataLoader(
        MapperDataset(train_dataset), 
        batch_size = BATCH_SIZE, 
        shuffle = True, 
        collate_fn = collator
    )
    
    # Init Validation Dataloader
    val_dataloader = DataLoader(
        MapperDataset(val_dataset), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collator
    )

    # ┌───────────────────────────────────────────────┐
    # │                 TRAINING LOOP                 │
    # └───────────────────────────────────────────────┘
    # Loop EPOCHS times
    for epoch in range(EPOCHS):

        # Set the LoRA Model in Training Mode
        model.train()

        total_train_loss = 0
        train_correct_tokens = 0
        train_total_tokens = 0
        
        # Init Progress Bar
        dataset_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in dataset_pbar:

            # Reset Gradients
            optimizer.zero_grad()
            
            # Move inputs to GPU
            soft_prompts = batch["soft_prompts"].to(DEVICE, dtype=DTYPE)                # (batch_size, soft_prompt_len, embed_dim)
            input_ids = batch["input_ids"].to(DEVICE)                                   # (batch_size, seq_len)
            attention_mask = batch["attention_mask"].to(DEVICE)                         # (batch_size, seq_len)
            labels = batch["labels"].to(DEVICE)                                         # (batch_size, seq_len)
            
            # Get embeddings for the discrete text
            with torch.no_grad():
                text_embeds = llama_word_embeddings(input_ids).detach()                 # (batch_size, seq_len, embed_dim)
            
            # Concatenate Embeddings: [Continuous Soft Prompt + Discrete Hard Prompt]
            inputs_embeds = torch.cat([soft_prompts, text_embeds], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
            
            # Concatenate Attention Masks (Add `1`s for the soft prompt so Llama Model pays attention to it)
            soft_prompt_mask = torch.ones((soft_prompts.shape[0], NUM_TOKENS), dtype=attention_mask.dtype, device=DEVICE)   # (batch_size, soft_prompt_len)
            full_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)           # (batch_size, soft_prompt_len + seq_len)
            
            # Concatenate Labels (-100s for the soft prompt so loss isn't calculated on it)
            soft_prompt_labels = torch.full((labels.shape[0], NUM_TOKENS), -100, dtype=labels.dtype, device=DEVICE)         # (batch_size, soft_prompt_len)
            full_labels = torch.cat([soft_prompt_labels, labels], dim=1)                         # (batch_size, soft_prompt_len + seq_len)
            
            # Forward Pass
            outputs = model(
                inputs_embeds = inputs_embeds,
                attention_mask = full_attention_mask,
                labels = full_labels
            )
            
            # Extract the CE Loss
            loss = outputs.loss

            # Backpropagate Loss and Update the Parameters
            loss.backward()
            optimizer.step()

            # Extract the logits
            logits = outputs.logits

            # Shift logits and labels so token i predicts i + 1
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = full_labels[..., 1:].contiguous()

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
            dataset_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = (train_correct_tokens / train_total_tokens) * 100 if train_total_tokens > 0 else 0
        
        # ┌───────────────────────────────────────────────┐
        # │                 VALIDATION LOOP               │
        # └───────────────────────────────────────────────┘
        model.eval()
        total_val_loss = 0
        val_correct_tokens = 0
        val_total_tokens = 0
        
        # Freeze all weights
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):

                # Move inputs to DEVICE
                soft_prompts = batch["soft_prompts"].to(DEVICE, dtype=DTYPE)            # (batch_size, soft_prompt_len, embed_dim)
                input_ids = batch["input_ids"].to(DEVICE)                               # (batch_size, seq_len)
                attention_mask = batch["attention_mask"].to(DEVICE)                     # (batch_size, seq_len)
                labels = batch["labels"].to(DEVICE)                                     # (batch_size, seq_len)
                
                # Get the text embeddings from Llama for the soft prompt token ids
                text_embeds = llama_word_embeddings(input_ids).detach()                 # (batch_size, seq_len, embed_dim)

                # Concatenate Embeddings: [Soft Prompt + Input Text]
                inputs_embeds = torch.cat([soft_prompts, text_embeds], dim=1)           # (batch_size, soft_prompt_len + seq_len, embed_dim)
                
                # Concatenate Attention Masks: Add `1`s so Llama pays attention to the soft prompt
                soft_prompt_mask = torch.ones((soft_prompts.shape[0], NUM_TOKENS), dtype=attention_mask.dtype, device=DEVICE)   # (batch_size, soft_prompt_len)
                full_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)  # (batch_size, soft_prompt_len + seq_len)
                
                # Concatenate Labels (-100s for the soft prompt so loss isn't calculated on it)
                soft_prompt_labels = torch.full((labels.shape[0], NUM_TOKENS), -100, dtype=labels.dtype, device=DEVICE)         # (batch_size, soft_prompt_len)
                full_labels = torch.cat([soft_prompt_labels, labels], dim=1)                # (batch_size, soft_prompt_len + seq_len)
                
                # Forward Pass
                outputs = model(
                    inputs_embeds = inputs_embeds,
                    attention_mask = full_attention_mask,
                    labels = full_labels
                )

                # Accumulate validation loss
                total_val_loss += outputs.loss.item()

                # Calculate Validation Accuracy
                logits = outputs.logits
                shifted_logits = logits[..., :-1, :].contiguous()
                shifted_labels = full_labels[..., 1:].contiguous()

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
    # │               SAVE LORA ADAPTERS              │
    # └───────────────────────────────────────────────┘
    lora_weights_save_dir = os.path.join(SAVE_DIR, DB_NAME)
    os.makedirs(lora_weights_save_dir, exist_ok=True)
    model.save_pretrained(lora_weights_save_dir)
    tokenizer.save_pretrained(lora_weights_save_dir)
    print(f"Mapper training complete! PEFT LoRA weights saved to {lora_weights_save_dir}")