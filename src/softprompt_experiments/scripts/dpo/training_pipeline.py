import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import evaluate
import ipdb
import torch.nn.functional as F

from softprompt_experiments.scripts.dpo import DPOCollator

# PyTorch Dataset wrapper on the Mapper Dataset from Soft Prompts to Hard Prompts
class DPODataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Return the (20, 4096) tensor and the target string
        return (
            self.data[idx]["z_prime"], 
            self.data[idx]["z_W"],
            self.data[idx]["z_L"],
            self.data[idx]["logp_ref_z_W"],
            self.data[idx]["logp_ref_z_L"],
        )
    

def get_logprob(soft_prompts, tokenized, model, llama_word_embeddings):

    # Unpack tokenized
    input_ids = tokenized["input_ids"].to(DEVICE)                                   # (batch_size, seq_len)
    attention_mask = tokenized["attention_mask"].to(DEVICE)                         # (batch_size, seq_len)
    labels = tokenized["labels"].to(DEVICE)                                         # (batch_size, seq_len)
    
    # get dims
    batchsize, softlen, _ = soft_prompts.shape

    # Get embeddings for the discrete text
    with torch.no_grad():
        text_embeds = llama_word_embeddings(input_ids).detach()                 # (batch_size, seq_len, embed_dim)
    
    # Concatenate Embeddings: [Continuous Soft Prompt + Discrete Hard Prompt]
    inputs_embeds = torch.cat([soft_prompts, text_embeds], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
    
    # Concatenate Attention Masks (Add `1`s for the soft prompt so Llama Model pays attention to it)
    soft_prompt_mask = torch.ones((batchsize, softlen), dtype=attention_mask.dtype, device=DEVICE)   # (batch_size, soft_prompt_len)
    full_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)           # (batch_size, soft_prompt_len + seq_len)
    
    # Concatenate Labels (-100s for the soft prompt so loss isn't calculated on it)
    soft_prompt_labels = torch.full((batchsize, softlen), -100, dtype=labels.dtype, device=DEVICE)         # (batch_size, soft_prompt_len)
    full_labels = torch.cat([soft_prompt_labels, labels], dim=1)                         # (batch_size, soft_prompt_len + seq_len)
    
    # Forward Pass
    logits = model( 
        inputs_embeds = inputs_embeds,
        attention_mask = full_attention_mask,
        labels = full_labels
    ).logits # [B, T, V]

    vocab_size = logits.shape[-1]

    # Get log probs
    losses = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size),
        full_labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    # (batch_size, seq_len + softlen - 1)
    losses = losses.view(batchsize, -1)

    # Sum over tokens to get one log-prob per sequence
    log_prob = -losses.sum(dim=1)

    # ipdb.set_trace()

    return log_prob


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
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mapper_dataset_path", type=str, default="./shared/datasets/dpo_preference_datasets/ROUGE-Lscore_10n_1k_0.5temp_01")
    parser.add_argument("--mapper_weights_dir", type=str, default="./shared/mapper_lora_weights_overfit/General-DoD-10x/meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dpo_save_dir", type=str, default="./shared/mapper_lora_weights")
    parser.add_argument("--optim_weight_decay", type=float, default=0.1) 
    parser.add_argument("--beta", type=float, default=0.1) 
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = args.model_name
    MAPPER_DATASET_PATH = args.mapper_dataset_path
    MAPPER_WEIGHTS_PATH = args.mapper_weights_dir
    DPO_MAPPER_SAVE_PATH = f"{args.dpo_save_dir}/DPO_rougeL_{args.beta}/meta-llama/Llama-3.1-8B-Instruct"
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    OPTIM_WEIGHT_DECAY = args.optim_weight_decay
    BETA = args.beta

    # Determine DEVICE and DTYPE
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token       # Llama doesn't have a default pad token, so we map it to EOS

    # ┌───────────────────────────────────────────────┐
    # │                 LORA MODEL PREP               │
    # └───────────────────────────────────────────────┘
    print(f"Loading base model {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    base_model.gradient_checkpointing_enable()

    print(f"Loading LoRA adapter from {MAPPER_WEIGHTS_PATH}...")
    model = PeftModel.from_pretrained(
        base_model,
        MAPPER_WEIGHTS_PATH,
        is_trainable=True,   # important: continue training
    )    

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
    train_dataset = torch.load(os.path.join(MAPPER_DATASET_PATH, 'train_dataset.pt'), map_location="cpu", weights_only=True)

    # train_dataset2 = torch.load(os.path.join("./shared/datasets/dpo_preference_datasets/LOGPROBscore_10n_1k_1.0temp", 'train_dataset.pt'), map_location="cpu", weights_only=True)
    # train_dataset = train_dataset + train_dataset2

    val_dataset = torch.load(os.path.join(MAPPER_DATASET_PATH, 'val_dataset.pt'), map_location="cpu", weights_only=True)
    
    print(f"Train Dataset size: {len(train_dataset)} | Validation Dataset size: {len(val_dataset)}")

    # Init Collator
    collator = DPOCollator(
        tokenizer = tokenizer, 
    )

    # Init Training Dataloader
    train_dataloader = DataLoader(
        DPODataset(train_dataset),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator
    )

    # Init Validation Dataloader
    val_dataloader = DataLoader(
        DPODataset(val_dataset), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collator
    )


    # ┌───────────────────────────────────────────────┐
    # │                 TRAINING LOOP                 │
    # └───────────────────────────────────────────────┘
    # Loop EPOCHS times
    best_val_loss = 1000
    for epoch in range(EPOCHS):

        # Set the LoRA Model in Training Mode
        model.train()

        total_train_loss = 0
        
        # Init Progress Bar
        dataset_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in dataset_pbar:

            # Reset Gradients
            optimizer.zero_grad()
            
            # Move inputs to GPU
            # "z_prime": soft prompt tensors (batch, seq_len, emb_dim)
            # "z_W_tokenized": z_W_tokenized {'input_ids', 'attention_mask', 'labels'}
            # "z_L_tokenized": z_L_tokenized {'input_ids', 'attention_mask', 'labels'}
            # "log_p_ref_z_W": logp_ref_z_W (batch)
            # "log_p_ref_z_L": logp_ref_z_L (batch)

            z_prime = batch["z_prime"].to(DEVICE, dtype=DTYPE) # (batch_size, soft_prompt_len, embed_dim)
           
            z_W_tokenized = batch["z_W_tokenized"].to(DEVICE)
            z_L_tokenized = batch["z_L_tokenized"].to(DEVICE)

            log_p_ref_z_W = batch["log_p_ref_z_W"].to(DEVICE, dtype=DTYPE)
            log_p_ref_z_L = batch["log_p_ref_z_L"].to(DEVICE, dtype=DTYPE)

            log_p_theta_z_W = get_logprob(z_prime, z_W_tokenized, model, llama_word_embeddings)
            log_p_theta_z_L = get_logprob(z_prime, z_L_tokenized, model, llama_word_embeddings)

            # ipdb.set_trace()

            # DPO
            loss = -F.logsigmoid(
                BETA * (
                    (log_p_theta_z_W - log_p_ref_z_W) 
                    - (log_p_theta_z_L - log_p_ref_z_L)
                )
            ).mean()

            total_train_loss += loss.item()

            # Backpropagate Loss and Update the Parameters
            loss.backward()
            optimizer.step()
        
            # Update progress bar
            dataset_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # ┌───────────────────────────────────────────────┐
        # │                 VALIDATION LOOP               │
        # └───────────────────────────────────────────────┘
        model.eval()
        total_val_loss = 0

        # Freeze all weights
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):                
                # Move inputs to GPU
                # "z_prime": soft prompt tensors (batch, seq_len, emb_dim)
                # "z_W_tokenized": z_W_tokenized {'input_ids', 'attention_mask', 'labels'}
                # "z_L_tokenized": z_L_tokenized {'input_ids', 'attention_mask', 'labels'}
                # "log_p_ref_z_W": logp_ref_z_W (batch)
                # "log_p_ref_z_L": logp_ref_z_L (batch)

                z_prime = batch["z_prime"].to(DEVICE, dtype=DTYPE) # (batch_size, soft_prompt_len, embed_dim)
            
                z_W_tokenized = batch["z_W_tokenized"].to(DEVICE)
                z_L_tokenized = batch["z_L_tokenized"].to(DEVICE)

                log_p_ref_z_W = batch["log_p_ref_z_W"].to(DEVICE, dtype=DTYPE)
                log_p_ref_z_L = batch["log_p_ref_z_L"].to(DEVICE, dtype=DTYPE)

                log_p_theta_z_W = get_logprob(z_prime, z_W_tokenized, model, llama_word_embeddings)
                log_p_theta_z_L = get_logprob(z_prime, z_L_tokenized, model, llama_word_embeddings)

                # DPO
                loss = -F.logsigmoid(
                    BETA * (
                        (log_p_theta_z_W - log_p_ref_z_W) 
                        - (log_p_theta_z_L - log_p_ref_z_L)
                    )
                ).mean()

                total_val_loss += loss.item()            
                
        avg_val_loss = total_val_loss / len(val_dataloader)


        tqdm.write(f"\nEpoch {epoch + 1} Summary:")
        tqdm.write(f"Train -> Loss: {avg_train_loss: .4f}")
        tqdm.write(f"Val   -> Loss: {avg_val_loss: .4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss    
            os.makedirs(DPO_MAPPER_SAVE_PATH, exist_ok=True)
            model.save_pretrained(DPO_MAPPER_SAVE_PATH)
            tokenizer.save_pretrained(DPO_MAPPER_SAVE_PATH)
            tqdm.write(f"New best val loss! PEFT LoRA weights saved to {DPO_MAPPER_SAVE_PATH}")

    # ┌───────────────────────────────────────────────┐
    # │               SAVE LORA ADAPTERS              │
    # └───────────────────────────────────────────────┘
    os.makedirs(DPO_MAPPER_SAVE_PATH, exist_ok=True)
    # model.save_pretrained(DPO_MAPPER_SAVE_PATH)
    # tokenizer.save_pretrained(DPO_MAPPER_SAVE_PATH)
    print(f"Mapper training complete! Best val loss: {best_val_loss}. PEFT LoRA weights saved to {DPO_MAPPER_SAVE_PATH}")
