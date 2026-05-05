import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import evaluate

# PyTorch Dataset wrapper on the Mapper Dataset from Soft Prompts to Hard Prompts
class MapperDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Return the (20, 4096) tensor and the target string
        return (
            self.data[idx]["soft_prompt"], 
            self.data[idx]["hard_prompt"], 
            self.data[idx]["soft_prompt_init_embeddings"],
            self.data[idx]["instances"]
        )


# Custom Data Collator for the Mapper Dataset
class MapperCollator:
    def __init__(self, tokenizer, soft_prompt_length=20):
        self.tokenizer = tokenizer
        self.soft_prompt_length = soft_prompt_length

    def __call__(self, batch):
        # Retrieve list of soft_prompts and hard_prompts (in that order)
        soft_prompts, hard_prompts, softprompt_init, instances = zip(*batch)
        
        # Stack the frozen soft prompts into a batch: (batch_size, soft_prompt_len, embed_dim)
        soft_prompts = torch.stack(soft_prompts)        # (batch_size, soft_prompt_len, embed_dim)
        softprompt_init = torch.stack(softprompt_init)        # (batch_size, soft_prompt_len, embed_dim)

        # Explicitly append the EOS token so the model learns when to stop
        hard_prompts = [prompt + self.tokenizer.eos_token for prompt in hard_prompts]
    
        # Tokenize the target hard prompts
        tokenized = self.tokenizer(
            hard_prompts, 
            padding=True, 
            truncation=True, 
            max_length=300, # TODO: Test this value
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
            "labels": labels,
            "init":softprompt_init,
            "instances":instances
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
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--mapper_dataset_path", type=str, default="./datasets/mapper_training_dataset/SUPER-NATURALINSTRUCTIONS-english-filtered_original_instructions")
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

    # Init Rouge Metric
    ROUGE_METRIC = evaluate.load("rouge")

    # ┌───────────────────────────────────────────────┐
    # │                 LORA MODEL PREP               │
    # └───────────────────────────────────────────────┘
    print(f"Loading base model {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    base_model.gradient_checkpointing_enable()

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

    with torch.no_grad():
        SOFT_MARKER = llama_word_embeddings(tokenizer("<SOFT:>", add_special_tokens=False, return_tensors='pt').to(DEVICE)['input_ids']).detach()
        HARD_MARKER = llama_word_embeddings(tokenizer("<HARD:>", add_special_tokens=False, return_tensors='pt').to(DEVICE)['input_ids']).detach()
        INIT_MARKER = llama_word_embeddings(tokenizer("<INIT:>", add_special_tokens=False, return_tensors='pt').to(DEVICE)['input_ids']).detach()

    def soft_to_hard(
        soft_prompts,
        attention_mask,
        labels,
        init,
        soft_marker,
        hard_marker,
        init_marker,
        text_embeds,
        batchsize,
    ):
        """
            Computes -log p(hard|soft)
            Input sequence looks like
        """
        # build sequence (init + soft + hard)
        inputs_embeds = torch.cat([init_marker, init, soft_marker, soft_prompts, hard_marker, text_embeds], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        prefix_len = init_marker.shape[1] + init.shape[1] + soft_marker.shape[1] + soft_prompts.shape[1] + hard_marker.shape[1]

        # build sequence (soft + hard)
        # inputs_embeds = torch.cat([soft_marker, soft_prompts, hard_marker, text_embeds], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        # prefix_len = soft_marker.shape[1] + soft_prompts.shape[1] + hard_marker.shape[1]

        # Concatenate Attention Masks (Add `1`s for the soft prompt so Llama Model pays attention to it)
        soft_prompt_mask = torch.ones((batchsize, prefix_len), dtype=attention_mask.dtype, device=DEVICE)   # (batch_size, soft_prompt_len)
        full_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)           # (batch_size, soft_prompt_len + seq_len)
        
        # Concatenate Labels (-100s for the soft prompt so loss isn't calculated on it)
        soft_prompt_labels = torch.full((batchsize, prefix_len), -100, dtype=labels.dtype, device=DEVICE)         # (batch_size, soft_prompt_len)
        full_labels = torch.cat([soft_prompt_labels, labels], dim=1)                         # (batch_size, soft_prompt_len + seq_len)
        
        # Forward Pass
        outputs = model(
            inputs_embeds = inputs_embeds,
            attention_mask = full_attention_mask,
            labels = full_labels
        )

        soft_to_hard_loss = outputs.loss

        #========================= rouge L eval ================================
        # TEACHER FORCING
        # Soft to hard ROUGE-L
        # Extract the logits
        logits = outputs.logits

        # Shift logits and labels so token i predicts i + 1
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = full_labels[..., 1:].contiguous()

        # Get the predicted token ids
        preds = torch.argmax(shifted_logits, dim = -1)

        # Replace -100 in the labels as we can't decode -100
        shifted_labels = torch.where(
            shifted_labels != -100, 
            shifted_labels, 
            tokenizer.pad_token_id
        )

        # Decode preds and labels into strings
        # skip_special_tokens=True removes EOS and Padding tokens from the text
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(shifted_labels, skip_special_tokens=True)

        # Calculate ROUGE-L for the current batch
        rouge_results = ROUGE_METRIC.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            use_stemmer=True
        )
        
        return soft_to_hard_loss, rouge_results

    def hard_to_soft(
        soft_prompts,
        attention_mask,
        labels,
        init,
        soft_marker,
        hard_marker,
        init_marker,
        text_embeds,
        batchsize,
    ):
        # build sequence (init + hard + soft)
        inputs_embeds = torch.cat([init_marker, init, hard_marker, text_embeds, soft_marker, soft_prompts], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        prefix_len = init_marker.shape[1] + init.shape[1] + hard_marker.shape[1]
        
        # build sequence (hard + soft)
        # inputs_embeds = torch.cat([hard_marker, text_embeds, soft_marker, soft_prompts], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        # prefix_len = hard_marker.shape[1] + soft_marker.shape[1]

        soft_len = soft_marker.shape[1] + soft_prompts.shape[1]

        # Concatenate Attention Masks (Add `1`s for the soft prompt so Llama Model pays attention to it)
        prefix_mask = torch.ones((batchsize, prefix_len), dtype=attention_mask.dtype, device=DEVICE)  
        softprompt_mask = torch.ones((batchsize, soft_len), dtype=attention_mask.dtype, device=DEVICE)   
        full_attention_mask = torch.cat([prefix_mask, attention_mask, softprompt_mask], dim=1)           
             
        # Forward Pass
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]  # (B, T, D)

        # predictions (shifted left)
        pred = outputs[:, :-1, :]              # (B, T-1, D)

        # take only positions that predict soft prompt
        pred_soft = pred[:, -soft_prompts.shape[1]:, :]     # (B, k, D)
        hard_to_soft_loss = torch.mean((soft_prompts - pred_soft)**2)
        
        return hard_to_soft_loss
    
    def generate_hard_from_soft(
        soft_prompts,
        init,
        soft_marker,
        hard_marker,
        init_marker,
        batchsize,
    ):
        # build input sequence (init + soft)
        inputs_embeds = torch.cat([init_marker, init, soft_marker, soft_prompts, hard_marker], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        prefix_len = init_marker.shape[1] + init.shape[1] + soft_marker.shape[1] + soft_prompts.shape[1] + hard_marker.shape[1]
        
        # Concatenate Attention Masks (Add `1`s for the soft prompt so Llama Model pays attention to it)
        attention_mask = torch.ones((batchsize, prefix_len), dtype=DTYPE, device=DEVICE)   # (batch_size, soft_prompt_len)
        
        # Generate the predicted tokens for the whole batch
        hard_hat_idxs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        hard_hat_attn_mask = (hard_hat_idxs != tokenizer.pad_token_id).long()
        hard_hat_embeds = llama_word_embeddings(hard_hat_idxs).detach()  

        return hard_hat_embeds, hard_hat_attn_mask
    
    def generate_soft_from_hard(
        text_embeds,
        attention_mask,
        init,
        soft_marker,
        hard_marker,
        init_marker,
        batchsize,
    ):
        # build input sequence (init + soft)
        inputs_embeds = torch.cat([init_marker, init, hard_marker, text_embeds, soft_marker], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        prefix_len = init_marker.shape[1] + init.shape[1] + hard_marker.shape[1]
        soft_len = soft_marker.shape[1]
        
        # Concatenate Attention Masks 
        prefix_mask = torch.ones((batchsize, prefix_len), dtype=attention_mask.dtype, device=DEVICE)  
        softprompt_mask = torch.ones((batchsize, soft_len), dtype=attention_mask.dtype, device=DEVICE)   
        full_attention_mask = torch.cat([prefix_mask, attention_mask, softprompt_mask], dim=1)           
        
        # Manually auto regress generate a soft prompt
        for _ in range(soft_len):
            output = model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                use_cache=True
            ).hidden_states[-1]  # (B, T, D)

            pred = output[:,-1,:].unsqueeze(1)
            inputs_embeds = torch.cat([inputs_embeds, pred], dim=1)

            new_prompt_token_mask = torch.ones((batchsize, 1), dtype=DTYPE, device=DEVICE)
            full_attention_mask = torch.cat([full_attention_mask, new_prompt_token_mask], dim=1)

        soft_hat_embeds = inputs_embeds[:, -soft_prompts.shape[1]:, :]     # (B, k, D)

        return soft_hat_embeds

    def prep_instances(instances):
        full_text = [f"{instance["input"]}{instance["output"]}" for instance in instances]
        tokenized = tokenizer(
            full_text, 
            padding='True', 
            truncation=True,
            max_length=512, 
            return_tensors="pt",
            add_special_tokens=True
        )    

        input_ids = tokenized["input_ids"]                              # (batch_size, seq_len)
        attention_mask = tokenized["attention_mask"]                    # (batch_size,)

        # Create the labels tensor
        labels = input_ids.clone()                                      # (batch_size, seq_len)
        
        # Mask out the input text and the padding tokens with -100
        for i, instance in instances:
            
            # Tokenize just the input text to find out how long it is
            inp_len = len(tokenizer.encode(instance["input"], add_special_tokens=True))

            # Mask the input portion so loss is not calculated on it
            labels[i, :inp_len] = -100

            # Mask any padding tokens added to the end of the sequence
            labels[i, attention_mask[i] == 0] = -100
        
        return input_ids, attention_mask, labels
        

    def eval_softprompt_on_instances(soft_prompt, instance_input_ids, instance_attn_mask, instance_labels):
        batchsize = instance_labels.shape[0]
        softprompt_len = soft_prompt.shape[1]

        soft_prompt_labels = torch.full((batchsize,softprompt_len), -100, dtype=torch.long, device=DEVICE)  # (batch_size, soft_prompt_len)
        full_labels = torch.cat([soft_prompt_labels, instance_labels], dim=1)                                         # (batch_size, soft_prompt_len + seq_len)

        soft_prompt_attn_mask = torch.ones((batchsize, softprompt_len), dtype=instance_attn_mask.dtype, device=DEVICE)
        full_attn_mask = torch.cat([soft_prompt_attn_mask, instance_attn_mask], dim=1)

        input_embeds = llama_word_embeddings(instance_input_ids)
        full_embeds = torch.cat([soft_prompt, input_embeds], dim=1)

        loss = model(
            inputs_embeds=full_embeds,
            attention_mask=full_attn_mask,
            labels=full_labels
        ).loss

        return loss


    # ┌───────────────────────────────────────────────┐
    # │                 TRAINING LOOP                 │
    # └───────────────────────────────────────────────┘

    NUM_EPOCHS_TO_PRETRAIN = 2
    NUM_EPOCHS_TO_BACKTRANSLATE = 1
    # Loop EPOCHS times
    for epoch in range(EPOCHS):

        # Set the LoRA Model in Training Mode
        model.train()

        total_train_loss = 0
        total_train_rouge_l = 0
        
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
            init = batch["init"].to(DEVICE)
            
            # Get embeddings for the discrete text
            with torch.no_grad():
                text_embeds = llama_word_embeddings(input_ids).detach()                 # (batch_size, seq_len, embed_dim)
            
            batchsize = soft_prompts.shape[0]
            soft_marker = SOFT_MARKER.expand(batchsize, -1, -1)
            hard_marker = HARD_MARKER.expand(batchsize, -1, -1)
            init_marker = INIT_MARKER.expand(batchsize, -1, -1)
            
            # ====================
            # PHASE 1: Pretraining
            # ====================
            soft_to_hard_loss, hard_to_soft_loss, bt_soft_to_hard_loss, bt_hard_to_soft_loss = 0., 0., 0., 0.
            if epoch < NUM_EPOCHS_TO_PRETRAIN:
                soft_to_hard_loss, rouge_results = soft_to_hard(
                    soft_prompts,
                    attention_mask,
                    labels,
                    init,
                    soft_marker,
                    hard_marker,
                    init_marker,
                    text_embeds,
                    batchsize,
                )
                hard_to_soft_loss = hard_to_soft(
                    soft_prompts,
                    attention_mask,
                    labels,
                    init,
                    soft_marker,
                    hard_marker,
                    init_marker,
                    text_embeds,
                    batchsize,
                )
                loss = soft_to_hard_loss + hard_to_soft_loss
            # =========================
            # PHASE 2: Backtranslations
            # =========================
            elif epoch < NUM_EPOCHS_TO_PRETRAIN + NUM_EPOCHS_TO_BACKTRANSLATE:
                soft_to_hard_loss, rouge_results = soft_to_hard(
                    soft_prompts,
                    attention_mask,
                    labels,
                    init,
                    soft_marker,
                    hard_marker,
                    init_marker,
                    text_embeds,
                    batchsize,
                )
                hard_to_soft_loss = hard_to_soft(
                    soft_prompts,
                    attention_mask,
                    labels,
                    init,
                    soft_marker,
                    hard_marker,
                    init_marker,
                    text_embeds,
                    batchsize,
                )
                with torch.no_grad():
                    hard_hat, hard_attn_mask = generate_hard_from_soft(
                        soft_prompts, 
                        init,
                        soft_marker,
                        hard_marker,
                        init_marker,
                        batchsize
                    )
                    soft_hat = generate_soft_from_hard(
                        text_embeds,
                        attention_mask,
                        init,
                        soft_marker,
                        hard_marker,
                        init_marker,
                        batchsize
                    )
                bt_soft_to_hard_loss, _ = soft_to_hard(
                    soft_hat.detach(),
                    attention_mask,
                    labels,
                    init,
                    soft_marker,
                    hard_marker,
                    init_marker,
                    text_embeds,
                    batchsize,
                )
                bt_hard_to_soft_loss = hard_to_soft(
                    soft_prompts,
                    hard_attn_mask.detach(),
                    labels,
                    init,
                    soft_marker,
                    hard_marker,
                    init_marker,
                    hard_hat.detach(),
                    batchsize,
                )
                loss = soft_to_hard_loss + hard_to_soft_loss + bt_soft_to_hard_loss + bt_hard_to_soft_loss
            # =================
            # PHASE 3: Finetune
            # =================
            else:
                soft_to_hard_loss, rouge_results = soft_to_hard(
                    soft_prompts,
                    attention_mask,
                    labels,
                    init,
                    soft_marker,
                    hard_marker,
                    init_marker,
                    text_embeds,
                    batchsize,
                )
                loss = soft_to_hard_loss + hard_to_soft_loss
            
            # Backpropagate Loss and Update the Parameters
            loss.backward()
            optimizer.step()

            # Accumulate metrics (initialize `total_train_rouge_l = 0` before the epoch instead of tokens)
            current_rouge_l = rouge_results['rougeL']
            total_train_rouge_l += current_rouge_l
            total_train_loss += loss.item()
            
            # Update progress bar
            dataset_pbar.set_postfix({
                "Loss": f"{loss.item():.2f}",
                "S->H": f"{soft_to_hard_loss:.2f}",
                "H->S": f"{hard_to_soft_loss:.2f}",
                "S->H (BT)": f"{bt_soft_to_hard_loss:.2f}",
                "H->S (BT)": f"{bt_hard_to_soft_loss:.2f}",
            })

            
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_rouge_l = total_train_rouge_l / len(train_dataloader)
        
        # ┌───────────────────────────────────────────────┐
        # │                 VALIDATION LOOP               │
        # └───────────────────────────────────────────────┘
        model.eval()
        total_val_loss = 0
        total_val_rouge_l = 0
        
        # Freeze all weights
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):

                # Move inputs to DEVICE
                soft_prompts = batch["soft_prompts"].to(DEVICE, dtype=DTYPE)                # (batch_size, soft_prompt_len, embed_dim)
                input_ids = batch["input_ids"].to(DEVICE)                                   # (batch_size, seq_len)
                attention_mask = batch["attention_mask"].to(DEVICE)                         # (batch_size, seq_len)
                labels = batch["labels"].to(DEVICE)                                         # (batch_size, seq_len)
                init = batch["init"].to(DEVICE)
                batch_instances = batch["instances"].to(DEVICE)
                
                # Get the text embeddings from Llama for the soft prompt token ids
                text_embeds = llama_word_embeddings(input_ids).detach()                 # (batch_size, seq_len, embed_dim)

                # "<SOFT:>" + softprompt + "<HARD>:" + hardprompt
                batchsize = soft_prompts.shape[0]
                soft_marker = SOFT_MARKER.expand(batchsize, -1, -1)
                hard_marker = HARD_MARKER.expand(batchsize, -1, -1)
                init_marker = INIT_MARKER.expand(batchsize, -1, -1)

                soft_to_hard_loss, rouge_results = soft_to_hard(
                    soft_prompts,
                    attention_mask,
                    labels,
                    init,
                    soft_marker,
                    hard_marker,
                    init_marker,
                    text_embeds,
                    batchsize,
                )
                loss = soft_to_hard_loss

                pred_soft_prompts = generate_soft_from_hard(
                    text_embeds,
                    attention_mask,
                    init,
                    soft_marker,
                    hard_marker,
                    init_marker,
                    batchsize
                )
                for i, instances in enumerate(batch_instances):
                    instance_input_ids, instance_attn_mask, instance_labels = prep_instances(instances)

                    gt_soft_prompt = soft_prompts[i].unsqueeze(0)
                    pred_soft_prompt = pred_soft_prompts[i].unsqueeze(0)

                    groundtruth_soft_performance = eval_softprompt_on_instances(
                        gt_soft_prompt,
                        instance_input_ids,
                        instance_attn_mask,
                        instance_labels
                    )

                    pred_soft_performance = eval_softprompt_on_instances(
                        pred_soft_prompt,
                        instance_input_ids,
                        instance_attn_mask,
                        instance_labels
                    )
                
                # Accumulate validation loss
                total_val_loss += soft_to_hard_loss.item()

                # Accumulate metrics (initialize `total_train_rouge_l = 0` before the epoch instead of tokens)
                current_rouge_l = rouge_results['rougeL']
                total_val_rouge_l += current_rouge_l
                
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_rouge_l = total_val_rouge_l / len(val_dataloader)

        tqdm.write(f"\nEpoch {epoch + 1} Summary:")
        tqdm.write(f"Train -> Loss: {avg_train_loss: .4f} | ROUGE-L: {avg_train_rouge_l: .2f}")
        tqdm.write(f"Val   -> Loss: {avg_val_loss: .4f} | ROUGE-L: {avg_val_rouge_l: .2f}\n")

    # ┌───────────────────────────────────────────────┐
    # │               SAVE LORA ADAPTERS              │
    # └───────────────────────────────────────────────┘
    lora_weights_save_dir = os.path.join(SAVE_DIR, DB_NAME)
    os.makedirs(lora_weights_save_dir, exist_ok=True)
    model.save_pretrained(lora_weights_save_dir)
    tokenizer.save_pretrained(lora_weights_save_dir)
    print(f"Mapper training complete! PEFT LoRA weights saved to {lora_weights_save_dir}")