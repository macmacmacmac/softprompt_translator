from peft import PromptTuningInit, PromptTuningConfig, TaskType, get_peft_model
from transformers import default_data_collator, get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def run(args_list=None):
    # =============================================
    # Basic housekeeping jsut setting up peft and dataset 
    # =============================================
    torch.manual_seed(0)

    # load basics
    MODEL_NAME = "meta-llama/Llama-3.1-8b-Instruct"
    NUM_TOKENS = 20

    DTYPE = torch.bfloat16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    word_embeddings = base_model.get_input_embeddings()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare Soft Prompt Model
    init_text = "1 2 3 4 5 6 7 8"
    init_ids = tokenizer.encode(init_text, return_tensors='pt', add_special_tokens=False).to(DEVICE)
    num_tokens = init_ids.shape[1]
    init_embeds = word_embeddings(init_ids).to(DEVICE).to(DTYPE)
    cust_softprompt = nn.Parameter(init_embeds)

    prompt_tuning_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text=init_text,
        num_virtual_tokens=num_tokens,
        tokenizer_name_or_path=MODEL_NAME
    )
    peft_softprompt_model = get_peft_model(base_model, prompt_tuning_config)


    training_samples = [
        ("Code phrase 1 is:","The quick brown fox jumped over the lazer dog"),
        ("Code phrase 2 is:","Does the black moon howl?"),
        ("Code phrase 3 is:","Funnel ferry butterbar"),
        ("Code phrase 4 is:","Sphinx of black quarts judge my vow"),
        ("Code phrase 5 is:","Lorem ipsum dolor sit amet, consectetur adipiscing elit"),
        ("Code phrase 6 is:","Pack my box with five dozen liquor jugs"),
    ]
    MAX_LENGTH = 128

    def tokenize_sample(sample):
        prompt, target = sample

        # Full sequence fed to model
        full_text = prompt + " " + target

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        # Prompt-only tokens (used to mask labels)
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = full_tokens["input_ids"][0]
        attention_mask = full_tokens["attention_mask"][0]

        labels = input_ids.clone()

        # Ignore prompt portion in loss
        prompt_len = prompt_tokens["input_ids"].shape[1]

        labels[:prompt_len] = -100

        # Ignore padding tokens in loss
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


    tokenized_dataset = [tokenize_sample(x) for x in training_samples]
    BATCH_SIZE = 3

    train_loader = DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    

    # ============================================================
    # Training Loop USING PEFT
    # ============================================================
    optimizer = torch.optim.AdamW(
        peft_softprompt_model.parameters(),
        lr=1e-3,
    )
    peft_softprompt_model.train()

    NUM_EPOCHS = 5

    for epoch in range(NUM_EPOCHS):

        total_loss = 0.0

        for step, batch in enumerate(train_loader):

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = peft_softprompt_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"Epoch {epoch+1} | Step {step+1} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Complete")
        print(f"Average Loss: {avg_loss:.4f}")

    peft_softprompt_model.eval()

    test_prompt = "Code phrase 3 is:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated = peft_softprompt_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id
        )
        logits = peft_softprompt_model(**inputs).logits
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

    print("="*30, "THIS IS USING LLM_peft(X)", "="*30)
    print("---Decoded generation example: \n", decoded)
    print("---Logits: \n",logits)

    # ============================================================
    # Training Loop USING cust
    # ============================================================
    
    base_model = peft_softprompt_model.base_model.to(DTYPE)

    NUM_EPOCHS = 5

    for epoch in range(NUM_EPOCHS):
        optimizer = torch.optim.AdamW(
            cust_softprompt,
            lr=1e-3,
        )
        total_loss = 0.0

        for step, batch in enumerate(train_loader):

            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            input_embeds = word_embeddings(input_ids)
            soft_embeds = cust_softprompt.expand(input_ids.shape[0],-1,-1)
            full_embeds = torch.cat([soft_embeds,input_embeds], dim=1).to(DEVICE).to(DTYPE)
            prefix_attn_mask = torch.ones(soft_embeds.shape[:2]).to(DEVICE).to(DTYPE)
            full_attn_mask = torch.cat([prefix_attn_mask, attn_mask], dim=-1).to(DEVICE)

            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"Epoch {epoch+1} | Step {step+1} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Complete")
        print(f"Average Loss: {avg_loss:.4f}")

    # =============================================
    # Comparison between peft vs custom 
    # =============================================

    softprompt_embeddings = cust_softprompt
    input_embeds = word_embeddings(inputs['input_ids']).to(DEVICE, dtype=DTYPE)
    full_embeds = torch.cat([softprompt_embeddings, input_embeds],dim=1).to(DTYPE)
    full_attn_mask = torch.cat([
        torch.ones(softprompt_embeddings.shape[:2]).to(DEVICE, dtype=DTYPE), 
        inputs['attention_mask']], 
        dim=1
    ).to(DTYPE)
    with torch.no_grad():
        cust_generated = base_model.generate(
            inputs_embeds = full_embeds,
            attention_mask = full_attn_mask,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_new_tokens=30,
        )
        cust_logits = base_model(
            inputs_embeds = full_embeds,
            attention_mask = full_attn_mask
        ).logits
    cust_decoded = tokenizer.decode(cust_generated[0], skip_special_tokens=True)

    print("="*30, "THIS IS USING LLM(X|Z)", "="*30)
    print("---Decoded generation example: \n", cust_decoded)
    print("---Logits: \n",cust_logits)

