import torch
import argparse
import random
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm

from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.utils import (
    get_train_test_from_softprompt_logits, 
    train_softprompt_from_embeds,
    eval_softprompt,
    log_json
)

import torch.nn.functional as F
import torch.nn as nn

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_datasetv2")
    parser.add_argument("--num_samples_to_eval", type=int, default=100)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--verbose", type=bool, default=False)

    args, _ = parser.parse_known_args(args_list)

    # MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    SAVE_DIR = args.save_directory
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_SAMPLES_TO_EVAL = args.num_samples_to_eval

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype
    ).to(device)
    model.eval()
    word_embeddings = model.get_input_embeddings()

    # Get dataset sub directories
    dataset_dirs = []
    for entry in os.scandir(SAVE_DIR):
        if entry.is_dir():  # Check if the entry is a directory
            if "dataset_" in entry.name:
                dataset_dirs.append(entry.path)

    num_datasets = len(dataset_dirs)
    if num_datasets > 0:
        print(f"\nFound ({num_datasets}) datasets in directory")
    else:
        raise ValueError("path to directory has no datasets")

    # loads in a dataset of trained softprompts
    train_dataset, test_dataset, train_loader, test_loader = get_train_test_from_softprompt_logits(
        model,
        word_embeddings,
        tokenizer,
        dataset_dirs,
        BATCH_SIZE,
        0.8,
    )


    # -----------------------
    # LOAD BASE MODEL
    # -----------------------
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # freeze base model

    # -----------------------
    # TODO: LoRA set up (completed)
    # -----------------------
    # Simple LoRA implementation: a wrapper around an existing nn.Linear that
    # adds a low-rank additive term: W x + alpha / r * (B (A x))
    class LoRALinear(nn.Module):
        def __init__(self, orig_linear: nn.Linear, r: int = 4, alpha: float = 16.0):
            super().__init__()
            # store original linear (frozen)
            self.linear = orig_linear
            # keep original params but freeze them
            for p in self.linear.parameters():
                p.requires_grad = False

            self.in_features = self.linear.in_features
            self.out_features = self.linear.out_features
            self.bias = self.linear.bias is not None

            # LoRA rank and scaling
            self.r = r
            self.alpha = alpha
            self.scaling = self.alpha / max(1, self.r)

            # A: (r, in_features)    -- project input down
            # B: (out_features, r)  -- project back up
            # initialize A small and B zeros (common LoRA init)
            if self.r > 0:
                self.A = nn.Parameter(torch.zeros((self.r, self.in_features), dtype=self.linear.weight.dtype))
                self.B = nn.Parameter(torch.zeros((self.out_features, self.r), dtype=self.linear.weight.dtype))
                # initialize A with kaiming normal scaled
                nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
                # B already zeros (so initial LoRA delta is zero)
            else:
                # r == 0 -> no LoRA
                self.register_parameter("A", None)
                self.register_parameter("B", None)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (..., in_features)
            # base output
            base = F.linear(x, self.linear.weight, self.linear.bias)

            if self.r <= 0:
                return base

            # compute LoRA delta: (x @ A.T) -> (..., r) then @ B.T -> (..., out_features)
            # A: (r, in_features) => A.T: (in_features, r)
            # x @ A.T -> (..., r)
            # then @ B.T (r, out_features) => (..., out_features)
            # multiply by scaling
            # ensure dtype matches linear weight dtype
            xa = torch.matmul(x, self.A.t())
            delta = torch.matmul(xa, self.B.t()) * self.scaling
            return base + delta

        @classmethod
        def from_linear(cls, linear: nn.Linear, r: int = 4, alpha: float = 16.0):
            return cls(linear, r=r, alpha=alpha)

    import math
    import torch.nn.functional as F

    # LoRA hyperparams (you can tweak)
    LORA_R = 16
    LORA_ALPHA = 16.

    # Helper to replace modules in-place by name
    def replace_linear_with_lora(model: nn.Module, r: int, alpha: float):
        """
        Replace ONLY q_proj and v_proj linear layers with LoRALinear wrappers.
        Everything else stays frozen.
        """
        name_to_module = dict(model.named_modules())

        for name, module in list(model.named_modules()):
            # only replace torch.nn.Linear
            if not isinstance(module, nn.Linear):
                continue

            lowered = name.lower()

            # Only patch q_proj and v_proj
            if not ("q_proj" in lowered or "v_proj" in lowered):
                continue

            # Avoid embedding layers or lm_head (safety)
            if "embed" in lowered or "lm_head" in lowered:
                continue

            # Find parent module
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = name_to_module.get(parent_name, None)
            else:
                parent = model
                child_name = name

            if parent is None:
                continue

            orig_linear = getattr(parent, child_name)

            # Avoid double-wrapping in case of re-entry
            if isinstance(orig_linear, LoRALinear):
                continue

            # Wrap with LoRA
            lora_module = LoRALinear.from_linear(
                orig_linear,
                r=r,
                alpha=alpha
            )

            # Ensure device/dtype matches original
            w = orig_linear.weight
            lora_module.to(device=w.device, dtype=w.dtype)

            # Replace in parent
            setattr(parent, child_name, lora_module)

    # run replacement
    replace_linear_with_lora(model, r=LORA_R, alpha=LORA_ALPHA)

    # collect LoRA params (trainable ones)
    lora_parameters = [p for p in model.parameters() if p.requires_grad]
    if len(lora_parameters) == 0:
        raise RuntimeError("No LoRA parameters found to train. Check replacement filters.")

    optimizer = torch.optim.AdamW(lora_parameters, lr=LR, weight_decay=0.1)

    # Suffix to mark end of input
    suffix = " <OUT> "
    suffix_ids = tokenizer(
        suffix,
        add_special_tokens=False,
        return_tensors='pt'
    )['input_ids'].to(model.device)
    SUFFIX_LEN = suffix_ids.shape[1]
    suffix_emb = model.get_input_embeddings()(suffix_ids).to(model.dtype).detach()

    # -----------------------
    # TRAINING LOOP (with test loss logging)
    # -----------------------
    tr_losses = []
    te_losses = []

    for epoch in tqdm(range(EPOCHS)):
        # -------------------
        # Train
        # -------------------
        model.train()     # only LoRA params train; base is frozen
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            softlogits, hardprompt_embeds, tokenized_hardprompt = [b.to(device) for b in batch]
            softlogit_embeds = softlogits @ word_embeddings.weight
            batched_suffixemb = suffix_emb.expand(softlogit_embeds.size(0), -1, -1)

            full_inputs = torch.cat([
                softlogit_embeds.to(model.dtype),
                batched_suffixemb,
                hardprompt_embeds.to(model.dtype)
            ], dim=1)

            labels = torch.cat([
                torch.full((softlogit_embeds.shape[0], softlogit_embeds.shape[1]), -100).to(device),
                torch.full((batched_suffixemb.shape[0], batched_suffixemb.shape[1]), -100).to(device),
                tokenized_hardprompt
            ], dim=1)

            outputs = model(inputs_embeds=full_inputs, labels=labels)

            # outputs = model(inputs_embeds=input_embeds, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        tr_loss = total_loss / len(train_loader)
        tr_losses.append(tr_loss)

        # -------------------
        # Test
        # -------------------
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                softlogits, hardprompt_embeds, tokenized_hardprompt = [b.to(device) for b in batch]
                softlogit_embeds = softlogits @ word_embeddings.weight
                
                batched_suffixemb = suffix_emb.expand(softlogit_embeds.size(0), -1, -1)

                full_inputs = torch.cat([
                    softlogit_embeds.to(model.dtype),
                    batched_suffixemb,
                    hardprompt_embeds.to(model.dtype)
                ], dim=1)

                labels = torch.cat([
                    torch.full((softlogit_embeds.shape[0], softlogit_embeds.shape[1]), -100).to(device),
                    torch.full((batched_suffixemb.shape[0], batched_suffixemb.shape[1]), -100).to(device),
                    tokenized_hardprompt
                ], dim=1)

                outputs = model(inputs_embeds=full_inputs, labels=labels)
                # outputs = model(inputs_embeds=input_embeds, labels=labels)

                total_test_loss += outputs.loss.item()

        te_loss = total_test_loss / len(test_loader)
        te_losses.append(te_loss)

        print(f"Epoch {epoch} | Train Loss: {tr_loss:.4f} | Test Loss: {te_loss:.4f}")

    # -----------------------
    # SAMPLE PREDICTIONS
    # -----------------------
    model.eval()
    with torch.no_grad():
        # --- TRAIN SET ---
        train_samples = random.sample(
            list(train_dataset), 
            min(NUM_SAMPLES_TO_EVAL, len(train_dataset))
        )
        for softlogits, hardprompt_embeds, tokenized_hardprompt in train_samples:
            softlogit_embeds = softlogits @ word_embeddings.weight
            full_inputs = torch.cat([
                softlogit_embeds.unsqueeze(0).to(model.dtype),
                suffix_emb.to(model.dtype),
            ], dim=1)
            
            max_new_tokens = len(tokenized_hardprompt)

            pred_ids = model.generate(inputs_embeds=full_inputs, max_new_tokens=max_new_tokens)
            pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            hardprompt = tokenizer.decode(tokenized_hardprompt, skip_special_tokens=True)

            print(f"Prediction (train): {pred_text}")
            print(f"hardprompt (train): {hardprompt}\n")
        # --- TEST SET ---
        test_samples = random.sample(
            list(test_dataset),
            min(NUM_SAMPLES_TO_EVAL, len(test_dataset))
        )
        for softlogits, hardprompt_embeds, tokenized_hardprompt in train_samples:
            softlogit_embeds = softlogits @ word_embeddings.weight
            full_inputs = torch.cat([
                softlogit_embeds.unsqueeze(0).to(model.dtype),
                suffix_emb.to(model.dtype),
            ], dim=1)

            max_new_tokens = len(tokenized_hardprompt)

            pred_ids = model.generate(inputs_embeds=full_inputs, max_new_tokens=max_new_tokens)
            pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            hardprompt = tokenizer.decode(tokenized_hardprompt, skip_special_tokens=True)

            print(f"Prediction (test): {pred_text}")
            print(f"hardprompt (test): {hardprompt}\n")

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









