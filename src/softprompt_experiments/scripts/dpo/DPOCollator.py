import torch

class DPOCollator:
    """
    Collator for DPO pipeline. Expects each batch to have, in this order:
    1. z_prime: list of soft prompts embeds [(1, T, E), (1, T, E), ...]
    2. z_W: list of preferred hard prompts [str, str, str, ...]
    3. z_L: list of dispreferred hard prompts [str, str, str, ...]
    4. logp_ref_z_W: log prob of preferred under ref policy [(), (), (), ...]
    5. logp_ref_z_L: log prob of dispreferred under ref policy [(), (), (), ...]
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Retrieve list of soft_prompts and hard_prompts (in that order)
        z_prime, z_W, z_L, logp_ref_z_W, logp_ref_z_L = zip(*batch)

        # Stack the log probs into a batch
        logp_ref_z_W = torch.stack(logp_ref_z_W.unsqueeze(0))
        logp_ref_z_L = torch.stack(logp_ref_z_L.unsqueeze(0))

        # Stack the frozen soft prompts into a batch: (batch_size, soft_prompt_len, embed_dim)
        z_prime = torch.stack(z_prime)        # (batch_size, soft_prompt_len, embed_dim)

        # Explicitly append the EOS token so the model learns when to stop
        z_W = [prompt + self.tokenizer.eos_token for prompt in z_W]
        z_L = [prompt + self.tokenizer.eos_token for prompt in z_L]

        # Tokenize the hard prompts
        z_W_tokenized = self.tokenizer(
            z_W, 
            padding=True, 
            truncation=True, 
            max_length=300, # TODO: Test this value
            return_tensors="pt",
            add_special_tokens=True
        )
        z_L_tokenized = self.tokenizer(
            z_L, 
            padding=True, 
            truncation=True, 
            max_length=300, # TODO: Test this value
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Get ids and masks for the hard prompts
        z_W_ids = z_W_tokenized["input_ids"]              # (batch_size, seq_len)
        z_W_attn_mask = z_W_tokenized["attention_mask"]    # (batch_size, seq_len)

        z_L_ids = z_L_tokenized["input_ids"]              # (batch_size, seq_len)
        z_L_attn_mask = z_L_tokenized["attention_mask"]    # (batch_size, seq_len)

        # Truncation can cut off the appended EOS for prompts longer than max_length;
        # force the last attended token to EOS so every sequence keeps a supervised
        # stop signal (no-op for non-truncated rows, whose last token is already EOS)
        last_positions = z_W_attn_mask.sum(dim=1) - 1                              # (batch_size,)
        z_W_ids[torch.arange(z_W_ids.size(0)), last_positions] = self.tokenizer.eos_token_id

        last_positions = z_L_attn_mask.sum(dim=1) - 1                              # (batch_size,)
        z_L_ids[torch.arange(z_L_ids.size(0)), last_positions] = self.tokenizer.eos_token_id

        # Create labels and mask the padding tokens with -100
        z_W_labels = z_W_ids.clone()                      # (batch_size, seq_len)
        z_W_labels[z_W_attn_mask == 0] = -100              # (batch_size, seq_len)
        z_W_tokenized['labels'] = z_W_labels

        z_L_labels = z_L_ids.clone()                      # (batch_size, seq_len)
        z_L_labels[z_L_attn_mask == 0] = -100              # (batch_size, seq_len)
        z_L_tokenized['labels'] = z_L_labels

        return {
            "z_prime": z_prime, # soft prompts tensor (batch, seq_len, emb_dim)
            "z_W_tokenized": z_W_tokenized, # tokenized pref hardprompt {'input_ids', 'attention_mask', 'labels'}
            "z_L_tokenized": z_L_tokenized, # tokenized dispref hardprompt {'input_ids', 'attention_mask', 'labels'}
            "log_p_ref_z_W": logp_ref_z_W, # log prob of pref hard under ref model (batch)
            "log_p_ref_z_L": logp_ref_z_L # log prob of dispref hard under ref model (batch)

        }
