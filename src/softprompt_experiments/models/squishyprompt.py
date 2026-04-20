import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW, Adam
import random
import os
import copy
from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.models.priors.logit_priors import LogitPrior

# from vec2text.models import InversionFromLogitsEmbModel
# from vec2text.models.config import InversionConfig
# from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments

# name = "jxm/t5-base__llama-7b__one-million-instructions__emb"
# config = InversionConfig.from_pretrained(name)
# inv_model = InversionFromLogitsEmbModel(config).from_pretrained(name)
def get_last_token_logits(logits, attention_mask):
    # logits: [B, T, V]
    # attention_mask: [B, T]

    # lengths = number of non-pad tokens
    lengths = attention_mask.sum(dim=1) - 1  # [B]

    # gather indices
    B = logits.size(0)
    V = logits.size(-1)

    # shape → [B, 1, V]
    last_logits = logits[torch.arange(B), lengths]

    return last_logits  # [B, V]

class SquishyPrompt(SoftPrompt):
    """
    An implementation of soft prompts that includes regularization using a learned prior term for logits
    - lambda: regularization constant
    - logits_prior: a torch.nn.Module that computes the log probability of observing a single token's logit
    """
    def __init__(
            self, 
            logits_prior: LogitPrior,
            model=None, 
            init=None,
            tokenizer=None, 
            word_embeddings=None, 
            path_to_model=None, 
            num_tokens=8, 
        ):
        super().__init__(
            model=model, 
            init=init,
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            path_to_model=path_to_model, 
            num_tokens=num_tokens
        )
        self.logits_prior = logits_prior

    def loss_fn(self, input_embeds, labels, attention_mask, return_entropy=False):
        """
            Computes normal task supervision loss based on next-token predictions.
            But also incorporates a prior over the logits.
        """
        # attention_mask = (labels != -100).long()
        # print(labels)
        # print("computing normal loss")
        outputs = self._model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,   # fully causal
            labels=labels          # HF computes CE internally
        )

        # last_logits = get_last_token_logits(logits, attention_mask) # [B, V]
        # sp_logits = logits[torch.arange(logits.size(0)), self.num_tokens]

        # print("computing prior term")
        prior_term = self.logits_prior.log_prob(
            outputs, 
            attention_mask,
            input_embeds=input_embeds,
            labels=labels,
            softprompt_len=self.num_tokens
        ).mean()
        # prior_term += self.logits_prior.log_prob(sp_logits).mean()
        # prior_term = prior_term/2

        loss = outputs.loss
        # loss = torch.tensor(0.0)

        return loss, prior_term    
