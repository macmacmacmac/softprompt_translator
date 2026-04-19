from softprompt_experiments.models.priors import logit_priors
from softprompt_experiments.models.LM_inverter import LM_inverter, load_model

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os

import transformers
from transformers.modeling_outputs import BaseModelOutput

def args_from_config(args_cls, config):
    args = args_cls()
    for key, value in vars(config).items():
        if key in dir(args):
            setattr(args, key, value)
    return args

class Inversion_Prior(logit_priors.LogitPrior):
    """
    Prior assuming logits follow a Gaussian mixture model after PCA.
    """
    def __init__(self, base_model, base_tokenizer):
        super().__init__()
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.word_embeddings = base_model.get_input_embeddings()
        self.inversion_model = load_model(base_model, base_tokenizer)
        self.inversion_model.to(base_model.device)
        self.gen_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
            'min_length': 1,
            'max_length': 128
        }

    def sample_z_prime_from_q(
            self, 
            x_z: BaseModelOutput, 
            x_z_attn_mask: torch.Tensor,
            labels: torch.Tensor,
        ):
        """
        Samples a z' ~ q(z'|x,z)
        """
        with torch.no_grad():
            # auto regressively generates z prime
            z_prime = self.inversion_model.generate_from_output(
                x_z,
                x_z_attn_mask,
                self.gen_kwargs,
                labels=labels,
            )
            # decoded = self.inversion_model.tokenizer.batch_decode(
            #     inversion,
            #     skip_special_tokens=True
            # )
            return z_prime
        
    def log_prob_q_of_z_prime_given_x_z(
            self,
            z_prime: torch.Tensor, 
            x_z: BaseModelOutput, 
            x_z_attn_mask: torch.Tensor
        ):
        """
        Computes log q(z'|x,z)
        """
        outputs = self.inversion_model(x_z, z_prime, x_z_attn_mask)
        
        # Method 1: actual
        # HF logits are shifted left like [b^,c^,d^,next_token^]
        # logits = outputs.logits[:, :-1]
        # labels = z_prime[:, 1:]

        # log_probs = F.log_softmax(logits, dim=-1)

        # token_log_probs = log_probs.gather(
        #     -1, labels.unsqueeze(-1)
        # ).squeeze(-1)
        # log_q = token_log_probs.sum(dim=-1)


        # Method 2: lazy
        log_q = outputs.loss

        return log_q

    def log_prob_p_of_z_prime(
            self,
            embs_z_prime: torch.Tensor,
            attn_mask_z_prime: torch.tensor,
            labels_z_prime: torch.Tensor
        ):
        """
        Computes log p(z')
        embs_z_prime: z_prime in llama emb space
        """
        loss = self.base_model(
            input_embeds=embs_z_prime, 
            attention_mask=attn_mask_z_prime,
            labels=labels_z_prime
        ).loss
        num_tokens = (labels_z_prime != -100).sum()
        log_prob = -loss * num_tokens
        return log_prob
    
    def log_p_of_y_given_x_z_prime(
        self,
        embs_z_prime: torch.Tensor,
        embs_z_x_y: torch.Tensor,
        attn_mask_z_prime: torch.tensor,
        attn_mask_z_x_y: torch.tensor,
        labels_z_x_y: torch.Tensor,
        softprompt_len: int
    ):
        """
            Computes loss using the new z_prime
            z_x_y: input_embeds 
        """
        #1) remove soft prompt z from original sequence
        x_y = z_x_y[:,softprompt_len:,:]
        attn_mask_x_y = attn_mask_z_x_y[:,softprompt_len:,:]
        labels_x_y = labels_z_x_y[:,softprompt_len:,:]

        #2) stick hard prompt z prime in there
        z_prime_x_y = torch.cat([embs_z_prime, embs_x_y], dim=1)
        attn_mask_z_prime_x_y = torch.cat([attn_mask_z_prime, attn_mask_x_y], dim=1)
        labels_z_prime = torch.full(
            (labels_x_y.shape[0], embs_z_prime.shape[1] + suffix_emb.shape[1]),
            -100,
            dtype=labels.dtype,
            device=device
        )
        labels_z_prime_x_y = torch.cat([labels_z_prime, labels_x_y], dim=1)

        loss = self.base_model(
            input_embeds=z_prime_x_y,
            attention_mask=attn_mask_z_prime_x_y,
            labels=labels_z_prime_x_y
        ).loss
        num_tokens = (labels_z_prime_x_y != -100).sum()
        log_prob = -loss * num_tokens

        return log_prob
    
    def build_batch(
        self,
        z_prime: torch.Tensor,
    ):
        """
            z_prime: tokens in T5 token space [B,T_z',V]
        """
        #1) decode z_prime into NL and retokenize it into Llama tokenspace
        #   then embed it
        decoded_z_prime = self.inversion_model.tokenizer.batch_decode(
            z_prime,
            skip_special_tokens=True
        )
        z_prime_llama = self.base_tokenizer(
            decoded_z_prime,
            padding='longest', 
            return_tensors='pt',
        )['input_ids']

        embs_z_prime = self.word_embeddings(z_prime_llama)
        attn_mask_z_prime = (z_prime_llama != self.base_tokenizer.pad_token_id).long()

        labels_z_prime = z_prime_llama.clone()
        labels_z_prime[labels_z_prime==self.base_tokenizer.pad_token_id] = -100

        return embs_z_prime, attn_mask_z_prime, labels_z_prime

    """
        Log_prob v1:
        - Sample a hard prompt z' using the inverter
        - Compute CE between log p(y|z',x) and log(p|z,x)
        - Treat z^ as a detached random variable, no gradient
    """
    def log_prob(self, output, attn_mask_z_x_y, **kwargs):
        """
        output: transformers llm output
        attention_mask: [B, T]
        input_embeds: input sequence embeds including softprompt and Y [B, T, D]
        labels: tokenized labels [B, T]
        softprompt_len: number of softprompt tokens (len of Z)
        returns: (B,) log probabilities
        """

        device = output.logits.device

        input_embeds = kwargs["input_embeds"]   # [B, T, D]
        attn_mask_z_x_y = kwargs["attn_mask_z_x_y"]         # [B, T]
        labels = kwargs["labels"]
        S = kwargs["softprompt_len"]

        # Sample z_primes [B, T, V] (t5 tokenized), build batch
        z_primes = self.sample_z_prime_from_q(output, x_z_attn_mask, labels)
        embs_z_prime, attn_mask_z_prime, labels_z_prime = self.build_batch(z_primes)

        # Calc p(z_prime)
        log_p_z_prime = self.log_prob_p_of_z_prime(embs_z_prime, attn_mask_z_prime, labels_z_prime)

        # Calc p(y|x,z_prime)
        log_p_y_x_z_prime = self.log_p_of_y_given_x_z_prime(
            embs_z_prime, 
            input_embeds,
            attn_mask_z_prime,
            attn_mask_z_x_y,
            labels,
            S
        )

        reward = (
            # v1: just reconstruction and NL loss
            log_p_y_x_z_prime + log_p_z_prime

            # full:
            # log_p_y_x_z_prime + CE_pz_pz_prime + log_p_z_prime - log_q_z_prime_x_z
        )
        loss = (
            log_q_z_prime_x_z * reward.detach()
        )
        
        return loss