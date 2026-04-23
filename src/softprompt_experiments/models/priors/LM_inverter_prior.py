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

from typing import Dict, Optional, Tuple

import transformers
from transformers.modeling_outputs import BaseModelOutput

def args_from_config(args_cls, config):
    args = args_cls()
    for key, value in vars(config).items():
        if key in dir(args):
            setattr(args, key, value)
    return args

class LM_inverter_prior(logit_priors.LogitPrior):
    """
    Prior assuming logits follow a Gaussian mixture model after PCA.
    """
    def __init__(self, base_model, base_tokenizer, base_word_embeddings, softprompt_len):
        super().__init__()
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.word_embeddings = base_word_embeddings
        self.softprompt_len = softprompt_len
        self.inversion_model = load_model(base_model, base_tokenizer)
        self.inversion_model.to(base_model.device)
        # self.gen_kwargs = {
        #     "early_stopping": False,
        #     "num_beams": 1,
        #     "do_sample": False,
        #     "no_repeat_ngram_size": 0,
        #     'min_length': 1,
        #     'max_length': 128
        # }
        self.gen_kwargs = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "num_beams": 1,
            "max_length": 128,
            "early_stopping": False
        }

    def sample_z_prime_from_q(
            self, 
            x_z: BaseModelOutput, 
            labels_z_x_y: Optional[torch.Tensor]=None,
            softprompt_len: Optional[int]=None
        ):
        """
        Samples a z' ~ q(z'|x,z)
        """
        with torch.no_grad():
            # auto regressively generates z prime

            B = x_z.logits.shape[0]
            # first logit to predict token after prompt
            # last_logits_idxs = softprompt_len - 1
            # first logit to predict the first label token
            last_logits_idxs = (labels_z_x_y != -100).float().argmax(dim=1) - 1
        
            last_logits = x_z.logits[torch.arange(B), last_logits_idxs]

            z_prime = self.inversion_model.generate_from_output(
                last_logits,
                self.gen_kwargs,
            )

            return z_prime
    
    def log_prob_from_logits(self, logits, labels):
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)

        # replace -100 with a safe index (won't be used)
        safe_labels = labels.clone()
        safe_labels[labels == -100] = 0

        token_log_probs = log_probs.gather(
            -1,
            safe_labels.unsqueeze(-1)
        ).squeeze(-1)

        # zero out ignored positions
        token_log_probs = token_log_probs.masked_fill(labels == -100, 0.0)

        return token_log_probs.sum(dim=-1) / (labels!=-100).sum(-1)
    def log_prob_q_of_z_prime_given_x_z(
            self,
            z_prime: torch.Tensor, 
            x_z: BaseModelOutput, 
            labels_z_x_y: Optional[torch.Tensor]=None,
            softprompt_len: Optional[int]=None
        ):
        """
        Computes log q(z'|x,z)
        """
        B = x_z.logits.shape[0]

        # first logit to predict token after prompt
        # last_logits_idxs = softprompt_len - 1
        # first logit to predict the first label token
        last_logits_idxs = (labels_z_x_y != -100).float().argmax(dim=1) - 1
    
        last_logits = x_z.logits[torch.arange(B), last_logits_idxs]

        labels_z_prime = z_prime.clone()
        labels_z_prime[labels_z_prime==self.base_tokenizer.pad_token_id] = -100

        outputs = self.inversion_model.forward(
            last_logits, 
            labels=labels_z_prime
        )
        
        # Method 1: actual
        # HF logits are shifted left like [b^,c^,d^,next_token^]
        log_prob = self.log_prob_from_logits(outputs.logits, labels_z_prime)
        
        # Method 2: lazy
        # loss = outputs.loss
        # num_tokens = (labels_z_prime != -100).sum()
        # log_prob = -loss * num_tokens
        return log_prob

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
        outputs = self.base_model(
            inputs_embeds=embs_z_prime, 
            attention_mask=attn_mask_z_prime,
            # labels=labels_z_prime
        )
        # Method 1: actual
        log_prob = self.log_prob_from_logits(outputs.logits, labels_z_prime)
        
        # Method 2: lazy
        # loss = outputs.loss
        # num_tokens = (labels_z_prime != -100).sum()
        # log_prob = -loss * num_tokens
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
        embs_x_y = embs_z_x_y[:,softprompt_len:,:]
        attn_mask_x_y = attn_mask_z_x_y[:,softprompt_len:]
        labels_x_y = labels_z_x_y[:,softprompt_len:]

        #2) stick hard prompt z prime in there
        z_prime_x_y = torch.cat([embs_z_prime, embs_x_y], dim=1)
        # print(f"attn_mask_z_x_y {attn_mask_z_x_y.shape}")
        # print(f"attn_mask_z_x_y raw {attn_mask_z_x_y}")
        # print(f"attn_mask_x_y {attn_mask_x_y.shape}")
        # print(f"attn_mask_z_prime {attn_mask_z_prime.shape}")
        # print(f"attn_mask_z_prime raw {attn_mask_z_prime}")
        attn_mask_z_prime_x_y = torch.cat([attn_mask_z_prime, attn_mask_x_y], dim=1)
        # print(f"attn_mask_z_prime_x_y raw {attn_mask_z_prime_x_y.shape}")
        # print(f"attn_mask_z_prime_x_y raw {attn_mask_z_prime_x_y}")
        labels_z_prime = torch.full(
            (labels_x_y.shape[0], embs_z_prime.shape[1]),
            -100,
            dtype=labels_x_y.dtype,
            device=self.base_model.device
        )
        labels_z_prime_x_y = torch.cat([labels_z_prime, labels_x_y], dim=1)

        # print(f"z_prime_x_y: {z_prime_x_y.shape}")
        # print(f"attn_mask_z_prime_x_y: {attn_mask_z_prime_x_y.shape}")
        outputs = self.base_model(
            inputs_embeds=z_prime_x_y,
            attention_mask=attn_mask_z_prime_x_y,
            # labels=labels_z_prime_x_y
        )

        # Method 1: actual
        log_prob = self.log_prob_from_logits(outputs.logits, labels_z_prime_x_y)
        # Method 2: lazy
        # loss = outputs.loss
        # num_tokens = (labels_z_prime_x_y != -100).sum()
        # log_prob = -loss * num_tokens

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
        )['input_ids'].to(z_prime.device)

        embs_z_prime = self.word_embeddings(z_prime_llama).detach()
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
    def roll_out(self, output, attn_mask_z_x_y, **kwargs):
        """
        output: transformers llm output
        attention_mask: [B, T]
        input_embeds: input sequence embeds including softprompt and Y [B, T, D]
        labels: tokenized labels [B, T]
        softprompt_len: number of softprompt tokens (len of Z)
        returns: (B,) log probabilities

        This needs to be differentiable w.r.t. input_embed
        Use REINFORCE to handle the sampling issue
        """

        device = output.logits.device

        input_embeds = kwargs["input_embeds"]   # [B, T, D]
        labels = kwargs["labels"]
        S = self.softprompt_len

        """       
            ========= 1: Sample the trajectory (hard prompt z') =========
        """
        # Sample z_primes [B, T, V] (t5 tokenized), build batch
        z_primes = self.sample_z_prime_from_q(output, labels_z_x_y=labels, softprompt_len=S)
        # print(z_primes.shape)
        embs_z_prime, attn_mask_z_prime, labels_z_prime = self.build_batch(z_primes)

        """       
            ========= 2: Calculate my log probability terms =========
        """

        # Calc q(z'|x,z)
        log_q_z_prime_x_z = self.log_prob_q_of_z_prime_given_x_z(z_primes, output, labels_z_x_y=labels, softprompt_len=S)

        # Calc p(z')
        log_p_z_prime = self.log_prob_p_of_z_prime(embs_z_prime, attn_mask_z_prime, labels_z_prime)

        # Calc p(y|x,z')
        log_p_y_x_z_prime = self.log_p_of_y_given_x_z_prime(
            embs_z_prime, 
            input_embeds,
            attn_mask_z_prime,
            attn_mask_z_x_y,
            labels,
            S
        )

        batch_info_for_logging = (
            f"\t-log p(y|x,z') mu: {- log_p_y_x_z_prime.mean().item():.3f}, var: {log_p_y_x_z_prime.var(unbiased=False).item():.3f}, best:{(-log_p_y_x_z_prime).min():.3f}\n"
            f"\t-log p(z') mu: {- log_p_z_prime.mean().item():.3f}, var: {log_p_z_prime.var(unbiased=False).item():.3f}, best:{(-log_p_z_prime).min():.3f}\n"
            f"\t-log q(z'|x,z) mu: {- log_q_z_prime_x_z.mean().item():.3f}, var: {log_q_z_prime_x_z.var(unbiased=False).item():.3f}, best:{(-log_q_z_prime_x_z).min():.3f}\n"
        )

        """       
            ========= 3: Calculate my log probability terms =========
        """
        reward = (
            # v1: 
            log_p_y_x_z_prime.detach() 
            
            # v2:
            # log_p_y_x_z_prime.detach() + log_p_z_prime.detach()

            # v3:
            # log_p_y_x_z_prime.detach() + CE_pz_pz_prime + log_p_z_prime.detach() - log_q_z_prime_x_z
        )
        
        return {
            "z_prime": z_primes.detach(),
            "log_q_z_prime_x_z": log_q_z_prime_x_z,
            "reward": reward.detach(),
            "input_embeds": input_embeds.detach(),
            "labels": labels.detach(),
            "attn_mask_z_x_y": attn_mask_z_x_y.detach(),
            "log":batch_info_for_logging
        }
    
    def log_prob(self, output, attn_mask_z_x_y, **kwargs):
        if 'trajectories' in kwargs:
            return self.PPO(output, attn_mask_z_x_y, **kwargs)
        else:
            with torch.no_grad():
                loss = self.REINFORCE(output, attn_mask_z_x_y, **kwargs)
                return loss
            
    # VANILLA REINFORCE      
    def REINFORCE(self, output, attn_mask_z_x_y, **kwargs):
        # only for eval at this point
        trajectories = self.roll_out(output, attn_mask_z_x_y, **kwargs)
        log_q_z_prime_x_z = trajectories['log_q_z_prime_x_z']
        reward = trajectories['reward']
        advantage = (reward - reward.mean()) / (reward.std() + 1e-8)        
        loss = (
            -1. * log_q_z_prime_x_z * advantage.detach()
        )
        return loss

    def sample_old_rollouts(self, output, attn_mask_z_x_y, **kwargs):
        """
           Run this through each batch in the entire trainset once every epoch
           Then collect each batch trajectories in a list of trajectories batches
        """
        with torch.no_grad():
            trajectories = self.roll_out(output, attn_mask_z_x_y, **kwargs)
            return trajectories
        
    # PPO
    def PPO(self, output, attn_mask_z_x_y, **kwargs):
        """
            PPO version, this expects a randomly sampled trajectories batch
            from the list of trajectories batches [trajectories, trajectories, ...]

        """
        epsilon = 0.2
        trajectories =  kwargs['trajectories']
        baseline = kwargs['baseline']
        labels = kwargs["labels"]
        z_primes = trajectories["z_prime"]
        log_q_old = trajectories["log_q_z_prime_x_z"].detach()
        reward = trajectories["reward"]
        S = self.softprompt_len


        # using V(x)
        # advantage = reward - baseline.detach()
        # advantage = advantage / (advantage.std() + 1e-8)     
        # without using V(x)   
        advantage = (reward - reward.mean()) / (reward.std() + 1e-8)

        # recompute with CURRENT policy
        log_q_new = self.log_prob_q_of_z_prime_given_x_z(
            z_primes,
            output,  # or recompute forward pass
            labels_z_x_y=labels,
            softprompt_len=S
        )

        ratio = torch.exp(log_q_new - log_q_old)

        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage

        entropy = -log_q_new.mean()

        L_clipped = torch.min(unclipped, clipped).mean()
        L_entropy = 0.02 * entropy

        print(
            # f"\tReward: {reward.mean()}\n"
            # f"\tAdvantage: {advantage.mean()}\n"
            f"\tL_clipped: {L_clipped}\n"
            f"\tL_entropy: {L_entropy}"
        )

        return (L_clipped + L_entropy)