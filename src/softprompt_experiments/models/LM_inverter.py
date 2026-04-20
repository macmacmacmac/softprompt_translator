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

from transformers.modeling_outputs import BaseModelOutput

from vec2text.models.config import InversionConfig
from vec2text.models import InversionFromLogitsEmbModel
from vec2text.models.model_utils import (
    load_encoder_decoder,
)

def args_from_config(args_cls, config):
    args = args_cls()
    for key, value in vars(config).items():
        if key in dir(args):
            setattr(args, key, value)
    return args

NAME ="jxm/t5-base__llama-7b__one-million-instructions__emb"

def load_model(embedder, embedder_tokenizer):
    model_kwargs = {"low_cpu_mem_usage": True}
    encoder_decoder = load_encoder_decoder("t5-base")

    inv_model = LM_inverter.from_pretrained(NAME, embedder=embedder, embedder_tokenizer=embedder_tokenizer, encoder_decoder=encoder_decoder)
    return inv_model

class LM_inverter(InversionFromLogitsEmbModel):
    def __init__(self, config, embedder=None, embedder_tokenizer=None, **kwargs):
        super().__init__(
            config=config,
            embedder=embedder,
            embedder_tokenizer=embedder_tokenizer,
            **kwargs
        )

    def _process_embedder_output(
        self,
        outputs: BaseModelOutput,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = outputs.logits.shape[0]

        if labels == None:
            last_logits_idxs = attention_mask.sum(1) - 1
        else:
            # first logit to predict the first label token
            last_logits_idxs = (labels != -100).float().argmax(dim=1) - 1
        
        last_logits = outputs.logits[torch.arange(B), last_logits_idxs]

        embeddings = last_logits.log_softmax(dim=1)
        zeros = torch.zeros(
            (B, self.num_zeros_to_add),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        return torch.cat((embeddings, zeros), dim=1)

    def project_embedding(self, embeddings):
        num_tokens = self.num_tokens
        # Remove any extraneous zeros
        embeddings = embeddings[:, : self.tokenizer_mapping.numel()]  # (B, V)

        # Map embeddings to our space.
        batch_size = embeddings.shape[0]
        new_embeddings = torch.zeros(
            (batch_size, self.encoder_decoder.config.vocab_size),
            device=embeddings.device,
            dtype=torch.double,
        )
        mapping = (
            self.tokenizer_mapping[None]
            .repeat((batch_size, 1))
            .to(new_embeddings.device)
        )
        embeddings = new_embeddings.scatter_add(
            dim=1, index=mapping, src=embeddings.to(torch.double).exp()
        ).log()
        embeddings = (
            embeddings.nan_to_num()
        )  # replace empty values from -inf to tiny neg number

        if self.training:
            unigram_batch = embeddings.mean(dim=0, keepdim=True)
            # Update unigram.
            if self.unigram.sum() == 0:
                print("INFO: resetting unigram.")
                self.unigram.data = unigram_batch
            else:
                self.unigram.data = self.unigram.data * (
                    1 - self.unigram_beta
                ) + unigram_batch * (self.unigram_beta)
        embeddings = embeddings - self.unigram
        embeddings = embeddings.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        logits_zeros = torch.zeros(
            (batch_size, self.num_zeros_to_add),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        logits = torch.cat((embeddings, logits_zeros), dim=1).to(
            self.sequence_weights.dtype
        )
        logits = logits.reshape((batch_size, num_tokens, -1))

        with torch.no_grad():
            # Minibatch
            embeddings_list = []
            i = 0
            while i < batch_size:
                batch_logits = logits[i : i + self.minibatch_size, ...]
                batch_embeddings = torch.einsum(
                    "smd,bsm -> bsd", self.word_embeddings, batch_logits
                )
                embeddings_list.append(batch_embeddings)
                i += self.minibatch_size
            embeddings = torch.cat(embeddings_list, dim=0)

        embeddings = self.embedding_proj(embeddings)
        assert embeddings.shape == (
            batch_size,
            num_tokens,
            self.encoder_hidden_dim,
        )
        attention_mask = torch.ones(
            (batch_size, num_tokens), dtype=torch.long, device=embeddings.device
        )
        return embeddings, attention_mask

    def generate_from_output(
            self, 
            model_outputs: BaseModelOutput, 
            generation_kwargs: Dict[str, torch.Tensor],
            attention_mask: Optional[torch.Tensor]=None,
            labels: Optional[torch.Tensor]=None
        ):
        processed_embeddings = self._process_embedder_output(model_outputs, attention_mask, labels=labels)
        inputs_embeds, attention_mask = self.project_embedding(processed_embeddings)
        return self.encoder_decoder.generate(
            # required: input embeddings
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # optional: input IDs (for starting generation).
            # typically not set unless generating prefixes for
            # reranking.
            **generation_kwargs,
        )

    def forward(
        self, 
        model_outputs: BaseModelOutput,
        labels: Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
    ):
        # attention mask just here to tell us where the last valid token is
        processed_embeddings = self._process_embedder_output(model_outputs, attention_mask=attention_mask, labels=labels)
        inputs_embeds, attention_mask = self.project_embedding(processed_embeddings)

        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

