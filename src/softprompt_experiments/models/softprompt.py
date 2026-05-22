import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import math


class SoftPrompt(nn.Module):
    """
    An implementation of softprompt for prompt-tuning, subclasses nn.Module
    - model: a huggingface decoder model
    - init_randomly: decide whether to initialize soft prompt embedding randomly using N(0, 1) or from tokens from the model's vocab
    - init_text: either 'phrase' or 'random.' Former initializes from a passed in phrase
    - word_embeddings: it's word_embedding matrix from model.get_input_embeddings
    - tokenizer: the tokenizer to be used
    - path_to_model: if passed, loads a saved softprompt model instead of initializing one
    - num_tokens: number of virtual tokens in the softprompt
    """
    def __init__(self, 
                 model=None, 
                 init_randomly = False,
                 init_text=None, 
                 tokenizer=None, 
                 word_embeddings=None, 
                 path_to_model=None, 
                 num_tokens=8):
        super().__init__()

        # Register tokenizer, model and word_embedddings matrix as class instance variables without being
        # directly registered as child modules to avoid tight coupling of object of SoftPrompt class and these
        # during training.
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_word_embeddings", word_embeddings)

        # Member Variables
        self.num_tokens = num_tokens
        self.prompt_embeddings = None
        self.initial_tokens = None
        self.initial_embeddings = None

        # If the Soft Prompt does not need to be initialized with Pre-Trained weights
        if path_to_model is None:

            # Init Soft Prompt embeddings randomly from N(0, 1)
            if init_randomly:

                self.prompt_embeddings = nn.Parameter(
                    torch.randn(
                        self.num_tokens, 
                        word_embeddings.embedding_dim, 
                        dtype=model.dtype, 
                        device=model.device
                    )
                )

            # Init Soft Prompt embeddings using tokens from the vocab depending on init_text parameter
            else:

                # Get the Vocabulary Size
                vocab_size = word_embeddings.num_embeddings

                # If init_text is provided (then we initialize soft prompt using init_text)
                if init_text is not None:

                    # Tokenize init_text without special tokens
                    init_token_ids = tokenizer(init_text, add_special_tokens=False)["input_ids"]

                    # Calculate the total num of text tokens
                    num_text_tokens = len(init_token_ids)

                    # If num of text tokens greater than num of soft prompt tokens
                    if num_text_tokens > num_tokens:

                        # Then trim the text tokens to the size of num of soft prompt tokens
                        init_token_ids = init_token_ids[:num_tokens]

                    # If num of text tokens is less than num of soft prompt tokens
                    elif num_text_tokens < num_tokens:

                        # Find number of times to repeat the token ids
                        num_reps = math.ceil(num_tokens / num_text_tokens)

                        # Repeat the token ids until its equivalent to the number of soft prompt tokens
                        init_token_ids = init_token_ids * num_reps

                    # Perform the trimming again just to be certain that the token ids do not exceed num of soft prompt tokens
                    init_token_ids = init_token_ids[:num_tokens]

                    # Convert the token ids to a Long Tensor
                    # and move to the same device as word embedding matrix
                    init_token_ids = torch.LongTensor(init_token_ids).to(word_embeddings.weight.device)
                
                # If Random Init is requested
                else:

                    # Randomly sample "num_tokens" token_ids from the vocab
                    init_token_ids = torch.randint(
                        0, vocab_size,(self.num_tokens,), dtype=torch.long
                    ).to(model.device)

                # Compute embeddings for init_token_ids
                word_embedding_weights = word_embeddings(init_token_ids).detach().clone().to(model.dtype)

                # Create a Module Parameter using the computed token embeddings
                self.prompt_embeddings = nn.Parameter(word_embedding_weights.to(model.device))

                # Keep copies of initial token ids and token embeddings
                self.initial_tokens = copy.deepcopy(init_token_ids)
                self.initial_embeddings = copy.deepcopy(word_embedding_weights)

        # If a path to pretrained soft prompts are provided
        else:
            
            # Init a SoftPrompt instance using the pretrained soft prompts path
            self.load_softprompt(path_to_model)


    # NOTE: Unused code?
    def set_prompt_embeddings(self, new):
        if len(new.shape) == 2:
            with torch.no_grad():
                self.prompt_embeddings.data = new # Reassigns the underlying data
        else:
            raise ValueError(f"new prompt embeddings must be of shape [num_tokens, dim], found: {new.shape}")
    

    def forward(self):
        return self.prompt_embeddings.unsqueeze(0)  # [1, num_tokens, embed_dim]


    def loss_fn(self, input_embeds, labels, return_entropy=False):
        outputs = self._model(
            inputs_embeds=input_embeds,
            attention_mask=None,   # fully causal
            labels=labels          # HF computes CE internally
        )

        if not return_entropy:
            return outputs.loss

        # logits: [B, T, V]
        logits = outputs.logits

        # log p(y_t | ...)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # entropy per token: [B, T]
        token_entropy = -(probs * log_probs).sum(dim=-1)

        # mask out ignored labels (-100)
        valid_mask = (labels != -100)

        # mean entropy over valid tokens
        entropy = (token_entropy * valid_mask).sum() / valid_mask.sum()

        return outputs.loss, entropy    
    

    def generate_from_embeds(self, embeds=None, max_new_tokens=20, do_sample=True, suffix_str=None):
        """
        Generate text given softprompt embeddings.
        Args:
            embeds: [1, seq_len, hidden_dim] softprompt embeddings
            max_new_tokens: number of tokens to generate
            do_sample: whether to sample or use greedy decoding
            suffix_str: some string to be appended after the embeds
        Returns:
            generated string
        """
        with torch.no_grad():
            if embeds is not None:
                sp_embeds = self.forward()   # [1, soft_len, dim]
                sp_embeds = sp_embeds.expand(len(embeds), -1, -1) #[batchsize, soft_len, dim]
                full_embs = torch.cat([sp_embeds,embeds],dim=1)
                if suffix_str:
                    ids = self._tokenizer(suffix_str, return_tensors="pt").input_ids.to(self._model.device)
                    suffix_embs = self._word_embeddings(ids).to(dtype=self._model.dtype)
                    full_embs = torch.cat([full_embs, suffix_embs], dim=1)
                attention_mask = torch.ones(full_embs.size()[:-1], device=full_embs.device, dtype=torch.long)
                output_ids = self._model.generate(
                    inputs_embeds=full_embs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            else:
                full_embs = self.forward()   # [1, soft_len, dim]
                if suffix_str:
                    ids = self._tokenizer(suffix_str, return_tensors="pt").input_ids.to(self._model.device)
                    suffix_embs = self._word_embeddings(ids).to(dtype=self._model.dtype)
                    full_embs = torch.cat([full_embs, suffix_embs], dim=1)
                attention_mask = torch.ones(full_embs.size()[:-1], device=full_embs.device, dtype=torch.long)
                output_ids = self._model.generate(
                    inputs_embeds=full_embs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
        output = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        return output


    def save_softprompt(self, path_to_save):
        state_dict = {
            'prompt_embeddings':self.forward(),
            'initial_tokens':self.initial_tokens,
            'initial_embeddings':self.initial_embeddings,
            'num_tokens':self.num_tokens
        }
        torch.save(state_dict, os.path.join(path_to_save, "softprompt.pt"))
    

    def load_softprompt(self, path_to_load):
        state_dict = torch.load(path_to_load)
        self.initial_tokens = state_dict['initial_tokens']
        self.initial_embeddings = state_dict['initial_embeddings']
        self.num_tokens = state_dict['num_tokens']
        
        # Wrap the tensor of soft prompts into a learnable Parameter
        self.prompt_embeddings = nn.Parameter(state_dict['prompt_embeddings'].squeeze(0))


    # NOTE: Unused Code?
    def get_nearest_to_embeds(self, distance='cosine'):
        """
            Retrieves the discrete, nearest hard tokens to the prompt embeddings
            Args:
                distance: either 'l2' or 'cosine' defaults to l2
            Returns:
                nearest_idx: nearest token ids
                discrete_prompt: decoded nearest token ids
        """
        with torch.no_grad():
            prompt_embedding = self.forward().squeeze(0) #[num_tokens, embed_dim]
            base_embedding = self._word_embeddings.weight.data

            # print(prompt_embedding.shape) #8 by 4096
            # print(base_embedding) # 128256 by 4096

            norm_base = torch.nn.functional.normalize(base_embedding, dim=1)
            norm_embed = torch.nn.functional.normalize(prompt_embedding, dim=1)

            cos_sim = norm_embed @ norm_base.T
            nearest_idx = torch.argmax(cos_sim, dim=1).cpu().tolist()

            discrete_prompt = self._tokenizer.decode(nearest_idx)

        return nearest_idx, discrete_prompt
    
    # NOTE: Unused Code?
    def get_nearest_to_logits(self, k):
        """
            Retrieves of the k likeliest predicted next-prompt tokens based on logit probs
            Args:
                k: number of candidate predictions to return per prompt token
            Returns:
                decodeds: the decoded k likeliest predicted next prompt tokens
                topk_vals: their respective probabilities
        """
        with torch.no_grad():
            prompt_embedding = self.forward().squeeze(0) #[num_tokens, embed_dim]
            base_embedding = self._word_embeddings

            logits, probs = self.get_prompt_logits()
            topk_vals, topk_idx = probs.topk(k, dim=-1)

            decodeds = []
            for i in range(logits.size(0)):
                toks = [self._tokenizer.decode(tid) for tid in topk_idx[i]]
                decodeds.append(toks)
        return decodeds, topk_vals
    

    def _get_prompt_logits(self):
        prompt_embeds = self.forward()
        logits = self._model(inputs_embeds=prompt_embeds, output_hidden_states=False, use_cache=False).logits
        logits = logits[:, :self.num_tokens-1, :]  # align with next-token positions
        probs = F.softmax(logits, dim=-1)

        return logits, probs


    def get_prompt_logits(self):
        """
            Retrieves the last hidden state logits for each prompt token except for the last
            Returns:
                logits: the raw unnormalized logits
                probs: the token probabilities
        """
        with torch.no_grad():
            logits, probs = self._get_prompt_logits()
        return logits, probs
    
    
    def get_parsability(self):
        """
        Negative mean cosine similarity between input embeddings and predicted next-token embeddings.
        Returns a tensor for backprop.
        """
        prefix_embed = self.forward()

        logits, probs = self._get_prompt_logits()

        vocab_embed_mat = self._word_embeddings.weight
        weighted_avg = probs @ vocab_embed_mat  # [1, seq_len-1, hidden_dim]
        weighted_avg = weighted_avg.squeeze(0)
        
        cos = F.cosine_similarity(prefix_embed[0,1:], weighted_avg, dim=-1)  # [seq_len-1]
        return -cos.mean()  # negative for loss minimization



        