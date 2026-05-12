import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from typing import List, Dict


# Determine DEVICE and DTYPE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Few Shot Examples for Classification for creating Target Prompts
FEW_SHOT_EXAMPLES = [
    "Identify the emotion expressed in this text: joy, sadness, anger, fear",
    "Is the information in this sentence correct?: True, False",
    "Classify this passage from a book or movie into its genre: science fiction, romance, thriller",
    "Determine who is the author of a given text: Shakespeare or Marlowe",
    "Identify which season is described in this text: summer, winter, autumn or spring",
    "Categorize customer feedback into different types: bug report, feature request, compliment",
    "Identify the type of this email: spam or not spam",
    "Identify the political leaning of a text or author: left or right"
]

FEW_SHOT_EXAMPLES_DOD = [
    "joy, sadness, anger, fear",
    "True, False",
    "science fiction, romance, thriller",
    "Shakespeare or Marlowe",
    "summer, winter, autumn or spring",
    "bug report, feature request, compliment",
    "spam or not spam",
    "left or right"   
]


# Combination of All Layers for Patchscopes Experiment
ALL_LAYER_COMBINATIONS = [
    {"min_source": -1, "max_source": 30, "min_target": -1, "max_target": 30}
]


# Best Patches of Layer Combinations for Patchscopes Experiment
BEST_PATCHES = [
    # {"min_source": -1, "max_source": -1, "min_target": -1, "max_target": -1},
    # {"min_source": 10, "max_source": 10, "min_target": 0, "max_target": 0},
    # {"min_source": 5, "max_source": 5, "min_target": 24, "max_target": 24},
    # {"min_source": 6, "max_source": 6, "min_target": 26, "max_target": 26},
    # {"min_source": 22, "max_source": 22, "min_target": 28, "max_target": 28},
    {"min_source": 13, "max_source": 15, "min_target": 24, "max_target": 26}
]



def elicit_description_using_inspect_technique(
        model,
        tokenizer,
        num_tokens,
        soft_prompt,
        dataset_name,
        layer_combinations,
        target_prompt_type='few_shot'
) -> List[Dict]:

    # Create a Target Prompt for the experiments
    target_prompt = create_target_prompt(num_tokens, target_prompt_type, dataset_name)

    # Determine End Token
    # TODO: Can remove this if this will never be actually used by the autoregressive decoding code
    end_token = tokenizer.encode('$', add_special_tokens=False)[0]

    # Init List to store results
    results = []

    # Determine Layer Cominations
    combinations = layer_combinations or ALL_LAYER_COMBINATIONS
    
    # Calculate total iterations for progress bar
    total_iterations = sum(
        (comb.get("max_source") - comb.get("min_source") + 1) * 
        (comb.get("max_target") - comb.get("min_target") + 1)
        for comb in combinations
    )

    with tqdm(total=total_iterations, desc="Patchscopes experiments") as pbar:
        # For each combination
        for comb in combinations:

            # For each source layer ranging from min to max value
            for source_layer in range(comb.get("min_source"), comb.get("max_source") + 1):

                # For each target layer ranging from min to max value
                for target_layer in range(comb.get("min_target"), comb.get("max_target") + 1):

                    # Run patchscopes experiment using the src and tgt layers
                    patched_output = perform_inspect_for_src_tgt_pair(
                        model, 
                        tokenizer, 
                        soft_prompt, 
                        target_prompt, 
                        num_tokens, 
                        source_layer, 
                        target_layer,
                        end_token,
                        target_prompt_type
                    )

                    # Append the experiment output to the results list
                    results.append({
                        "source_layer": source_layer,
                        "target_layer": target_layer,
                        "output": patched_output
                    })

                    pbar.update(1)

    # Return the results list comprising of patchscopes output
    # for each and every combination of source and target layers
    return results
    

def create_target_prompt(num_tokens: int, target_prompt_type: str, dataset_name: str):
    match target_prompt_type:
        case 'few_shot':
            return create_few_shot_prompt(num_tokens)
        case 'cot':
            # Find a random test example from the test dataset 
            # TODO: Change this to the DoD dataset instead of using InSPEcT datasets, as the soft prompt is trained on DoD
            text_field = determine_text_field_for_inspect_dataset(dataset_name)
            random_test_example = load_dataset(dataset_name, trust_remote_code=True, split='test').shuffle()[0][text_field]

            # Create Chain of Thought styled Target Prompt
            return create_cot_prompt(num_tokens, random_test_example)
        case _:
            raise ValueError("Invalid target_prompt_type supplied. Supported values are: 'few_shot', 'cot'.")


def create_few_shot_prompt(num_tokens, separator='|', seed=47):
    random.seed(seed)
    selected_examples = random.sample(FEW_SHOT_EXAMPLES, 3)
    separator = " " + separator + " "
    prompt = separator.join(selected_examples) + separator[:-1] + " x" * num_tokens
    return prompt


def determine_text_field_for_inspect_dataset(dataset_name):
    match dataset_name:
        case 'stanfordnlp/sst2':
            return 'sentence'
        case _:
            return 'text'


def create_cot_prompt(num_tokens, random_test_example):
    # CoT Target Prompt should look like: 
    # <SOFT_PROMPT> + " text: " + <EXAMPLE_TEXT_FROM_TEST_DATASET> + " Label: " + " First I should classify this as:"
    soft_prompt_placeholder =  " x" * num_tokens
    cot_prompt = f"{soft_prompt_placeholder} Text: {random_test_example} Label: First I should classify this as one of the following"
    return cot_prompt


def perform_inspect_for_src_tgt_pair(                             
        model, 
        tokenizer, 
        soft_prompt, 
        target_prompt, 
        num_tokens, 
        source_layer, 
        target_layer,
        end_token,
        target_prompt_type
):
    # TODO: Move this out of this code block as it will be called for the same soft prompt, multiple times
    # Run soft prompt through the model and capture hidden states at every layer
    hs_cache, _ = build_soft_hs_cache(soft_prompt, model, tokenizer, num_tokens)

    # Tokenize the target prompt
    target_inp = tokenizer(target_prompt, return_tensors="pt").to(DEVICE)

    # Determine target position based on target prompt type
    if target_prompt_type == 'few_shot':
        # Calculate position for patching (at placeholder token (x) positions)
        target_position = target_inp["input_ids"].shape[1] - num_tokens # Can add -1 for the ":" in the end

    # For 'cot' target prompt type
    else:
        # Use target_position as 1, since first token is BOS
        target_position = 1

    # Determine End Token (Default is '$' but will be set to ',' here)
    end_token = end_token or tokenizer.encode(',', add_special_tokens=False)[0]

    # Create a deep copy of target inp tensors
    target_inp_copy = {}
    for _k, _v in target_inp.items():
        target_inp_copy[_k] = _v.detach().clone().to(DEVICE)

    # Create hs_patch_config
    # hs_cache[i] stores hidden_states[i] which is:
    #   - i=0: embedding output (before any transformer layer)
    #   - i=1 to 32: output of layer i (or i - 1 according to model.model.layers indexing style)
    # So to get output of source_layer, we access hs_cache[source_layer + 1]
    hs_patch_config = {
        target_layer: [
            (target_position, hs_cache[source_layer + 1][0])
        ]
    }

    # Generate text with the patched hidden states
    return generate_greedy_deterministic(hs_patch_config, 
                                         target_inp_copy, 
                                         60, 
                                         end_token, 
                                         model, 
                                         tokenizer, 
                                         num_tokens, 
                                         target_layer = target_layer,
                                         source_layer = source_layer,
                                         do_sample = False,
                                         visualize_confidence = False)



def build_soft_hs_cache(soft_prompt, model, tokenizer, num_of_tokens):
    # Create a list [0, 1, 2, ..., num_layers] - All layers including embeddings
    layers_to_cache = list(range(model.config.num_hidden_layers+1))

    # Init Hidden State Cache as a dict
    hs_cache = {}

    # Tokenize Placeholder text like " x x x" (example for 3 tokens)
    inp = tokenizer(" x" * num_of_tokens, return_tensors="pt").to(DEVICE)

    # Calculate starting position of the "x" tokens in the sequence.
    pos = inp['input_ids'].shape[1] - num_of_tokens

    # Convert soft_prompt to model's dtype (bfloat16)
    soft_prompt = soft_prompt.to(model.dtype)
    
    # Inject soft_prompt at layer 0
    patch_hooks = set_soft_prompt_patch_hook(model, soft_prompt, pos, num_of_tokens) 

    # Run forward pass of the placeholder tokens
    with torch.no_grad():
        output = model(**inp, output_hidden_states = True)

    # Remove all hooks (call-back methods after each forward call on a layer)
    remove_hooks(patch_hooks)

    # Debug: Check if soft_prompt was injected at embedding layer
    # print(f"Checking layer 0 (embeddings):")
    # print(f"Match: {torch.allclose(output['hidden_states'][0][0][pos:], soft_prompt, atol=1e-5)}")
    # print(f"Max diff: {(output['hidden_states'][0][0][pos:] - soft_prompt).abs().max().item()}")

    # For each layer idx in layers to cache
    for layer in layers_to_cache:

        # If layer is not in cache yet
        if layer not in hs_cache:

            # Allocate an empty list for that layer
            hs_cache[layer] = []

        # Store hidden state at the patched position
        hs_cache[layer].append(output["hidden_states"][layer][0][pos:])

    # Save the hs_cache
    # torch.save(hs_cache, 'hs_cache_first_layer.pt')

    # Return the hidden state cache and the tokenized placeholder text
    return hs_cache, inp


def set_soft_prompt_patch_hook(model, soft_prompt, source_position, num_of_tokens):
    # Create a hook that replaces hidden states at source_position with soft_prompt tensor
    def patch_sp(name, soft_prompt):
        def hook(module, input, output):
            # (batch, sequence, hidden_state)
            if model.config.model_type == "llama":
                # output[0][0, source_position : source_position + num_of_tokens] = soft_prompt     #NOTE: A Bug Maybe, as it is straightforwardly incompatible...
                output[0][source_position : source_position + num_of_tokens] = soft_prompt
            
            else:
                raise ValueError(f"Unknown model: {model.config.model_type}")
        
        return hook
    
    # If its a Llama model
    if model.config.model_type == "llama":

        # Fetch the first layer
        first_layer: torch.nn.Module = model.model.layers[0]

        # Fetch the embedding layer
        embed_layer: torch.nn.Module = model.model.embed_tokens
    
    # Raise Error if model is any other than a Llama
    else:
        raise ValueError(f"Unknown model: {model.config.model_type}")
    
    # Attach the hook to layer 1 (first decoder layer) (so soft_prompt enters at the start of processing)
    # hooks = [first_layer.register_forward_hook(
    #     patch_sp("patch_sp_0", soft_prompt)
    # )]

    # Attach the hook to layer 0 (embed_tokens layer) (so soft_prompt enters at the start of processing)
    hooks = [embed_layer.register_forward_hook(
        patch_sp("patch_sp_0", soft_prompt)
    )]

    # Return all the created hooks (only 1 here though)
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def generate_greedy_deterministic(hs_patch_config, 
                                  inp, 
                                  max_length, 
                                  end_token, 
                                  model, 
                                  tokenizer, 
                                  num_of_tokens,
                                  source_layer, 
                                  target_layer,
                                  do_sample = False,
                                  temperature = 1.0,
                                  visualize_confidence = False):
    
    # Copy target prompt's input token ids for generation (IDK why again)
    input_ids = inp["input_ids"].detach().clone().to(DEVICE)

    # Without this, we mostly get warnings
    model.set_attn_implementation('eager')

    # Freeze all gradient calculation
    with torch.no_grad():
        for step in range(max_length):
            patch_hooks = set_hs_patch_hooks(model, hs_patch_config, num_of_tokens) 
            outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
            remove_hooks(patch_hooks)
            
            # Extract logits for the last token
            # output.logits has shape                       # (1, seq_len, vocab_size)
            logits = outputs.logits[:, -1, :]               # (1, vocab_size)

            # Compute probablilities (before temp scaling)
            # Apply softmax to the last dim (which is vocab dim)
            raw_probs = torch.softmax(logits, dim = -1)     # (1, vocab_size)

            # Get the next token id using the logits
            if do_sample:
                # Apply temperature scaling
                logits = logits / temperature

                # Sample from the softmax distribution
                probs = torch.softmax(logits, dim = -1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1) # (1,)

            else:
                next_token_id = torch.argmax(logits, dim=-1)                       # (1, )

            # Concat the next token id to the input_ids for autoregressive generation
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

            # If the end_token is predicted, break out of the autoregression loop
            if next_token_id.item() == end_token:
                break

    generated_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    patched_pattern = (" x" * num_of_tokens).strip()
    return "".join(generated_text).split(patched_pattern)[-1]


def set_hs_patch_hooks(model, hs_patch_config, num_of_tokens):
    def patch_hs(name, position_hs):
        def hook(module, input, output):
            for position_, hs_ in position_hs:
                # (batch, sequence, hidden_state)
                output[0][position_ : position_ + num_of_tokens] = hs_
        return hook

    hooks = []
    for l in hs_patch_config:
        if model.config.model_type == 'llama':
            if l == -1:
                layer = model.model.embed_tokens
            else:
                layer = model.model.layers[l]

        else:
            raise ValueError(f"Unknown model: {model.config.model_type}")

        hooks.append(layer.register_forward_hook(
            patch_hs(f"patch_hs_{l}", hs_patch_config[l])
        ))

    return hooks
