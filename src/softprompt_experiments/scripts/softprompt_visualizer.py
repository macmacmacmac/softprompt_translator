import torch
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm
import json
from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.models.squishyprompt import SquishyPrompt
from softprompt_experiments.utils import (
    get_train_test_from_tokenized, 
    train_softprompt_from_tokenized,
    eval_softprompt,
    log_json
)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


"""
run me
sbatch -t 90 --output explainabilityv2-%j.out job.sh --experiment math_dataset_generatorv2 math_softprompt_generator math_explainabilityv2 --num_datasets 100 --save_directory ./datasets/math_datasetv2_same --epochs 8 --init "________"
"""

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_directory", type=str, default="./datasets/math_datasetv2_same")
    parser.add_argument("--visualizations_dir", type=str, default="./visualizations")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    args, _ = parser.parse_known_args(args_list)

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    SAVE_DIR = args.save_directory
    VISUALIZATION_DIR = args.visualizations_dir

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

    soft_embeds = []
    hardprompts = []
    for dataset_dir in tqdm(dataset_dirs):
        hardprompt = torch.load(
            os.path.join(dataset_dir,'dataset.pt'),
            weights_only=False
        )['hardprompt']
        hardprompts.append(hardprompt)

        softprompt = SoftPrompt(
            model=model, 
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            path_to_model=os.path.join(dataset_dir, "softprompt.pt")
        )
        with torch.no_grad():
            soft_embeds.append(softprompt.forward())

    centroid = torch.mean(torch.cat(soft_embeds, dim=0), dim=0)

    normalized_embeds = [(soft_emb - centroid) for soft_emb in soft_embeds]

    def text_to_embed(text):
        tokenized = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)
        embed = word_embeddings(tokenized.to(device))
        return embed.to(dtype)

    init_emb = text_to_embed("1 2 3 4 ")
    normalized_init_embeds = [(soft_emb - centroid + init_emb) for soft_emb in soft_embeds]

    # TODO:
    # visualize two PCA plots, 
    # the first shows soft_embeds (red) and model's word embeddings (blue)
    # the second shows normalized_embeds (red) and the model's word embeddings (blue)

    # -----------------------------
    # Mean-pool over sequence dim
    # -----------------------------

    # Each soft_emb: [1, soft_len, hidden_dim]
    soft_means = torch.cat(
        [soft_emb.mean(dim=1) for soft_emb in soft_embeds],
        dim=0
    ).cpu()  # [num_datasets, hidden_dim]

    normalized_means = torch.cat(
        [soft_emb.mean(dim=1) for soft_emb in normalized_embeds],
        dim=0
    ).cpu()  # [num_datasets, hidden_dim]

    normalized_init_means = torch.cat(
        [soft_emb.mean(dim=1) for soft_emb in normalized_init_embeds],
        dim=0
    ).cpu()  # [num_datasets, hidden_dim]


    print(soft_means.shape)
    print(normalized_means.shape)

    # Word embeddings: [vocab_size, hidden_dim]
    word_tokens = word_embeddings.weight.detach().cpu()

    # Optional subsampling for visualization
    MAX_WORD_TOKENS = 5000
    if word_tokens.shape[0] > MAX_WORD_TOKENS:
        idx = torch.randperm(word_tokens.shape[0])[:MAX_WORD_TOKENS]
        word_tokens = word_tokens[idx]

    # -----------------------------
    # PCA + plotting helper
    # -----------------------------

    def plot_pca(red_tokens, blue_tokens, title):
        """
        red_tokens:  [N, D]  (soft prompt means)
        blue_tokens: [M, D]  (word embeddings)
        """
        X = torch.cat([red_tokens, blue_tokens], dim=0).detach().float().numpy()

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        red_2d = X_2d[: red_tokens.shape[0]]
        blue_2d = X_2d[red_tokens.shape[0] :]

        plt.figure(figsize=(8, 6))
        plt.scatter(
            blue_2d[:, 0], blue_2d[:, 1],
            s=5, alpha=0.25, label="Word embeddings"
        )
        plt.scatter(
            red_2d[:, 0], red_2d[:, 1],
            s=60, alpha=0.9, label="Soft prompts"
        )
        plt.legend()
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()

        # sanitize filename
        filename = title.replace(" ", "_").replace(":", "")
        save_path = os.path.join(VISUALIZATION_DIR, f"{filename}.jpg")

        plt.savefig(save_path, dpi=300)
        plt.close()
    # -----------------------------
    # Plot 1: raw soft prompts
    # -----------------------------

    os.makedirs(VISUALIZATION_DIR, exist_ok=True)


    plot_pca(
        soft_means,
        word_tokens,
        title="PCA: Mean-Pooled Soft Prompts vs Word Embeddings"
    )

    plot_pca(
        normalized_means,
        word_tokens,
        title="PCA: Mean-Pooled Centered Soft Prompts vs Word Embeddings"
    )

    plot_pca(
        normalized_init_means,
        word_tokens,
        title="PCA: Mean-Pooled Init Centered Soft Prompts vs Word Embeddings"
    )

    example_hard0, example_hard1, example_hard2 = (
        hardprompts[0],
        hardprompts[1],
        hardprompts[2]

    )
    example_soft0, example_soft1, example_soft2 = (
        soft_embeds[0], 
        soft_embeds[1], 
        soft_embeds[2]
    )

    print(f"hardprompts used as examples: {(example_hard0, example_hard1, example_hard2)}")

    for soft_embed, hardprompt in zip(soft_embeds, hardprompts):
        print(f"Actual hardprompt: {hardprompt}")
        inputembeds = torch.cat([
            text_to_embed("What does this say:\n"),
            (soft_embed)
        ], dim=1)
        generated = tokenizer.decode(
            model.generate(
                inputs_embeds=inputembeds, 
                attention_mask=torch.ones((inputembeds.shape[0], inputembeds.shape[1])).to(device), 
                do_sample=False, 
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )[0],            
            skip_special_tokens=True
        )
        print(f"Generating w/ zeroshot, unnormalized: {generated}\n")

        normed_inputembeds = torch.cat([
            text_to_embed("Describe the expressions, here are examples:\n"),
            (example_soft0),
            text_to_embed(f" = {example_hard0}, "),
            (example_soft1),
            text_to_embed(f" = {example_hard1}, "),
            (example_soft2),
            text_to_embed(f" = {example_hard2}, "),
            (soft_embed),
            text_to_embed(f" = "),
        ], dim=1)
        generated = tokenizer.decode(
            model.generate(
                inputs_embeds=normed_inputembeds, 
                attention_mask=torch.ones((normed_inputembeds.shape[0], normed_inputembeds.shape[1])).to(device), 
                do_sample=False, 
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )[0],
            skip_special_tokens=True
        )
        print(f"Generating w/ fewshot, unnormalized: {generated}\n")
        inputembeds = torch.cat([
            text_to_embed("What does this say:\n"),
            (soft_embed - centroid)
        ], dim=1)
        generated = tokenizer.decode(
            model.generate(
                inputs_embeds=inputembeds, 
                attention_mask=torch.ones((inputembeds.shape[0], inputembeds.shape[1])).to(device), 
                do_sample=False, 
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )[0],            
            skip_special_tokens=True
        )
        print(f"Generating w/ zeroshot, normalized: {generated}\n")

        normed_inputembeds = torch.cat([
            text_to_embed("Describe the expressions, here are examples:\n"),
            (example_soft0 - centroid),
            text_to_embed(f" = {example_hard0}, "),
            (example_soft1 - centroid),
            text_to_embed(f" = {example_hard1}, "),
            (example_soft2 - centroid),
            text_to_embed(f" = {example_hard2}, "),
            (soft_embed - centroid),
            text_to_embed(f" = "),
        ], dim=1)
        generated = tokenizer.decode(
            model.generate(
                inputs_embeds=normed_inputembeds, 
                attention_mask=torch.ones((normed_inputembeds.shape[0], normed_inputembeds.shape[1])).to(device), 
                do_sample=False, 
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )[0],
            skip_special_tokens=True
        )
        print(f"Generating w/ fewshot, normalized: {generated}\n")

    # def get_activation(embeds, layer_idx):
    #     """
    #     Retrieves the activations at a given layer
        
    #     :embeds: A list of [1, seq_len, dim] input embeddings
    #     :layer_idx: Layer to retrieve the activation from
    #     """
    #     acts = []

    #     def hook(module, input, output):
    #         acts.append(output[:, :, :].detach().unsqueeze(0))

    #     handle = model.model.layers[layer_idx].register_forward_hook(hook)

    #     for embed in embeds:
    #         with torch.no_grad():
    #             model(inputs_embeds=embed,attention_mask=None)
    #     handle.remove()
    #     return torch.cat(acts)
    
    # layer_idx = 20
    
    # soft_activations = get_activation(soft_embeds, layer_idx)
    # print(soft_activations.shape)
    # layer_centroid = soft_activations.mean(dim=0)
    # normalized_activations = []
    # for soft_act in soft_activations:
    #     normalized = soft_act - layer_centroid
    #     normalized = normalized / normalized.norm()
    #     normalized_activations.append(normalized)

    # alpha = 5.0  # steering strength

    # injected = False

    # def inject_concept(layer_idx, concept_vector, alpha=5.0):
    #     nonlocal injected
    #     def hook(module, input, output):
    #         nonlocal injected
    #         h = output.clone()
    #         L = concept_vector.shape[1]
    #         if not injected and h.shape[1] >= L:
    #             h[:, -L:, :] += alpha * concept_vector
    #             injected = True
    #         return h
    #     return hook

    # handle = model.model.layers[layer_idx].register_forward_hook(
    #     inject_concept(layer_idx, soft_activations[0])
    # )

    # normed_inputembeds = torch.cat([
    #     text_to_embed("Describe the expressions, here are examples:\n"),
    #     example_soft0,
    #     text_to_embed(f" = {example_hard0}, "),
    #     example_soft1,
    #     text_to_embed(f" = {example_hard1}, "),
    #     example_soft2,
    #     text_to_embed(f" = {example_hard2}, "),
    #     text_to_embed("________"),
    # ], dim=1)

    # with torch.no_grad():
    #     out = model.generate(
    #             inputs_embeds=normed_inputembeds, 
    #             attention_mask=torch.ones((normed_inputembeds.shape[0], normed_inputembeds.shape[1])).to(device), 
    #             do_sample=False, 
    #             max_new_tokens=50,
    #             pad_token_id=tokenizer.eos_token_id
    #         )

    # handle.remove()
    # print(f"genertion: {tokenizer.decode(out[0], skip_special_tokens=True)}")

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









