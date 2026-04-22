"""

"""

import torch
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm

import numpy as np

from softprompt_experiments.models.squishyprompt import SquishyPrompt
from softprompt_experiments.models.priors.GMM_prior import GMM_prior
from softprompt_experiments.models.priors.LM_inverter_prior import LM_inverter_prior

from softprompt_experiments.utils import (
    get_train_test_from_tokenized, 
    train_softprompt_from_tokenized,
    eval_softprompt,
    eval_softprompt_regression,
    log_json
)

from peft import PromptTuningInit, PromptTuningConfig, get_peft_model
import logging

# --scripts dataset_nl_custom squishyprompt_generator_regression softprompt_lm_inversion --model_name 'meta-llama/Llama-2-7b-hf' --save_directory ./datasets/logit_prior_inv_1 --verbose --lambd 0.1

def run(args_list):
    exp_name = os.path.basename(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--num_tokens", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambd", type=float, default=0.0)
    parser.add_argument("--no_auto_split",dest="auto_split",action="store_false")
    parser.set_defaults(auto_split=True)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_dataset")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="enable verbose logging")
    parser.set_defaults(verbose=False)
    parser.add_argument("--verbose_level", type=str, default='epoch')
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    
    args, _ = parser.parse_known_args(args_list)
    
    MODEL_NAME = args.model_name
    SAVE_DIR = args.save_directory
    AUTO_SPLIT = args.auto_split
    VERBOSE = args.verbose
    VERBOSE_LEVEL = args.verbose_level
    INIT = args.init
    LR = args.lr
    EPOCHS = args.epochs
    NUM_TOKENS = args.num_tokens
    BATCH_SIZE = args.batch_size
    SEED = args.seed
    LAMBD = args.lambd

    logging.getLogger().setLevel(logging.WARNING)

    logger = logging.getLogger(f"{exp_name}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            # logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            logging.Formatter("%(message)s")
        )

        # File handler
        file_handler = logging.FileHandler(os.path.join(SAVE_DIR,f"{exp_name}.log"), mode="w")
        file_handler.setFormatter(
            # logging.Formatter("%(levelname)s - %(message)s")
            logging.Formatter("%(message)s")
        )
        file_handler.flush = file_handler.stream.flush

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.propagate = False

    # logging.getLogger("transformers").setLevel(logging.INFO)
    # logging.getLogger("torch").setLevel(logging.INFO)

    logger.info(
        f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}"
    )

    logger.info("Args: %s", vars(args))    

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
    dataset_dirs = [
        os.path.join(SAVE_DIR, f"dataset_{i}")
        for i in range(num_datasets)
    ]
    if num_datasets > 0:
        logger.info(f"\nFound ({num_datasets}) datasets in directory")
    else:
        raise ValueError("path to directory has no datasets")

    for dataset_dir in tqdm(dataset_dirs):
        # load dataset
        _, test_dataset, train_loader, test_loader = get_train_test_from_tokenized(
            dataset_dir,
            BATCH_SIZE,
            train_portion = 0.8,
            auto_split=AUTO_SPLIT
        )
        del _

        # initialize softprompt
        if SEED is not None:
            vocab_size = word_embeddings.num_embeddings
            rng = np.random.default_rng(seed=SEED)
            init_token_ids = torch.from_numpy(
                rng.integers(0, vocab_size, size=NUM_TOKENS, dtype=np.int64)
            ).to(model.device)
            init = tokenizer.decode(init_token_ids)
        else:
            init = INIT
        
        # logger.info("Initial tokens: ", init)
        # logits_prior = GMM_prior()
        logits_prior = LM_inverter_prior(model, tokenizer, word_embeddings, NUM_TOKENS)
        squishyprompt = SquishyPrompt(
            logits_prior=logits_prior,
            model=model, 
            init=init,
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            num_tokens=NUM_TOKENS,
        )
        value_head = torch.nn.Sequential(
            torch.nn.Linear(word_embeddings.embedding_dim, word_embeddings.embedding_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(word_embeddings.embedding_dim, 1)
        )
        value_head = value_head.to(dtype=dtype)
        value_head.to(device)
        if AUTO_SPLIT:
            hardprompt = torch.load(
                os.path.join(dataset_dir,'dataset.pt'),
                weights_only=False
            )['hardprompt']
        else:
            hardprompt = torch.load(
                os.path.join(dataset_dir,'train_dataset.pt'),
                weights_only=False
            )['hardprompt']

        
        # begin training
        if VERBOSE:
            logger.info(hardprompt)
        # train_loss, test_loss, entropy = train_softprompt_from_tokenized(
        #     squishyprompt, LR, EPOCHS, train_loader, test_loader, 
        #     verbose=VERBOSE, verbose_level=VERBOSE_LEVEL,
        #     entropy_reg_constant=LAMBD, logger=logger
        # )
        model = squishyprompt._model
        tokenizer = squishyprompt._tokenizer
        word_embeddings = squishyprompt._word_embeddings
        dtype = model.dtype
        device = model.device

        # Freeze LM
        model.requires_grad_(False)
        squishyprompt.to(device)

        # Only train the softprompt parameters
        optimizer = torch.optim.AdamW(squishyprompt.parameters(), lr=LR)
        v_optimizer = torch.optim.AdamW(value_head.parameters(), lr=LR)

        def prep_inputs(input_ids, labels):
            batchsize = input_ids.size(0)
            # softprompt embeddings
            sp_embeds = squishyprompt.forward()   # [1, soft_len, dim]
            sp_embeds = sp_embeds.expand(batchsize, -1, -1)
            input_embeds = word_embeddings(input_ids).to(dtype=dtype)  #
            full_embeds = torch.cat([sp_embeds, input_embeds], dim=1)

            # Shift labels to align with concatenated softprompt
            pad_prefix = torch.full(
                (labels.shape[0], sp_embeds.shape[1]),
                -100,
                dtype=labels.dtype,
                device=device
            )
            labels_adjusted = torch.cat([pad_prefix, labels], dim=1)

            # build and shift attention mask
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            attn_prefix = torch.ones(
                (labels.shape[0], sp_embeds.shape[1]),
                dtype=labels.dtype,
                device=device
            )
            attention_mask = torch.cat([attn_prefix, attention_mask], dim=1)
            return full_embeds, labels_adjusted, attention_mask
        def forward_pass(input_ids, labels):
            full_embeds, labels_adjusted, attention_mask = prep_inputs(input_ids, labels)
            normal_loss, ppo_loss = squishyprompt.loss_fn(full_embeds, labels_adjusted, attention_mask, return_entropy=True)
            return normal_loss, ppo_loss
        def extract_x_batch(input_ids, attention_mask, labels, pad_token_id):
            """
            Returns:
                x_input_ids: [B, T] (y tokens replaced with PAD)
                x_attn_mask:  [B, T] (1 for x tokens, 0 otherwise)
            """

            # x region = where labels == -100 AND not padding
            x_mask = (labels == -100) & attention_mask.bool()   # [B, T]

            # replace y tokens with pad token (safe since we clone)
            x_input_ids = input_ids.clone()
            x_input_ids[~x_mask] = pad_token_id

            # attention mask only keeps x region
            x_attn_mask = x_mask.long()

            return x_input_ids, x_attn_mask
        def get_last_valid_hidden(hidden_states, attn_mask):
            """
            hidden_states: [B, T, D]
            attn_mask:     [B, T]
            """

            # lengths of valid (x) tokens per batch
            lengths = attn_mask.sum(dim=1) - 1  # [B]

            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)

            last_hidden = hidden_states[batch_idx, lengths]

            return last_hidden
        def pred_value(input_ids, labels):
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            x_input_ids, x_attn_mask = extract_x_batch(
                input_ids,
                attention_mask,
                labels,
                tokenizer.pad_token_id
            )

            # forward frozen base model
            with torch.no_grad():
                hidden = model(
                    input_ids=x_input_ids,
                    attention_mask=x_attn_mask,
                    output_hidden_states=True
                ).hidden_states[-1]

            # correct last-valid-token pooling
            last_hidden = get_last_valid_hidden(hidden, x_attn_mask)

            # value prediction
            value_pred = value_head(last_hidden).squeeze(-1)
            return value_pred

        final_train_loss = 0.0
        final_test_loss = 0.0
        ACTOR_UPDATES_PER_ROLLOUT = 4
        CRITIC_UPDATES_PER_ROLLOUT = 8
        for epoch in range(EPOCHS):

            squishyprompt.train()

            train_loss = 0.0

            rollout_buffer = []   # parallel array #1 (data)
            traj_buffer = []      # parallel array #2 (z', log_q_old, advantage)

            # =========================
            # PHASE 1: ROLLOUT
            # =========================
            for i, batch in enumerate(train_loader):
                logger.info(f"EPOCH: {epoch}, [ROLLOUT] batch {i}")

                input_ids, labels = [b.to(device) for b in batch]

                full_embeds, labels_adjusted, attention_mask = prep_inputs(input_ids, labels)

                traj = squishyprompt.sample_old_trajectories(
                    full_embeds,
                    labels_adjusted,
                    attention_mask
                )
                logger.info(f"{traj['log']}")

                # store STATE side (for recomputation during PPO update)
                rollout_buffer.append({
                    "input_ids": input_ids.detach(),
                    "labels": labels.detach(),
                })

                # store TRAJECTORY side
                traj_buffer.append(traj)

            # =========================
            # PHASE 3: Critic UPDATE
            # =========================
            # for _ in range(CRITIC_UPDATES_PER_ROLLOUT):
            #     for i, batch in enumerate(train_loader):
            #         print(f"[CRITIC] batch {i}")

            #         input_ids, labels = [b.to(device) for b in batch]
            #         value_pred = pred_value(input_ids, labels)

            #         # reward from rollout
            #         traj = traj_buffer[i]
            #         reward = traj["reward"]
            #         traj["value"] = value_pred.detach()

            #         # critic loss
            #         value_loss = (value_pred - reward.detach()).pow(2).mean()
            #         print(f"critic loss: {value_loss}")
            #         value_loss.backward()
            #         v_optimizer.step()
            #         v_optimizer.zero_grad()            
            # =========================
            # PHASE 3: PPO UPDATE
            # =========================
            for _ in range(ACTOR_UPDATES_PER_ROLLOUT):
                # optional: shuffle indices for PPO updates
                indices = torch.randperm(len(traj_buffer))

                for i in indices:
                    logger.info(f"EPOCH: {epoch}, [PPO UPDATE] batch {i}")
                    
                    batch_state = rollout_buffer[i]
                    traj = traj_buffer[i]

                    input_ids = batch_state["input_ids"]
                    labels = batch_state["labels"]

                    full_embeds, labels_adjusted, attention_mask = prep_inputs(input_ids, labels)
                    
                    # value_pred = traj["value"]
                    value_pred = None

                    normal_loss, ppo_loss = squishyprompt.loss_fn(
                        full_embeds,
                        labels_adjusted,
                        attention_mask,
                        trajectories=traj,
                        baseline=value_pred
                    )

                    loss = normal_loss - LAMBD * ppo_loss
                    logger.info(
                        f"\tNormal loss: {normal_loss:.3f},\n"
                        f"\tPPO loss: {ppo_loss:.3f}"
                    )
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    train_loss += loss.item()                
        # ---- evaluation ----
        squishyprompt.eval()

        test_loss = 0.0
        test_ppo_loss = 0.0

        if VERBOSE:
            with torch.no_grad():

                for batch in test_loader:
                    input_ids, labels = [b.to(device) for b in batch]

                    loss, ppo_loss = forward_pass(input_ids, labels)

                    test_loss += loss.item()
                    test_ppo_loss += ppo_loss.item()

            final_train_loss = train_loss / len(train_loader)
            final_test_loss = test_loss / len(test_loader)
            avg_ppo_loss = test_ppo_loss / len(test_loader)

            msg = (
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {final_train_loss:.4f} | "
                f"Test Loss: {final_test_loss:.4f} | "
                f"PPO Loss: {avg_ppo_loss:.4f}"
            )

            if logger:
                logger.info(msg)
            else:
                print(msg)

        # if verbose: generate sample output predictions using eval_softprompt
        if VERBOSE:
            outputs = eval_softprompt_regression(squishyprompt, test_dataset, dataset_dir)
            logger.info(outputs)
            performance = {
                'hardprompt':hardprompt,
                'train loss':train_loss,
                'test_loss':test_loss,
                'ppo_loss':avg_ppo_loss,
                'outputs': outputs
            }
            log_json(os.path.join(dataset_dir,'softprompt_performance.json'), performance)
        else:
            performance = {
                'hardprompt':hardprompt,
                'train loss':train_loss,
                'test_loss':test_loss,
                'ppo_loss':avg_ppo_loss,
            }
            log_json(os.path.join(dataset_dir,'softprompt_performance.json'), performance)

        squishyprompt.save_softprompt(dataset_dir)

    logger.info(
        f"{'='*100}\n\t\t\t\tCompleted script: {exp_name}\n{'='*100}"
    )









