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

# --scripts dataset_nl_custom squishyprompt_generator_regression softprompt_lm_inversion --model_name 'meta-llama/Llama-2-7b-hf' --save_directory ./datasets/logit_prior_inv --verbose --lambd 0.1

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
        logits_prior = LM_inverter_prior(model, tokenizer, word_embeddings)
        squishyprompt = SquishyPrompt(
            logits_prior=logits_prior,
            model=model, 
            init=init,
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            num_tokens=NUM_TOKENS,
        )

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
        train_loss, test_loss, entropy = train_softprompt_from_tokenized(
            squishyprompt, LR, EPOCHS, train_loader, test_loader, 
            verbose=VERBOSE, verbose_level=VERBOSE_LEVEL,
            entropy_reg_constant=LAMBD, logger=logger
        )

        # if verbose: generate sample output predictions using eval_softprompt
        if VERBOSE:
            outputs = eval_softprompt_regression(squishyprompt, test_dataset, dataset_dir)
            logger.info(outputs)
            performance = {
                'hardprompt':hardprompt,
                'train loss':train_loss,
                'test_loss':test_loss,
                'entropy':entropy,
                'outputs': outputs
            }
            log_json(os.path.join(dataset_dir,'softprompt_performance.json'), performance)
        else:
            performance = {
                'hardprompt':hardprompt,
                'train loss':train_loss,
                'test_loss':test_loss,
                'entropy':entropy,
            }
            log_json(os.path.join(dataset_dir,'softprompt_performance.json'), performance)

        squishyprompt.save_softprompt(dataset_dir)

    logger.info(
        f"{'='*100}\n\t\t\t\tCompleted script: {exp_name}\n{'='*100}"
    )









