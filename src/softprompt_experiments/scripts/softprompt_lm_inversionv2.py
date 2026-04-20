import torch
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm

from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.models.squishyprompt import SquishyPrompt
from softprompt_experiments.models.LM_inverter import LM_inverter, load_model
from softprompt_experiments.utils import (
    get_train_test_from_tokenized, 
    log_json
)

import json


from typing import Dict, Optional, Tuple
import torch
import logging


def run(args_list):
    exp_name = os.path.basename(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_onetok")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--no_auto_split",dest="auto_split",action="store_false")
    parser.set_defaults(auto_split=True)

    args, _ = parser.parse_known_args(args_list)

    MODEL_NAME = args.model_name
    SAVE_DIR = args.save_directory
    BATCH_SIZE = args.batch_size
    AUTO_SPLIT = args.auto_split

    logging.getLogger().setLevel(logging.WARNING)

    logger = logging.getLogger(f"{exp_name}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s - %(message)s")
        )

        # File handler
        file_handler = logging.FileHandler(os.path.join(SAVE_DIR,f"{exp_name}.log"), mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(levelname)s - %(message)s")
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

    inversion_model = load_model(model, tokenizer)
    inversion_model.to(device)
    gen_kwargs = {
        "early_stopping": False,
        "num_beams": 1,
        "do_sample": False,
        "no_repeat_ngram_size": 0,
        'min_length': 1,
        'max_length': 128
    }
    sample_text = "Task: find the smallest number in the following list:\n"
    logger.info(f"Sample text: {sample_text}")    
    def sanity_decode(text):
        tokenized_text = tokenizer(
            text, 
            return_tensors='pt',
            add_special_tokens=False
        ).to(device)
        output = model(**tokenized_text)
        inversion = inversion_model.generate_from_output(
            output,
            gen_kwargs,
            attention_mask=tokenized_text['attention_mask'],
        )[0]
        logger.info(f"inversion tokens: {inversion}")
        decoded_inversion = inversion_model.tokenizer.decode(inversion,skip_special_tokens=True)
        return decoded_inversion
    
    logger.info(f"Decoded sample text: {sanity_decode(sample_text)}")    

    # from types import MethodType
    # # Get dataset sub directories
    # dataset_dirs = []
    # for entry in os.scandir(SAVE_DIR):
    #     if entry.is_dir():  # Check if the entry is a directory
    #         if "dataset_" in entry.name:
    #             dataset_dirs.append(entry.path)
    # num_datasets = len(dataset_dirs)
    # dataset_dirs = [
    #     os.path.join(SAVE_DIR, f"dataset_{i}")
    #     for i in range(num_datasets)
    # ]

    # if num_datasets > 0:
    #     logger.info(f"\nFound ({num_datasets}) datasets in directory")
    # else:
    #     raise ValueError("path to directory has no datasets")
    
    # trainer.sanity_decode((
    #     "What is the correct answer? List: 1, 2, 3. Answer:"
    # ))


    # for dataset_dir in tqdm(dataset_dirs):
    #     train_dataset, test_dataset, train_loader, test_loader = get_train_test_from_tokenized(
    #         dataset_dir,
    #         BATCH_SIZE,
    #         train_portion = 0.8,
    #         auto_split=AUTO_SPLIT
    #     )

    #     hardprompt = torch.load(
    #             os.path.join(dataset_dir,'dataset.pt'),
    #             weights_only=False
    #     )['hardprompt']
    #     logger.info(f"\n\n\nHardprompt: {hardprompt}")

    #     def call_just_embedding_model(
    #         self,
    #         input_ids: torch.Tensor,
    #         attention_mask: torch.Tensor,
    #     ) -> torch.Tensor:
    #         embedder = inversion_model.embedder

    #         model_output = embedder(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #         )
        
    #         return self._process_embedder_output(model_output, attention_mask)

    #     def call_softprompt_embedding_model(
    #         self,
    #         input_ids: torch.Tensor,
    #         attention_mask: torch.Tensor,
    #     ) -> torch.Tensor:
    #         embedder = inversion_model.embedder

    #         inputs_str = self.embedder_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    #         tokenized_input = self.embedder_tokenizer(inputs_str[0], return_tensors='pt',add_special_tokens=False).to(embedder.device)
    #         tokenized_input_ids = tokenized_input['input_ids']
    #         # tokenized_input_ids = torch.cat(
    #         #     [tokenized_input_ids, tokenizer(" ", return_tensors='pt',add_special_tokens=False).input_ids.to(embedder.device)], 
    #         #     dim=-1
    #         # )

    #         softprompt = SoftPrompt(
    #             model=inversion_model.embedder, 
    #             tokenizer=tokenizer, 
    #             word_embeddings=None, 
    #             path_to_model=os.path.join(dataset_dir,'softprompt.pt')
    #         )

    #         embedder.eval()

    #         word_embedding = embedder.get_input_embeddings()
    #         inputs_embeds = word_embedding(tokenized_input_ids)

    #         sp_embeds = softprompt.forward()   # [1, soft_len, dim]
    #         # fake_str = "The capital city of Thailand is Bangkok. "
    #         # tokenized_fakeinput = self.embedder_tokenizer(fake_str, return_tensors='pt',add_special_tokens=False).to(embedder.device)['input_ids']
    #         # fake_embeds = word_embedding(tokenized_fakeinput)
    #         # sp_embeds = fake_embeds

    #         sp_embeds = sp_embeds.expand(len(inputs_embeds), -1, -1) #[batchsize, soft_len, dim]
    #         full_embs = torch.cat([sp_embeds,inputs_embeds],dim=1)
    #         sp_attn_mask = torch.ones(full_embs.size()[:-1], device=full_embs.device, dtype=torch.long)

    #         model_output = embedder(
    #             # input_ids=tokenized_input['input_ids'],
    #             inputs_embeds=full_embs,
    #             attention_mask=sp_attn_mask
    #         )
    #         # return self._process_embedder_output(model_output, full_attention_mask)

    #         # model_output = embedder(
    #         #     input_ids=input_ids,
    #         #     attention_mask=attention_mask,
    #         # )
        
    #         return self._process_embedder_output(model_output, sp_attn_mask)

    #     random_idxs = torch.randint(0, len(test_dataset), (args.num_samples,))
    #     for idx in random_idxs:
    #         labels = test_dataset[idx][1].to(inversion_model.embedder.device)
    #         full_ids = test_dataset[idx][0].to(inversion_model.embedder.device)
    #         mask = (labels==-100).to(inversion_model.embedder.device)
    #         antimask = (labels!=-100).to(inversion_model.embedder.device)

    #         tokenized_text = full_ids[mask].to(inversion_model.embedder.device)
    #         input_text = embedder_tokenizer.decode(tokenized_text, skip_special_tokens=True)

    #         logger.info(f"INPUT: {input_text}")

    #         inversion_model.call_embedding_model = MethodType(call_softprompt_embedding_model, inversion_model)
    #         output_str = trainer.sanity_decode((
    #             input_text
    #         ))
    #         logger.info(f"\t<Softprompt Decoding Begin>{output_str}<Softprompt Decoding End>")
    #         inversion_model.call_embedding_model = MethodType(call_just_embedding_model, inversion_model)
    #         output_str = trainer.sanity_decode((
    #             input_text
    #         ))
    #         logger.info(f"\t<Control Decoding Begin>{output_str}<Control Decoding End>\n")


    logger.info(
        f"{'='*100}\n\t\t\t\tCompleted script: {exp_name}\n{'='*100}"
    )









