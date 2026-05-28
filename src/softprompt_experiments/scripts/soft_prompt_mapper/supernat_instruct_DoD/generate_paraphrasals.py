import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from tqdm import tqdm
import gc
import json
import torch

def run(args_list):
    exp_name = os.path.basename(__file__)
    print("="*100, f"\n\t\t\tRunning script: {exp_name}\n", "="*100, "\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="SoftPromptTranslator/SUPER-NATURALINSTRUCTIONS-english-filtered")
    parser.add_argument("--new_dataset_path", type=str, default="SoftPromptTranslator/SUPER-NATURALINSTRUCTIONS-english-filtered-10x-augmented-enriched")
    parser.add_argument("--teacher_model", type=str, default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--tokenizer_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num_paraphrasals", type=int, default=10, help="Number of paraphrasals to generate per instruction")
    args, _ = parser.parse_known_args(args_list)

    # Parse all arguments into Global Variables
    DATASET_PATH = args.dataset_path
    NEW_DATASET_PATH = args.new_dataset_path
    TEACHER_MODEL = args.teacher_model
    TOKENIZER_MODEL = args.tokenizer_model
    NUM_PARAPHRASALS = args.num_paraphrasals

    # Load the original dataset
    print(f"\nLoading dataset {DATASET_PATH}...")
    dataset_dict = load_dataset(DATASET_PATH)

    # Drop reduced_instruction(s) column if it already exists
    for split in dataset_dict.keys():
        if "reduced_instruction" in dataset_dict[split].column_names:
            dataset_dict[split] = dataset_dict[split].remove_columns("reduced_instruction")
        if "reduced_instructions" in dataset_dict[split].column_names:
            dataset_dict[split] = dataset_dict[split].remove_columns("reduced_instructions")

    # Extract unique instructions globally
    print(f"\nExtracting unique instructions...")

    # We will build a dict mapping instruction to a set of task_names
    # str -> set()
    unique_instructions = {}
    
    # Iterate over train and test splits to find all unique instructions
    for split in dataset_dict.keys():
        for row in dataset_dict[split]:
            inst = row["instruction"]
            if inst not in unique_instructions:
                unique_instructions[inst] = set()
            unique_instructions[inst].add(row["task_name"])

    print(f"\nFound {len(unique_instructions)} unique instructions across the dataset.")

    # Tokenize and filter tasks that exceed the threshold
    print(f"\nLoading Tokenizer {TOKENIZER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    instructions_to_paraphrase = []  # list of instructions
    paraphrased_map = {}             # original_instruction -> paraphrased_instruction
    original_token_counts = {}       # original_instruction -> token_count

    print(f"\nChecking token lengths...")
    for instruction in tqdm(unique_instructions.keys(), desc="Tokenizing"):
        # We don't need the actual tokens, just the count
        token_count = len(tokenizer.encode(instruction, add_special_tokens=False))
        original_token_counts[instruction] = token_count
        instructions_to_paraphrase.append(instruction)

    # Generate paraphrases
    print(f"\nLoading {TEACHER_MODEL} into vLLM...")
    llm = LLM(
        model=TEACHER_MODEL,
        tokenizer_mode="mistral",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        max_model_len=4096,
        max_num_seqs=256,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        n=NUM_PARAPHRASALS,
        temperature=0.5 if NUM_PARAPHRASALS > 1 else 0.3,
        presence_penalty=0.5,
        max_tokens=300,
    )

    # TODO: Refine this further after spot checking
    # system_prompt = (
    #     "You are an expert at paraphrasing instructions. "
    #     "Your task is to rephrase the following instruction using different words and sentence structures while keeping the exact same meaning and level of detail. "
    #     "CRITICAL: You MUST explicitly preserve all specific classes, exact tags, labels, output formats, and special syntax constraints. "
    #     "CRITICAL: Do NOT alter or remove specific mappings between concepts (e.g., specifying which sentence is the premise, exact sentence counts, positional logic, or structural relationships). "
    #     "Output ONLY the paraphrased instruction text and absolutely nothing else. Do NOT include phrases like 'Here is the paraphrased version'."
    # )

    system_prompt = (
        "You are an expert at writing prompt instructions."
        "Your task is to paraphrase the following instruction, stripping away redundant wording while enriching the description to provide specific details while remaining concise."
        "For instance, if the original instruction is about classifying text, add informative details about specific features to look for."
        "CRITICAL: You MUST explicitly preserve all specific classes, exact tags, labels, output formats, and special syntax constraints. "
        "CRITICAL: Do NOT oversimplify or remove specific mappings between concepts (e.g., specifying which sentence is the premise, exact sentence counts, positional logic, or structural relationships). "
        "Use AT MOST of 4 to 5 short sentences. Strip away ONLY filler words, long narrative examples, and conversational redundant explanations. "
        "Output ONLY the paraphrased instruction text and absolutely nothing else. Do NOT include phrases like 'Here is the paraphrased version'."
    )

    messages_list = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Instruction:\n{inst}"}
        ]
        for inst in instructions_to_paraphrase
    ]

    print(f"\nGenerating paraphrased instructions in batches...")
    outputs = llm.chat(messages=messages_list, sampling_params=sampling_params)

    for i, output in enumerate(outputs):
        instruction = instructions_to_paraphrase[i]

        # Strip whitespace just in case the model padded the output
        paraphrased_texts = [out.text.strip() for out in output.outputs]
        paraphrased_map[instruction] = paraphrased_texts

    # We delete the LLM object to free up GPU VRAM as soon as generation is done
    del llm
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nSaving a log of the paraphrased instructions to 'paraphrased_instructions_log.json'...")
    log_data = []
    for instruction in instructions_to_paraphrase:
        paraphrased_instructions = paraphrased_map[instruction]
        paraphrased_token_counts = [len(tokenizer.encode(ri, add_special_tokens=False)) for ri in paraphrased_instructions]
        log_data.append({
            "task_names": list(unique_instructions[instruction]),
            "original_instruction": instruction,
            "original_token_count": original_token_counts[instruction],
            "paraphrased_instructions": paraphrased_instructions,
            "paraphrased_token_counts": paraphrased_token_counts
        })
    
    with open("paraphrased_instructions_log.json", "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    
    # Map the reduced instructions back to the main dataset
    print(f"\nMapping paraphrased instructions back to the main dataset rows...")
    
    def add_paraphrased_instructions(example):
        example["paraphrased_instructions"] = paraphrased_map[example["instruction"]]
        return example

    # Using map with batched=False here is extremely fast for dictionary lookups
    dataset_dict = dataset_dict.map(add_paraphrased_instructions, desc="Applying mapping")

    # Push to HuggingFace
    print(f"\nPushing updated dataset to {NEW_DATASET_PATH}...")
    dataset_dict.push_to_hub(NEW_DATASET_PATH)
    print(f"\nSuccessfully pushed to Hugging Face at path {NEW_DATASET_PATH}")
    
