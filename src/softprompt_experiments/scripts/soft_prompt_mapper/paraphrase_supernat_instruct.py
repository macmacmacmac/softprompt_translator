import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import gc
import json

def run(args_list):
    exp_name = os.path.basename(__file__)
    print("="*100, f"\n\t\t\tRunning script: {exp_name}\n", "="*100, "\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="Suryanshg/SUPER-NATURALINSTRUCTIONS-english")
    parser.add_argument("--teacher_model", type=str, default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--tokenizer_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--token_threshold", type=int, default=100, help="Max tokens allowed; if higher, we paraphrase.")
    args, _ = parser.parse_known_args(args_list)

    # Parse all arguments into Global Variables
    DATASET_PATH = args.dataset_path
    TEACHER_MODEL = args.teacher_model
    TOKENIZER_MODEL = args.tokenizer_model
    TOKEN_THRESHOLD = args.token_threshold

    # Load the original dataset
    print(f"\nLoading dataset {DATASET_PATH}...")
    dataset_dict = load_dataset(DATASET_PATH)

    # Drop reduced_instruction column if it already exists
    for split in dataset_dict.keys():
        if "reduced_instruction" in dataset_dict[split].column_names:
            dataset_dict[split] = dataset_dict[split].remove_columns("reduced_instruction")

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
    paraphrased_map = {}             # original_instruction -> reduced_instruction

    print(f"\nChecking token lengths...")
    for instruction in tqdm(unique_instructions.keys(), desc="Tokenizing"):
        # We don't need the actual tokens, just the count
        token_count = len(tokenizer.encode(instruction, add_special_tokens=False))
        if token_count > TOKEN_THRESHOLD:
            instructions_to_paraphrase.append(instruction)
        else:
            # If it's already short enough, the reduced instruction is the same as the original
            paraphrased_map[instruction] = instruction

    print(f"\nInstructions needing paraphrasing (> {TOKEN_THRESHOLD} tokens): {len(instructions_to_paraphrase)}")

    # Generate paraphrases if needed
    if len(instructions_to_paraphrase) > 0:
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
            gpu_memory_utilization=0.95,
        )

        sampling_params = SamplingParams(
            temperature=0.3,
            presence_penalty=0.5,
            max_tokens=TOKEN_THRESHOLD,  # Force model to keep it short
        )

        # TODO: Refine this further after spot checking
        system_prompt = (
            "You are an expert at simplifying and extremely condensing instructions. "
            "Your task is to paraphrase the following instruction into a highly concise statement. "
            "Extract ONLY the core objective, input format, and output constraints. "
            "Use a maximum of 2 to 3 short sentences and strip away all filler words, long examples, and redundant explanations. "
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
            inst = instructions_to_paraphrase[i]
            # Strip whitespace just in case the model padded the output
            reduced_text = output.outputs[0].text.strip()
            paraphrased_map[inst] = reduced_text

        print(f"\nSaving a log of the paraphrased instructions to 'paraphrased_instructions_log.json'...")
        log_data = []
        for inst in instructions_to_paraphrase:
            log_data.append({
                "task_names": list(unique_instructions[inst]),
                "original_instruction": inst,
                "reduced_instruction": paraphrased_map[inst]
            })
        
        with open("paraphrased_instructions_log.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)

        # We delete the LLM object to free up GPU VRAM just in case it takes up too much memory during the map step
        del llm
        gc.collect()

    # Map the reduced instructions back to the main dataset
    print(f"\nMapping paraphrased instructions back to the main dataset rows...")
    
    def add_reduced_instruction(example):
        example["reduced_instruction"] = paraphrased_map[example["instruction"]]
        return example

    # Using map with batched=False here is extremely fast for dictionary lookups
    dataset_dict = dataset_dict.map(add_reduced_instruction, desc="Applying mapping")

    # Push to HuggingFace
    print(f"\nPushing updated dataset to {DATASET_PATH}...")
    dataset_dict.push_to_hub(DATASET_PATH)
    print(f"\nSuccessfully pushed to Hugging Face at path {DATASET_PATH}")
