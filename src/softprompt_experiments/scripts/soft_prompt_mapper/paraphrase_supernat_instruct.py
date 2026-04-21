import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import gc

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
    print(f"Loading dataset {DATASET_PATH}...")
    dataset_dict = load_dataset(DATASET_PATH)

    # Extract unique instructions globally (assuming task_name corresponds 1:1 to an instruction)
    print("Extracting unique task instructions...")

    # We will build a dictionary: task_name -> original_instruction
    unique_tasks = {}
    
    # Iterate over train and test splits to find all unique task_names and their instructions
    for split in dataset_dict.keys():
        for row in dataset_dict[split]:
            task_name = row["task_name"]
            if task_name not in unique_tasks:
                unique_tasks[task_name] = row["instruction"]

    print(f"Found {len(unique_tasks)} unique tasks across the dataset.")

    # Tokenize and filter tasks that exceed the threshold
    print(f"Loading Tokenizer {TOKENIZER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    tasks_to_paraphrase = {}  # task_name -> instruction
    paraphrased_map = {}      # task_name -> reduced_instruction

    print("Checking token lengths...")
    for task_name, instruction in tqdm(unique_tasks.items(), desc="Tokenizing"):
        # We don't need the actual tokens, just the count
        token_count = len(tokenizer.encode(instruction, add_special_tokens=False))
        if token_count > TOKEN_THRESHOLD:
            tasks_to_paraphrase[task_name] = instruction
        else:
            # If it's already short enough, the reduced instruction is the same as the original
            paraphrased_map[task_name] = instruction

    print(f"Tasks needing paraphrasing (> {TOKEN_THRESHOLD} tokens): {len(tasks_to_paraphrase)}")

    # Generate paraphrases if needed
    if len(tasks_to_paraphrase) > 0:
        print(f"Loading {TEACHER_MODEL} into vLLM...")
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

        task_names_list = list(tasks_to_paraphrase.keys())
        messages_list = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Instruction:\n{tasks_to_paraphrase[tn]}"}
            ]
            for tn in task_names_list
        ]

        print("Generating paraphrased instructions in batches...")
        outputs = llm.chat(messages=messages_list, sampling_params=sampling_params)

        for i, output in enumerate(outputs):
            task_name = task_names_list[i]
            # Strip whitespace just in case the model padded the output
            reduced_text = output.outputs[0].text.strip()
            paraphrased_map[task_name] = reduced_text

        # We delete the LLM object to free up GPU VRAM just in case it takes up too much memory during the map step
        del llm
        gc.collect()

    # Map the reduced instructions back to the main dataset
    print("Mapping paraphrased instructions back to the main dataset rows...")
    
    def add_reduced_instruction(example):
        example["reduced_instruction"] = paraphrased_map[example["task_name"]]
        return example

    # Using map with batched=False here is extremely fast for dictionary lookups
    dataset_dict = dataset_dict.map(add_reduced_instruction, desc="Applying mapping")

    # Push to HuggingFace
    print(f"Pushing updated dataset to {DATASET_PATH}...")
    dataset_dict.push_to_hub(DATASET_PATH)
