import os
import json
import pickle
from typing import List, Annotated
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import argparse
from tqdm import tqdm
import random
import nltk
from nltk.corpus import brown
# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')

# ┌───────────────────────────────────────────────┐
# │             DEFINE THE JSON SCHEMA            │
# └───────────────────────────────────────────────┘
# We use a Regex pattern to mathematically force the model to output 
# strictly alphabetic, single words. It physically cannot generate a space.
StrictSingleWord = Annotated[str, Field(pattern=r"^[A-Za-z]+$")]

class CategoryItem(BaseModel):
    category: str = Field(description="The broad domain or topic name")
    classes: List[StrictSingleWord] = Field(
        min_length=5, 
        max_length=5, 
        description="Exactly 5 mutually exclusive class labels. MUST BE SINGLE WORDS ONLY."
    )

class DatasetBatch(BaseModel):
    datasets: List[CategoryItem] = Field(description="A list of generated classification datasets")
    
# Define the exact JSON schema we want vLLM to force the model to follow.
JSON_SCHEMA = json.dumps(DatasetBatch.model_json_schema())

# Hardcode the target classes from the InSPEcT paper
FORBIDDEN_VOCAB_SET = {
    # SST2 and SST5 Classes
    "positive", "negative", "terrible", "bad", "neutral", "good", "great",

    # AGNews Classes
    "world", "sports", "business", "technology"

    # Subj Classes
    "objective", "subjective", 

    # TREC Classes
    "abbreviation", "entity", "description", "human", "location", "number"
}

def get_safe_keywords(target_pool_size = 15000, restrict_by_forbidden_vocab = True):
    nltk.download('brown')

    forbidden_vocab_set = FORBIDDEN_VOCAB_SET if restrict_by_forbidden_vocab else set()
    
    # Get standard nouns and adjectives
    tagged_words = [(word.lower(), tag) for word, tag in brown.tagged_words()]
    valid_words = [
        word for word, tag in tagged_words 
        if (tag.startswith('NN') or tag.startswith('JJ')) and word.isalpha()
    ]
    
    # Filter by frequency to ensure the words are common enough for an LLM to understand
    freq_dist = nltk.FreqDist(valid_words)
    
    safe_pool = []
    for word, _ in freq_dist.most_common():
        if word not in forbidden_vocab_set and len(word) > 3: # Skip tiny words
            safe_pool.append(word)
            
        if len(safe_pool) >= target_pool_size:
            break
            
    return safe_pool


# Driver Code
def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n",
        f"\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_datasets", type=int, default=15_000)
    parser.add_argument("--json_processing_batch_size", type=int, default=10)
    parser.add_argument("--keyword_pickle_path", type=str, default="./datasets/mapper_classification_datasets/keywords_DoD3_10k_2.pkl")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    TEACHER_MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    JSON_PROCESSING_BATCH_SIZE = args.json_processing_batch_size
    NUM_OF_DATASETS = args.num_of_datasets
    KEYWORD_PICKLE_PATH = args.keyword_pickle_path

    # ┌───────────────────────────────────────────────┐
    # │             LOAD EXISTING DATA                │
    # └───────────────────────────────────────────────┘
    semantic_dataset = set()
    seen_categories = set()
    if os.path.exists(KEYWORD_PICKLE_PATH):
        print(f"Found existing pickle file at {KEYWORD_PICKLE_PATH}. Loading data...")
        try:
            with open(KEYWORD_PICKLE_PATH, 'rb') as f:
                semantic_dataset = pickle.load(f)

            for category, _ in semantic_dataset:
                seen_categories.add(category.strip().lower())

            print(f"Successfully loaded {len(semantic_dataset)} existing unique categories.")
        except Exception as e:
            print(f"Error loading existing pickle file: {e}. Starting fresh.")
    else:
        print(f"No existing data found at {KEYWORD_PICKLE_PATH}. Starting fresh.")


    # ┌───────────────────────────────────────────────┐
    # │                 MODEL PREP                    │
    # └───────────────────────────────────────────────┘
    print(f"Loading {TEACHER_MODEL_NAME} into VRAM via vLLM...")

    # Load Teacher Model using vLLM
    llm = LLM(
        model = TEACHER_MODEL_NAME,
        tokenizer_mode = "mistral",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        max_model_len = 32768,
        tensor_parallel_size = 1,
        gpu_memory_utilization = 0.9 # Let vLLM use 90% of GPU VRAM for KV Cache
    )

    # Setup SamplingParams for vLLM with guided decoding via JSON schema.
    sampling_params = SamplingParams(
        temperature = 0.5,
        max_tokens = 4096,
        structured_outputs = StructuredOutputsParams(json = JSON_SCHEMA)
    )

    # Create prompt structures.
    system_prompt = (
        "You are an expert Machine Learning dataset curator. Your task is to generate diverse, "
        "semantically cohesive classification tasks. CRITICAL RULES:\n"
        "1. Every single class label MUST be exactly ONE word. No spaces, no hyphens.\n"
        "2. Do NOT squish multi-word concepts together. If it requires two words (like 'Pad Thai'), DO NOT use it.\n"
        "3. Every word MUST be a real, correctly spelled dictionary word or standard industry acronym."
    )

    # user_prompt = (
    #     f"Generate exactly {JSON_PROCESSING_BATCH_SIZE} distinct classification categories. "
    #     "Vary the domains wildly. Do not repeat categories. Remember: ONE SINGLE, CORRECTLY SPELLED WORD per class."
    # )

    # Build the list of chat requests. Each request asks for one JSON batch.
    num_prompts = NUM_OF_DATASETS // JSON_PROCESSING_BATCH_SIZE
    generation_tasks = []

    # Generate a massive pool of 15,000 safe dictionary words
    print("Generating NLTK lexical seeds...")
    safe_pool = get_safe_keywords(restrict_by_forbidden_vocab=False)

    for _ in range(num_prompts):

        # Sample 3 completely random, unrelated words from the dictionary
        random_seeds = random.sample(safe_pool, 3)
        seed_string = ", ".join(random_seeds)

        # Use the random seed words as an abstract creative anchor
        user_prompt = (
            f"Generate exactly {JSON_PROCESSING_BATCH_SIZE} distinct classification categories. "
            f"To ensure absolute diversity, use the following random seed words as abstract inspiration: '{seed_string}'. "
            f"You do NOT need to use these specific words as categories. Instead, let their concepts, related industries, or themes guide your generation. "
            f"CRITICAL RULES:\n"
            f"1. You MUST NOT use the exact seed words as categories.\n"
            f"2. Every single class label MUST be exactly ONE word. No spaces, no hyphens.\n"
            f"3. Do NOT squish multi-word concepts together."
        )


        generation_tasks.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        })


    print("\n" + "="*80)
    print(f"Submitting {num_prompts} chat requests to vLLM...")
    print("="*80 + "\n")

    # Submit all prepared chat requests in one vLLM call.
    batch_conversations = [task["messages"] for task in generation_tasks]
    if batch_conversations:
        all_outputs = llm.chat(messages = batch_conversations, sampling_params = sampling_params)
    else:
        all_outputs = []

    # ┌───────────────────────────────────────────────┐
    # │                DATA EXTRACTION                │
    # └───────────────────────────────────────────────┘
    print("\nExtracting outputs and merging with existing data...")
    for output in tqdm(all_outputs):
        generated_text = output.outputs[0].text
        
        try:
            # Try to load the generated text as a JSON
            parsed_data = json.loads(generated_text)
            
            for item in parsed_data.get("datasets", []):
                category = item['category']
                clean_category = category.strip().lower()

                # Check if we have seen this category name before
                if clean_category not in seen_categories:
                    clean_classes = [c.lower() for c in item['classes']]
                    classes_frozenset = frozenset(clean_classes)
                    
                    # Add to our master set (duplicates will automatically be ignored by the set logic)
                    semantic_dataset.add((category, classes_frozenset))
                    seen_categories.add(clean_category)
                
        except json.JSONDecodeError as e:
            print(f"Warning: Skipped a JSON parse error: {e}")


    # Print some initial examples of keywords generated
    print("\nSample Generated Categories:")
    for category, classes in list(semantic_dataset)[:10]:
        print(f"Category: {category}")
        print(f"Classes: {classes}\n")

    # Create directory for pickle file, if it not exists already
    output_dir = os.path.dirname(KEYWORD_PICKLE_PATH)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving {len(semantic_dataset)} unique cohesive categories to {KEYWORD_PICKLE_PATH}...")

    # Save to Pickle file
    with open(KEYWORD_PICKLE_PATH, 'wb') as f:
        pickle.dump(semantic_dataset, f)
        
    print("Tight Keyword generation complete! Ready for DoD Generation Now!")