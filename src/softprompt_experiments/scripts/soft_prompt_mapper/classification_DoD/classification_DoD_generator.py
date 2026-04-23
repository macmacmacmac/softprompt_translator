import os
import argparse
import random
import sqlite3
import json
import nltk
from nltk.corpus import brown
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import re
from tqdm import tqdm
import pickle

SENTENCE_GENERATION_SYSTEM_PROMPT = """
You are an expert linguistic data generator creating training data for a classifier.

TASK:
Generate diverse text samples that describe, imply, or relate to the Target Keyword WITHOUT using the keyword itself.

STYLE & LENGTH DISTRIBUTION (CRITICAL):
Mix the lengths and styles across your output to simulate varied human text:
- Long (3-4 sentences): Elaborate, multi-clause thoughts, descriptive scenes, or storytelling snippets.
- Medium (3-4 sentences): Complete thoughts with moderate detail.
- Short (2-3 sentences): Punchy expressions, quick reactions, fragments, or rhetorical questions.

VARIETY REQUIREMENTS:
- Include varied contexts: everyday life, technical, emotional, professional.
- Mix formality levels: from casual conversational tones to highly professional statements.

HARD CONSTRAINTS:
1. NEVER use the target keyword, its root, or direct derivations.
2. Every sentence must be conceptually UNIQUE.
"""

# Define the exact JSON schema we want vLLM to force the model to follow
# This guarantees an object with a "sentences" array of strings.
JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 10,
            "maxItems": 10
        }
    },
    "required": ["sentences"]
})


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


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    # CJK Unified Ideographs + Extension A + Symbols/Punctuation
    return bool(re.search(r'[\u4E00-\u9FFF\u3400-\u4DBF\u3000-\u303F]', text))


def setup_database(db_path):
    """Initializes the SQLite schema designed for PyTorch dataloading speed."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            hard_prompt TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS keywords (
            keyword_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            keyword TEXT NOT NULL,
            label_index INTEGER NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        );
        CREATE TABLE IF NOT EXISTS sentences (
            sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            keyword_id INTEGER NOT NULL,
            sentence TEXT NOT NULL UNIQUE,
            split TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
            FOREIGN KEY (keyword_id) REFERENCES keywords(keyword_id)
        );
        CREATE INDEX IF NOT EXISTS idx_sentences_dataset_split 
        ON sentences(dataset_id, split);
    """)
    conn.commit()
    return conn, cursor


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
    parser.add_argument("--mini_dataset_size", type=int, default=500)
    parser.add_argument("--num_of_datasets", type=int, default=5500)
    parser.add_argument("--save_directory", type=str, default="./datasets/mapper_classification_datasets")
    parser.add_argument("--keyword_pickle_path", type=str, default="./datasets/mapper_classification_datasets/keywords_DoD3_10k_2.pkl")
    parser.add_argument("--db_name", type=str, default="DoD_3_5k.sqlite")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    TEACHER_MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    # TEACHER_MODEL_NAME = "OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym"
    # TEACHER_MODEL_NAME = "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ"
    MINI_DATASET_SIZE = args.mini_dataset_size
    NUM_OF_DATASETS = args.num_of_datasets
    SAVE_DIRECTORY = args.save_directory
    DB_NAME = args.db_name
    KEYWORD_PICKLE_PATH = args.keyword_pickle_path
    VALID_CATEGORY_REGEX = r"^[a-zA-Z0-9\s_\-]+$"

    # Other variables
    NUM_KEYWORDS = 5
    SENTENCES_PER_KEYWORD = MINI_DATASET_SIZE // NUM_KEYWORDS

    # Ask Teacher LLM to generate CHUNK_SIZE sentences at a time to maintain high quality
    CHUNK_SIZE = 10 
    NUM_CHUNKS_PER_KEYWORD = max(1, SENTENCES_PER_KEYWORD // CHUNK_SIZE)

    # Delete the Dataset if it already exists
    db_path = os.path.join(SAVE_DIRECTORY, DB_NAME)
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Setup the SQLite DB
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    conn, cursor = setup_database(db_path)

    # Load Existing Keywords
    semantic_keyword_sets = set()
    print(f"Loading keywords from {KEYWORD_PICKLE_PATH} ...")
    try:
        with open(KEYWORD_PICKLE_PATH, 'rb') as f:
            semantic_keyword_sets = pickle.load(f)
        print(f"Successfully loaded {len(semantic_keyword_sets)} existing unique categories.")
    except Exception as e:
        print(f"Error loading existing pickle file: {e}. Exiting ...")
        exit(-1)

    # Convert to list and predictably shuffle for ML reproducibility
    keyword_list = sorted(list(semantic_keyword_sets))
    random.seed(42) # Guarantees the exact same random sequence every run
    random.shuffle(keyword_list)

    # Find all keyword sets which are valid.
    # Valid keyword sets are those which:
    # - have 5 classes
    # - have category comprised of alphanumeric and special characters
    # - have target leakage
    valid_keyword_sets = []
    
    for category, classes in keyword_list:
        clean_category = category.strip().lower()
        lower_classes = [c.lower() for c in classes]

        # 1. Check for exactly 5 classes
        if len(classes) != 5:
            continue
            
        # 2. Check for Target Leakage
        if clean_category in lower_classes:
            continue

        # 3. Check for valid english categories
        if not re.search(VALID_CATEGORY_REGEX, clean_category):
            continue
            
        # If it passes all checks, add it to our final generation queue!
        valid_keyword_sets.append((clean_category, classes))

        # Break as soon as we reach the requested number of datasets
        if len(valid_keyword_sets) == NUM_OF_DATASETS:
            break

    if len(valid_keyword_sets) < NUM_OF_DATASETS:
        print(f"Warning: Only found {len(valid_keyword_sets)} valid datasets out of requested {NUM_OF_DATASETS}.")

    # Maintain a Maps of mini-dataset -> keywords
    dod_keyword_maps = []

    print("Initializing mini datasets, using loaded semantic keyword sets, in SQLite ...")

    # For each keyword set
    for i, (category, classes) in enumerate(valid_keyword_sets):

        # Sort to create a deterministic baseline
        keywords = sorted(list(classes))
        
        # Seed with 'i' so every dataset shuffles differently, but predictably!
        random.seed(42 + i) 
        random.shuffle(keywords)

        # Add the entry for dataset id and its associated category and keywords
        dod_keyword_maps.append({
            "dataset_id": i,
            "category": category,
            "keywords": keywords
        })

        # Init a Hard Prompt for this mini dataset and insert it into the DB's datasets table
        hard_prompt = f"Classify the following sentence as: {', '.join(keywords)}"
        cursor.execute("INSERT INTO datasets (dataset_id, category, hard_prompt) VALUES (?, ?, ?)", (i, category, hard_prompt))

        # Insert keywords and related data into the keywords table
        for label_idx, kw in enumerate(keywords):
            cursor.execute("INSERT INTO keywords (dataset_id, keyword, label_index) VALUES (?, ?, ?)", (i, kw, label_idx))

    # Commit all the inserts into the DB
    conn.commit()

    # Load Teacher Model using vLLM
    print(f"Loading {TEACHER_MODEL_NAME} into vLLM...")
    llm = LLM(
        model = TEACHER_MODEL_NAME,
        tokenizer_mode = "mistral",
        # tokenizer = "mistralai/Mistral-Small-24B-Instruct-2501",
        # quantization="awq",

        quantization="bitsandbytes",
        load_format="bitsandbytes",
        # max_model_len = 119712,

        tensor_parallel_size = 1,

        enable_prefix_caching = True, 
        max_model_len = 4096, 
        max_num_seqs = 256,

        gpu_memory_utilization = 0.95, # Let vLLM use 90% of GPU VRAM for KV Cache
    )

    # Setup SamplingParams for the vLLM along with a guided json schema for guided decoding
    sampling_params = SamplingParams(
        temperature = 0.4, 
        presence_penalty = 0.5,
        max_tokens = 2048,
        structured_outputs = StructuredOutputsParams(json = JSON_SCHEMA)
    )

    # Generate Sentences for each dataset
    generation_tasks = []
    print("Creating prompts for sentence generation")
    for dataset in tqdm(dod_keyword_maps, desc="Creating Prompts"):
        dataset_id = dataset["dataset_id"]
        category = dataset["category"]

        for kw in dataset["keywords"]:
            cursor.execute("SELECT keyword_id FROM keywords WHERE dataset_id = ? AND keyword = ?", (dataset_id, kw))
            keyword_id = cursor.fetchone()[0]

            # Chunking Strategy: Send multiple small requests instead of a massive one
            for _ in range(NUM_CHUNKS_PER_KEYWORD):

                # Construct user prompt
                user_prompt = (
                    f"Category Context: '{category}'.\n"
                    f"Target Keyword: '{kw}'.\n"
                    f"Generate a JSON with {CHUNK_SIZE} unique sentences that describe or relate to this specific keyword "
                    f"within the context of the given category. Do NOT use the keyword itself."
                )

                # Construct messages to be sent to the LLM
                messages = [
                    {"role": "system", "content": SENTENCE_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]

                generation_tasks.append({
                    "dataset_id": dataset_id,
                    "keyword_id": keyword_id,
                    "keyword": kw,
                    "messages": messages
                })

    # Process and save 10,000 prompts at a time
    VLLM_BATCH_SIZE = 10_000
    total_chunks = (len(generation_tasks) + VLLM_BATCH_SIZE - 1) // VLLM_BATCH_SIZE
    success_count = 0

    # Tracking counters for validation issues
    json_failure_count = 0
    non_english_sentence_count = 0
    non_english_dataset_ids = set()
    duplicate_count = 0
    seen_sentences = set()

    # Tracking counters for sentence length distribution
    total_sentence_count = 0
    long_sentence_count = 0  # 15+ words
    very_long_sentence_count = 0  # 20+ words
    
    print(f"Submitting {len(generation_tasks)} tasks to vLLM in chunks of {VLLM_BATCH_SIZE}...")
    
    # Iterate through the tasks in chunks
    for i in tqdm(range(0, len(generation_tasks), VLLM_BATCH_SIZE), desc = "Processing vLLM requests"):
        batch_tasks = generation_tasks[i : i + VLLM_BATCH_SIZE]
        batch_conversations = [task["messages"] for task in batch_tasks]
        
        current_chunk = (i // VLLM_BATCH_SIZE) + 1
        
        print(f"\n" + "="*50)
        print(f"Processing Chunk {current_chunk} of {total_chunks}...")
        print(f"="*50)

        # Generate the JSONs for this specific chunk
        outputs = llm.chat(messages = batch_conversations, sampling_params = sampling_params)

        # Parse and Insert the outputs for this chunk
        for task, output in zip(batch_tasks, outputs):
            generated_text = output.outputs[0].text

            try:
                # Try to parse the generated text as JSON
                data = json.loads(generated_text)

                # Simple 80/20 Train/Test split assignment
                for idx, sentence in enumerate(data.get("sentences", [])):
                    # Skip duplicates using normalized comparison
                    normalized_sentence = sentence.strip().lower()
                    if normalized_sentence in seen_sentences:
                        duplicate_count += 1
                        continue
                    seen_sentences.add(normalized_sentence)

                    split = "test" if idx % 10 >= 8 else "train"

                    # Track sentence length distribution
                    word_count = len(sentence.split())
                    total_sentence_count += 1
                    if word_count >= 15:
                        long_sentence_count += 1
                    if word_count >= 20:
                        very_long_sentence_count += 1

                    # Track Chinese characters (still insert for analysis)
                    if contains_chinese(sentence):
                        non_english_sentence_count += 1
                        non_english_dataset_ids.add(task["dataset_id"])

                    cursor.execute(
                        "INSERT INTO sentences (dataset_id, keyword_id, sentence, split) VALUES (?, ?, ?, ?)",
                        (task["dataset_id"], task["keyword_id"], sentence, split)
                    )

                success_count += 1

            # Catch any JSON decoding errors
            except json.JSONDecodeError as e:
                json_failure_count += 1
                truncated_output = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
                print(f"JSON FAILURE [{json_failure_count}]: dataset={task['dataset_id']}, keyword='{task['keyword']}', error={e}")
                print(f"  Raw output: {truncated_output}")
                continue

        # Commit the data to the DB immediately (once this batch is processed)
        conn.commit()
        print(f"Chunk {current_chunk} securely committed to SQLite disk. Total valid sentences so far: {success_count * CHUNK_SIZE}")

    print(f"\nDone! Successfully generated data for {success_count * CHUNK_SIZE} sentences across {NUM_OF_DATASETS} datasets.")

    # Print validation summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"JSON parsing failures: {json_failure_count}")
    print(f"Duplicate sentences skipped: {duplicate_count}")
    print(f"Sentences with Chinese characters: {non_english_sentence_count}")
    print(f"Datasets affected by Chinese characters: {len(non_english_dataset_ids)}")
    if non_english_dataset_ids:
        sample_ids = list(non_english_dataset_ids)[:10]
        print(f"  Sample affected dataset_ids: {sample_ids}")

    # Print sentence length distribution
    print("\n" + "-"*50)
    print("SENTENCE LENGTH DISTRIBUTION")
    print("-"*50)
    print(f"Total sentences: {total_sentence_count}")
    print(f"Sentences with 15+ words: {long_sentence_count} ({100*long_sentence_count/max(1,total_sentence_count):.1f}%)")
    print(f"Sentences with 20+ words: {very_long_sentence_count} ({100*very_long_sentence_count/max(1,total_sentence_count):.1f}%)")

    # Close the connection when completely finished
    conn.close()