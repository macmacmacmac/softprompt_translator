import sqlite3
import os
import argparse
import json
import re
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from tqdm import tqdm

SENTENCE_GENERATION_SYSTEM_PROMPT = """
You are an expert linguistic data generator creating training data for a classifier.

TASK:
Generate diverse text samples that describe, imply, or relate to the Target Keyword WITHOUT using that keyword.

LENGTH DISTRIBUTION (CRITICAL):
- At least 4 sentences MUST be 20-35 words long (detailed, descriptive sentences)
- 3-4 sentences should be medium length (10-19 words)
- 2-3 sentences can be short phrases (3-9 words)

REALISM REQUIREMENTS - Make text feel natural and messy:
- Sometimes skip punctuation at sentence ends
- Sometimes use improper punctuation (double periods.., misplaced commas, or missing commas)
- Vary capitalization: some sentences start lowercase, occasional CAPS for emphasis
- Include informal styles: contractions, fragments, trailing off...
- Mix formality levels (casual chat vs professional tone)

VARIETY REQUIREMENTS - Mix these styles:
- Long detailed sentences (20-35 words): elaborate thoughts, multi-clause statements, storytelling snippets
- Medium sentences (10-19 words): complete thoughts with some detail
- Short phrases (3-9 words): punchy expressions, quick reactions
- Incomplete thoughts: trailing off with "..."
- Questions: rhetorical or conversational

CONSTRAINTS:
1. NEVER use the target keyword, its root, or direct derivations.
2. Every sentence must be UNIQUE - no repetition across outputs.
3. Cover different contexts: everyday life, technical, emotional, professional
4. Output ONLY the JSON - no explanations, no filler
"""


def get_json_schema(num_sentences: int) -> str:
    """Generate a JSON schema for a specific number of sentences."""
    return json.dumps({
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": num_sentences,
                "maxItems": num_sentences
            }
        },
        "required": ["sentences"]
    })


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r'[\u4E00-\u9FFF\u3400-\u4DBF\u3000-\u303F]', text))


def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n",
        f"\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="./datasets/mapper_classification_datasets/DoD_2_5k_Mistral.sqlite")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    DB_PATH = args.db_path

    # Perform DB Repair
    repair_database(DB_PATH)


def repair_database(db_path):
    print(f"Connecting to database: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print(f"Successfully Connected!")

    # Constants matching the generator
    TARGET_SENTENCES_PER_DATASET = 500
    NUM_KEYWORDS = 5
    TARGET_SENTENCES_PER_KEYWORD = TARGET_SENTENCES_PER_DATASET // NUM_KEYWORDS  # 100
    MAX_CHUNK_SIZE = 10
    MIN_CHUNK_SIZE = 5  # Always request at least 5 sentences to avoid truncation issues

    # Identify all datasets that don't have exactly 500 sentences
    print('-' * 50)
    print("Scanning for incomplete datasets...")
    cursor.execute("""
        SELECT dataset_id, COUNT(sentence_id) as sentence_count
        FROM sentences
        GROUP BY dataset_id
        HAVING COUNT(sentence_id) < 500
    """)

    incomplete_datasets = cursor.fetchall()
    print(f"Found {len(incomplete_datasets)} incomplete datasets out of the original 5500.")

    # If no incomplete datasets found, we exit the script as no repair is needed
    if not incomplete_datasets:
        print("All datasets are complete! No repair needed.")
        conn.close()
        return

    # Build set of existing sentences for duplicate detection
    print("Loading existing sentences for duplicate detection...")
    cursor.execute("SELECT LOWER(TRIM(sentence)) FROM sentences")
    seen_sentences = set(row[0] for row in cursor.fetchall())
    print(f"Loaded {len(seen_sentences)} existing sentences.")

    # For each incomplete dataset, find how many sentences each keyword needs
    print('-' * 50)
    print("Analyzing sentence counts per keyword for incomplete datasets...")

    # Group tasks by request_size for efficient batch processing with same schema
    tasks_by_size = {i: [] for i in range(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE + 1)}

    for dataset_id, current_count in tqdm(incomplete_datasets, desc="Building repair tasks"):
        # Get keywords for this dataset with their current sentence counts
        cursor.execute("""
            SELECT k.keyword_id, k.keyword, COUNT(s.sentence_id) as sentence_count
            FROM keywords k
            LEFT JOIN sentences s ON k.keyword_id = s.keyword_id AND s.dataset_id = k.dataset_id
            WHERE k.dataset_id = ?
            GROUP BY k.keyword_id, k.keyword
        """, (dataset_id,))

        keyword_counts = cursor.fetchall()

        # For each keyword_id, keyword and its sentence count
        for keyword_id, keyword, count in keyword_counts:

            # Determine number of sentences needed to complete the dataset
            sentences_needed = TARGET_SENTENCES_PER_KEYWORD - count

            # While sentences needed are not fulfilled
            while sentences_needed > 0:
                # Determine how many we actually need for this task
                sentences_to_insert = min(sentences_needed, MAX_CHUNK_SIZE)

                # Always request at least MIN_CHUNK_SIZE to avoid truncation issues
                request_size = max(sentences_to_insert, MIN_CHUNK_SIZE)

                # Construct messages
                messages = [
                    {"role": "system", "content": SENTENCE_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Target Keyword: {keyword}. Generate a JSON with {request_size} unique sentences without using the keyword."}
                ]

                # Insert a task for current request size
                tasks_by_size[request_size].append({
                    "dataset_id": dataset_id,
                    "keyword_id": keyword_id,
                    "keyword": keyword,
                    "messages": messages,
                    "request_size": request_size,
                    "sentences_to_insert": sentences_to_insert  # Only insert this many
                })

                # Subtract sentences_to_insert from the sentences needed, as we have created a task for this
                sentences_needed -= sentences_to_insert

    # Determine total number of tasks to execute
    total_tasks = sum(len(tasks) for tasks in tasks_by_size.values())
    print(f"Created {total_tasks} repair tasks grouped by request size:")

    # For each (request_size, tasks) pair, print their info
    for size, tasks in tasks_by_size.items():
        if tasks:
            print(f"  Request {size} sentences: {len(tasks)} tasks")
    print('-' * 50)

    # If there are no tasks at all, then exit
    if total_tasks == 0:
        print("No repair tasks needed.")
        conn.close()
        return

    # Load Teacher Model using vLLM
    TEACHER_MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    print(f"Loading {TEACHER_MODEL_NAME} into vLLM...")
    llm = LLM(
        model=TEACHER_MODEL_NAME,
        tokenizer_mode="mistral",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        max_model_len=119712,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9
    )

    # Tracking counters
    success_count = 0
    json_failure_count = 0
    duplicate_count = 0
    non_english_sentence_count = 0
    sentences_inserted = 0

    VLLM_BATCH_SIZE = 10_000

    # Process each group of tasks with the same request size (same JSON schema)
    for request_size in range(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE + 1):

        # Retrieve all tasks of the current group
        group_tasks = tasks_by_size[request_size]

        # If there are no tasks for the current group, then proceed to the next group
        if not group_tasks:
            continue

        print(f"\n" + "=" * 50)
        print(f"Processing {len(group_tasks)} tasks requesting {request_size} sentence(s) each...")
        print("=" * 50)

        # Create sampling params with the correct JSON schema for this group
        sampling_params = SamplingParams(
            temperature=0.4,
            presence_penalty=0.5,
            max_tokens=1000,
            structured_outputs=StructuredOutputsParams(json=get_json_schema(request_size))
        )

        # Process in batches
        for i in tqdm(range(0, len(group_tasks), VLLM_BATCH_SIZE), desc=f"Processing {request_size}-sentence batches"):

            # Get the batch of tasks and create a batch of converations from it
            batch_tasks = group_tasks[i:i + VLLM_BATCH_SIZE]
            batch_conversations = [task["messages"] for task in batch_tasks]

            # Send the batch of conversations to the LLM for batch processing
            outputs = llm.chat(messages=batch_conversations, sampling_params=sampling_params)

            # Zip the LLM batch_tasks and the llm outputs and for each pair
            for task, output in zip(batch_tasks, outputs):

                # Get the generated text
                generated_text = output.outputs[0].text
                try:

                    # Try decoding to a json
                    data = json.loads(generated_text)

                    # Track how many sentences we've inserted for this task
                    task_inserted = 0
                    max_to_insert = task["sentences_to_insert"]

                    # For each sentence in the json's "sentences" field
                    for idx, sentence in enumerate(data.get("sentences", [])):

                        # Stop if we've inserted enough for this task
                        if task_inserted >= max_to_insert:
                            break

                        # Normalize the sentence
                        normalized_sentence = sentence.strip().lower()

                        # Skip duplicates
                        if normalized_sentence in seen_sentences:
                            duplicate_count += 1
                            continue

                        # Add the sentence in the seen sentences set
                        seen_sentences.add(normalized_sentence)

                        # TODO: This splitting is unfair right now
                        # 80/20 train/test split
                        split = "test" if idx % 10 >= 8 else "train"

                        # Track Chinese characters
                        if contains_chinese(sentence):
                            non_english_sentence_count += 1

                        cursor.execute(
                            "INSERT INTO sentences (dataset_id, keyword_id, sentence, split) VALUES (?, ?, ?, ?)",
                            (task["dataset_id"], task["keyword_id"], sentence, split)
                        )
                        sentences_inserted += 1
                        task_inserted += 1

                    success_count += 1

                except json.JSONDecodeError as e:
                    json_failure_count += 1
                    truncated_output = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
                    print(f"JSON FAILURE [{json_failure_count}]: dataset={task['dataset_id']}, keyword='{task['keyword']}', error={e}")
                    print(f"  Raw output: {truncated_output}")
                    continue

            # Commit after each batch
            conn.commit()
            print(f"Batch committed. Total sentences inserted so far: {sentences_inserted}")

    print(f"\nRepair complete! Inserted {sentences_inserted} new sentences.")

    # Print validation summary
    print("\n" + "=" * 50)
    print("REPAIR SUMMARY")
    print("=" * 50)
    print(f"Successful generations: {success_count}")
    print(f"JSON parsing failures: {json_failure_count}")
    print(f"Duplicate sentences skipped: {duplicate_count}")
    print(f"Sentences with Chinese characters: {non_english_sentence_count}")
    print(f"Total sentences inserted: {sentences_inserted}")

    # Sanity check - verify all datasets now have 500 sentences
    print("\n" + "=" * 50)
    print("SANITY CHECK")
    print("=" * 50)

    cursor.execute("""
        SELECT dataset_id, COUNT(sentence_id) as sentence_count
        FROM sentences
        GROUP BY dataset_id
        HAVING COUNT(sentence_id) < 500
    """)

    still_incomplete = cursor.fetchall()

    if still_incomplete:
        print(f"WARNING: {len(still_incomplete)} datasets are still incomplete!")
        print("Datasets still missing sentences:")
        for dataset_id, count in still_incomplete[:20]:  # Show first 20
            print(f"  dataset_id={dataset_id}: {count} sentences (missing {500 - count})")
        if len(still_incomplete) > 20:
            print(f"  ... and {len(still_incomplete) - 20} more")
    else:
        print("SUCCESS: All datasets now have 500 sentences!")

    # Also verify no dataset has more than 500
    cursor.execute("""
        SELECT dataset_id, COUNT(sentence_id) as sentence_count
        FROM sentences
        GROUP BY dataset_id
        HAVING COUNT(sentence_id) > 500
    """)

    over_500 = cursor.fetchall()
    if over_500:
        print(f"\nWARNING: {len(over_500)} datasets have MORE than 500 sentences!")
        for dataset_id, count in over_500[:10]:
            print(f"  dataset_id={dataset_id}: {count} sentences")

    conn.close()
    print("\nRepair and verification complete!")
