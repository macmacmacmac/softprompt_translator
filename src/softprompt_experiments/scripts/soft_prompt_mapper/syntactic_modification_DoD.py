import argparse
import os
import sqlite3
import random
from tqdm import tqdm


def perform_syntactic_changes(sentence: str) -> str:
    transformed_sentence = sentence

    # Determine whether to capitalize or not
    if random.randint(0, 1) == 1:
        transformed_sentence = transformed_sentence.capitalize()

    # Determine how many periods to add
    num_periods = random.randint(0, 2)
    transformed_sentence = transformed_sentence + ('.' * num_periods)

    # Determine how many words to keep (between 75% - 100%)
    transformed_sentence_list = transformed_sentence.split()
    total_num_words = len(transformed_sentence_list)
    if total_num_words == 0:
        return transformed_sentence

    num_words_to_keep = random.randint((total_num_words * 3) // 4, total_num_words)
    max_start_idx = total_num_words - num_words_to_keep
    rand_start_idx = random.randint(0, max_start_idx)
    reduced_sentence_list = transformed_sentence_list[rand_start_idx: rand_start_idx + num_words_to_keep]

    return " ".join(reduced_sentence_list)




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
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    DB_PATH = args.db_path
    SEED = args.seed

    # Set the Random Seed for this experiment
    random.seed(SEED)
    print(f"Using random seed: {SEED}")

    # Connect to the DB
    print(f"Connecting to database: {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    print(f"Successfully Connected!")

    collision_count = 0


    # Get all dataset_ids
    print("Fetching all dataset ids ...")
    cursor.execute("""
        SELECT dataset_id
        FROM datasets
    """)
    dataset_ids = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(dataset_ids)} dataset ids.")

    # For each dataset
    for dataset_id in tqdm(dataset_ids, desc = "Transforming DoD"):
        # Get the sentence table for this dataset
        cursor.execute(
            """
            SELECT sentence_id, dataset_id, keyword_id, sentence, split
            FROM sentences
            WHERE dataset_id = ?
            """, (dataset_id,))
        
        sentence_table = cursor.fetchall()

        for sentence_id, _, keyword_id, sentence, split in sentence_table:
        # for sentence_id, _, keyword_id, sentence, split in sentence_table[:5]:
            
            # print(keyword_id)
            # print(f"Original Sentence:\n {sentence}")
            transformed_sentence = perform_syntactic_changes(sentence)
            # print(f"Transformed Sentence:\n {transformed_sentence}\n")

            # Update Sentence in the DB
            try:
                cursor.execute(
                    """
                    UPDATE sentences
                    SET sentence = ?
                    WHERE sentence_id = ? AND dataset_id = ?
                    """, (transformed_sentence, sentence_id, dataset_id)
                )
            except sqlite3.IntegrityError:
                # Skip rows that would violate global sentence uniqueness.
                collision_count += 1

        conn.commit()


    if collision_count:
        print(f"Skipped {collision_count} sentence updates due to UNIQUE collisions.")

    print("Running VACUUM to compact the database file ...")
    cursor.execute("VACUUM")
    print("VACUUM completed.")


    # Close the DB connect
    conn.close()

    




    
