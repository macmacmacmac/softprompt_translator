import sqlite3
import os
import argparse


def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n",
        f"\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="./datasets/mapper_classification_datasets/DoD_3_5k.sqlite")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    DB_PATH = args.db_path

    # Perform DB Analysis 
    analyze_database(DB_PATH)



def analyze_database(db_path):
    print(f"Connecting to database: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print(f"Successfully Connected!")

    # Find Count of sentences in the Datasets
    cursor.execute("""
        SELECT COUNT(sentence_id) as sentence_count
        FROM sentences
    """)
    count = cursor.fetchall()
    print(f"Found total {count} sentences in the DB {db_path}")
    
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

    # Print sentence counts for each incomplete dataset
    if incomplete_datasets:
        print("Sentence counts per incomplete dataset:")
        for dataset_id, count in incomplete_datasets:
            if count < 450:
                print(f"  dataset_id={dataset_id}: {count} sentences")
    print('-' * 50)


    # Identify all datasets that train/test imbalance
    print('-' * 50)
    print("Scanning for datasets with train/test imbalance ...")
    cursor.execute("""
        SELECT dataset_id, COUNT(split) as train_split_count
        FROM sentences
        WHERE split = 'train'
        GROUP BY dataset_id
        HAVING COUNT(split) != 400
    """)

    imbalanced_datasets = cursor.fetchall()
    print(f"Found {len(imbalanced_datasets)} datasets with train/test imbalance.")

    # Print train split count for each imbalanced dataset
    if imbalanced_datasets:
        print("Train split counts per imbalanced dataset:")
        for dataset_id, count in imbalanced_datasets[:10]:
            print(f"  dataset_id={dataset_id}: {count} train sentences")
        if len(imbalanced_datasets) > 10:
            print(f"  ... and {len(imbalanced_datasets) - 10} more")
    print('-' * 50)

    
    # # Identify all datasets that have duplicate sentences
    # print('-' * 50)
    # print("Scanning for datasets with duplicate sentences...")
    # cursor.execute("""
    #     SELECT DISTINCT dataset_id
    #     FROM sentences
    #     GROUP BY dataset_id, sentence
    #     HAVING COUNT(*) > 1
    # """)

    # dup_sentence_dataset_ids = [row[0] for row in cursor.fetchall()]
    # print(f"Found {len(dup_sentence_dataset_ids)} datasets with duplicate sentences.")
    # print('-' * 50)

    # # Print example of duplicate sentences from one dataset
    # if dup_sentence_dataset_ids:
    #     print('-' * 50)
    #     example_dataset_id = dup_sentence_dataset_ids[0]
    #     print(f"Example: Duplicate sentences in dataset_id={example_dataset_id}")
    #     cursor.execute("""
    #         SELECT sentence_id, keyword_id, sentence, split
    #         FROM sentences
    #         WHERE dataset_id = ? AND sentence IN (
    #             SELECT sentence
    #             FROM sentences
    #             WHERE dataset_id = ?
    #             GROUP BY sentence
    #             HAVING COUNT(*) > 1
    #         )
    #         ORDER BY sentence, sentence_id
    #     """, (example_dataset_id, example_dataset_id))

    #     for row in cursor.fetchall():
    #         print(f"  sentence_id={row[0]}, keyword_id={row[1]}, split={row[3]}")
    #         print(f"    sentence: {row[2][:100]}...")
    #     print('-' * 50)


    # Check DB for less than 5 keyword counts
    # Identify all datasets that don't have exactly 5 keywords
    print('-' * 50)
    print("Scanning for datasets with less than 5 keywords ...")
    cursor.execute("""
        SELECT dataset_id, COUNT(keyword_id) as keyword_count
        FROM keywords
        GROUP BY dataset_id
        HAVING COUNT(keyword_id) < 5
    """)

    incomplete_datasets = cursor.fetchall()
    print(f"Found {len(incomplete_datasets)} datasets with less than 5 keywords.")

    # Print keyword counts for each incomplete dataset
    if incomplete_datasets:
        print("Keyword counts per incomplete dataset:")
        for dataset_id, count in incomplete_datasets:
            print(f"  dataset_id={dataset_id}: {count} keywords")
    print('-' * 50)


    conn.close()
    print("Analysis complete!")