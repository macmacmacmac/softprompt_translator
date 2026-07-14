import os
import argparse
import random
import sqlite3
import pandas as pd
from tqdm import tqdm


def fetch_dataset_ids(cursor):
    cursor.execute("SELECT dataset_id FROM datasets")
    return [row[0] for row in cursor.fetchall()]


def datasets_table_has_category(cursor):
    cursor.execute("PRAGMA table_info(datasets)")
    columns = [row[1] for row in cursor.fetchall()]
    return "category" in columns


def fetch_dataset_metadata(cursor, dataset_id, has_category):
    if has_category:
        cursor.execute("SELECT category, hard_prompt FROM datasets WHERE dataset_id = ?", (dataset_id,))
        return cursor.fetchone()

    cursor.execute("SELECT hard_prompt FROM datasets WHERE dataset_id = ?", (dataset_id,))
    (hard_prompt,) = cursor.fetchone()
    return None, hard_prompt


def fetch_keywords(cursor, dataset_id):
    cursor.execute(
        "SELECT keyword, label_index FROM keywords WHERE dataset_id = ? ORDER BY label_index",
        (dataset_id,)
    )
    return cursor.fetchall()


def fetch_split_rows(cursor, dataset_id, split):
    cursor.execute("""
        SELECT s.sentence, k.label_index
        FROM sentences s
        JOIN keywords k ON s.keyword_id = k.keyword_id
        WHERE s.dataset_id = ? AND s.split = ?
    """, (dataset_id, split))
    return cursor.fetchall()


def write_tsv(rows, path):
    df = pd.DataFrame(rows, columns=["sentence", "label"])
    df.to_csv(path, sep="\t", index=False)


def write_classes_md(path, dataset_id, category, hard_prompt, keywords):
    lines = [f"# Dataset {dataset_id}", ""]
    if category is not None:
        lines += [f"**Category:** {category}", ""]
    lines += [
        f"**Hard Prompt:** {hard_prompt}",
        "",
        "## Classes",
        "",
        "| Label Index | Class |",
        "|---|---|",
    ]
    for keyword, label_index in keywords:
        lines.append(f"| {label_index} | {keyword} |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


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
    parser.add_argument("--db_path", type=str, default="./datasets/mapper_classification_datasets/DoD_3_5k.sqlite")
    parser.add_argument("--save_dir", type=str, default="./datasets/rl_prompt_datasets")
    parser.add_argument("--num_datasets", type=int, default=10)
    parser.add_argument("--num_train", type=int, default=32)
    parser.add_argument("--num_dev", type=int, default=32)
    parser.add_argument("--seed", type=int, default=47)
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    DB_PATH = args.db_path
    SAVE_DIR = args.save_dir
    NUM_DATASETS = args.num_datasets
    NUM_TRAIN = args.num_train
    NUM_DEV = args.num_dev
    SEED = args.seed
    random.seed(SEED)

    # Connect to the DoD SQLite DB
    print(f"Connecting to database: {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    HAS_CATEGORY = datasets_table_has_category(cursor)

    # Fetch and shuffle all dataset ids to determine sampling order
    dataset_ids = fetch_dataset_ids(cursor)
    random.shuffle(dataset_ids)
    print(f"Found {len(dataset_ids)} mini-datasets in the DB. Sampling {NUM_DATASETS} of them...")

    os.makedirs(SAVE_DIR, exist_ok=True)

    success_count = 0
    skipped_count = 0

    dataset_pbar = tqdm(dataset_ids, desc="Sampling Datasets")

    for dataset_id in dataset_pbar:
        if success_count >= NUM_DATASETS:
            break

        dataset_pbar.set_postfix({"Sampled": success_count, "Skipped": skipped_count})

        out_dir = os.path.join(SAVE_DIR, f"dataset_{dataset_id}")

        # Resumability: skip datasets already fully exported
        if os.path.exists(os.path.join(out_dir, "classes.md")):
            success_count += 1
            continue

        category, hard_prompt = fetch_dataset_metadata(cursor, dataset_id, HAS_CATEGORY)
        keywords = fetch_keywords(cursor, dataset_id)

        train_split_rows = fetch_split_rows(cursor, dataset_id, "train")
        test_rows = fetch_split_rows(cursor, dataset_id, "test")

        # Skip incomplete mini-datasets that don't have enough data
        if len(train_split_rows) < NUM_TRAIN + NUM_DEV or len(test_rows) == 0:
            skipped_count += 1
            tqdm.write(
                f"Skipping dataset_id={dataset_id}: "
                f"{len(train_split_rows)} train-split rows, {len(test_rows)} test-split rows"
            )
            continue

        # Sample train/dev uniformly at random from the DB train-split pool,
        # using a per-dataset RNG so the outer dataset-selection shuffle is untouched
        rng = random.Random(SEED + dataset_id)
        rng.shuffle(train_split_rows)
        train_rows = train_split_rows[:NUM_TRAIN]
        dev_rows = train_split_rows[NUM_TRAIN:NUM_TRAIN + NUM_DEV]

        os.makedirs(out_dir, exist_ok=True)

        write_tsv(train_rows, os.path.join(out_dir, "train.tsv"))
        write_tsv(dev_rows, os.path.join(out_dir, "dev.tsv"))
        write_tsv(test_rows, os.path.join(out_dir, "test.tsv"))
        write_classes_md(os.path.join(out_dir, "classes.md"), dataset_id, category, hard_prompt, keywords)

        success_count += 1

    conn.close()

    print(f"\nDone! Exported {success_count} mini-datasets to {SAVE_DIR}.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} incomplete mini-datasets.")
    if success_count < NUM_DATASETS:
        print(f"Warning: Requested {NUM_DATASETS} datasets but only {success_count} were available/valid.")
