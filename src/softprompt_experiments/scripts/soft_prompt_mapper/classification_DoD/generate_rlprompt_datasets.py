import os
import argparse
import random
import sqlite3
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


def fetch_dataset_ids(cursor):
    cursor.execute("SELECT dataset_id FROM datasets")
    return [row[0] for row in cursor.fetchall()]


def fetch_dataset_metadata(cursor, dataset_id):
    cursor.execute("SELECT category, hard_prompt FROM datasets WHERE dataset_id = ?", (dataset_id,))
    return cursor.fetchone()


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


def stratified_split(train_split_rows, num_classes, num_train, num_dev, rng):
    """Split rows into train/dev with per-class counts as equal as possible.

    Returns None if any class doesn't have enough rows for its quota, so
    callers can skip the dataset instead of producing a class-imbalanced
    (or class-missing) train/dev split.
    """
    rows_by_label = defaultdict(list)
    for sentence, label in train_split_rows:
        rows_by_label[label].append((sentence, label))

    if len(rows_by_label) < num_classes:
        return None

    for label in rows_by_label:
        rng.shuffle(rows_by_label[label])

    base_train, extra_train = divmod(num_train, num_classes)
    base_dev, extra_dev = divmod(num_dev, num_classes)

    train_rows, dev_rows = [], []
    for i, label in enumerate(sorted(rows_by_label.keys())):
        n_train = base_train + (1 if i < extra_train else 0)
        n_dev = base_dev + (1 if i < extra_dev else 0)

        class_rows = rows_by_label[label]
        if len(class_rows) < n_train + n_dev:
            return None

        train_rows.extend(class_rows[:n_train])
        dev_rows.extend(class_rows[n_train:n_train + n_dev])

    return train_rows, dev_rows


def write_tsv(rows, path):
    df = pd.DataFrame(rows, columns=["sentence", "label"])
    df.to_csv(path, sep="\t", index=False)


def write_classes_md(path, dataset_id, category, hard_prompt, keywords):
    lines = [
        f"# Dataset {dataset_id}",
        "",
        f"**Category:** {category}",
        "",
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

        category, hard_prompt = fetch_dataset_metadata(cursor, dataset_id)
        keywords = fetch_keywords(cursor, dataset_id)

        train_split_rows = fetch_split_rows(cursor, dataset_id, "train")
        test_rows = fetch_split_rows(cursor, dataset_id, "test")

        # Skip incomplete mini-datasets that don't have enough data
        if len(test_rows) == 0:
            skipped_count += 1
            tqdm.write(f"Skipping dataset_id={dataset_id}: no test-split rows")
            continue

        # Sample train/dev with per-class counts as equal as possible, so every
        # class is guaranteed representation in both splits (avoids a class
        # ending up with zero train/dev examples under a plain uniform sample),
        # using a per-dataset RNG so the outer dataset-selection shuffle is untouched
        rng = random.Random(SEED + dataset_id)
        split_result = stratified_split(train_split_rows, len(keywords), NUM_TRAIN, NUM_DEV, rng)
        if split_result is None:
            skipped_count += 1
            tqdm.write(
                f"Skipping dataset_id={dataset_id}: insufficient per-class train-split rows "
                f"for {len(keywords)} classes ({len(train_split_rows)} total train-split rows)"
            )
            continue
        train_rows, dev_rows = split_result

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
