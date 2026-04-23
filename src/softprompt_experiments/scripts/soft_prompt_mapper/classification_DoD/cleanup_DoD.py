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
    parser.add_argument("--db_path", type=str, default="./datasets/mapper_classification_datasets/DoD_2_5k_Mistral.sqlite")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    DB_PATH = args.db_path

    # Perform DB Cleanup
    clean_database(DB_PATH)



def clean_database(db_path):
    print(f"Connecting to database: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print(f"Successfully Connected!")
    
    # Identify all datasets that don't have exactly 500 sentences
    print("Scanning for incomplete datasets...")
    cursor.execute("""
        SELECT dataset_id 
        FROM sentences 
        GROUP BY dataset_id 
        HAVING COUNT(sentence_id) < 500
    """)

    incomplete_dataset_ids = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(incomplete_dataset_ids)} incomplete datasets out of the original 5500.")

    # # Delete the data from the bottom up to respect foreign keys
    # if incomplete_dataset_ids:
    #     print("Scrubbing incomplete data from tables...")
        
    #     for dataset_id in incomplete_dataset_ids:
    #         print(f"Removing data related to dataset_id: {dataset_id}")
    #         cursor.execute("DELETE FROM sentences WHERE dataset_id = ?", (dataset_id,))
    #         cursor.execute("DELETE FROM keywords WHERE dataset_id = ?", (dataset_id,))
    #         cursor.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
            
    #     conn.commit()
    #     print("Successfully purged imperfect datasets.")
    
    # # Verify the final count
    # cursor.execute("SELECT COUNT(DISTINCT dataset_id) FROM datasets")
    # remaining_datasets = cursor.fetchone()[0]
    
    # cursor.execute("SELECT COUNT(*) FROM sentences")
    # remaining_sentences = cursor.fetchone()[0]
    
    # print(f"\nFinal Verified Stats:")
    # print(f"Perfect Datasets: {remaining_datasets}")
    # print(f"Total Sentences:  {remaining_sentences}")
    
    # # VACUUM to reclaim hard drive space
    # # Deleting rows in SQLite doesn't shrink the file size automatically.
    # # Vacuum rebuilds the B-Tree index and shrinks the file footprint.
    # print("\nVacuuming database to optimize size and indexing speed (this may take a minute)...")
    # cursor.execute("VACUUM")
    # conn.commit()
    
    conn.close()
    print("Cleanup complete!")