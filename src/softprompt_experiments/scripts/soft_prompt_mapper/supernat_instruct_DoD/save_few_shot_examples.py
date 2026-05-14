import os
import argparse
from datasets import load_dataset
import pickle


# Driver Code
def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="Suryanshg/SUPER-NATURALINSTRUCTIONS-english-filtered-100x-augmented")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    DATASET_PATH = args.dataset_path

    # Fetch all hard prompts from Hugging Face Dataset
    hf_dataset = load_dataset(DATASET_PATH).select_columns(['task_name', 'reduced_instructions'])
    
    # Convert to Pandas
    train_dataset_df = hf_dataset['train'].to_pandas()

    # Keep only the first reduced_instruction in the reduced_instructions list
    train_dataset_df['reduced_instructions'] = train_dataset_df.apply(
        lambda row: list(row['reduced_instructions'])[0],
        axis=1 # Apply row by row
    )

    # Randomly sample 10 rows from the dataframe
    sampled_df = train_dataset_df.sample(n=10, random_state=47).copy()

    paraphrased_instructions_pool = sampled_df['reduced_instructions'].tolist()
    forbidden_task_names = sampled_df['task_name'].tolist()

    print(f"Found {len(paraphrased_instructions_pool)} instructions for the few-shot pool")
    print(f"Found {len(forbidden_task_names)} tasks for InSPEcT on Train split on SuperNat DoD")

    with open('paraphrased_instructions_few_shot_pool.pkl', 'wb') as f:
        pickle.dump(paraphrased_instructions_pool, f)

    with open('forbidden_task_names_for_inspect.pkl', 'wb') as f:
        pickle.dump(forbidden_task_names, f)


    



