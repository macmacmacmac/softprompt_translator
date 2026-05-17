import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.softprompt_experiments.InSPEcT_utils import elicit_description_using_inspect_technique, ALL_LAYER_COMBINATIONS, BEST_PATCHES
import pandas as pd
import nltk
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
import string

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))
ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)


# InSPEcT Paper benchmarks and classes
INSPECT_BENCHMARKS = {
    "sst2": {
        "classes": ["positive", "negative"],
        "references": [
            "Is the sentiment of this sentence positive or negative?",
            "Is the sentiment of this text positive or negative?",
            "Would you classify this sentence as having a positive or negative sentiment?",
            "Do you think this sentence has a positive or negative tone?",
            "Would you consider the sentiment of this sentence to be positive or negative?",
            "How would you rate the sentiment of this sentence: positive or negative?",
            "How would the sentiment of this sentence be described? Positive, Negative.",
            "Can you identify whether the sentiment of this sentence is positive or negative?",
            "What is the tone of this sentence: positive or negative?"
        ]
    },
    "sst5": {
        "classes": ["terrible", "bad", "neutral", "good", "great"],
        "references": [
            "Is the sentiment of this sentence terrible, bad, neutral, good or great?",
            "Is the sentiment of this text terrible, bad, neutral, good or great?",
            "Would you classify this sentence as having a terrible, bad, neutral, good or great sentiment?",
            "Do you think this sentence has a terrible, bad, neutral, good or great tone?",
            "Would you consider the sentiment of this sentence to be terrible, bad, neutral, good or great?",
            "How would you rate the sentiment of this sentence: terrible, bad, neutral, good or great?",
            "How would the sentiment of this sentence be described? terrible, bad, neutral, good, great.",
            "Can you identify whether the sentiment of this sentence is terrible, bad, neutral, good or great?",
            "What is the tone of this sentence: terrible, bad, neutral, good or great?"
        ]
    },
    "ag_news": {
        "classes": ["world", "sports", "business", "technology"],
        "references": [
            "What is this text about? World, Sports, Business, Technology",
            "Which topic is this article about? World, Sports, Business, Technology",
            "What is the main topic discussed in this news story: World, Sports, Business, Technology",
            "What is the main topic discussed in this article: World, Sports, Business, Technology",
            "Which topic best captures the essence of this article? World, Sports, Business, Technology",
            "What is the most fitting summary for this article? World, Sports, Business, Technology",
            "Under which category does this article best fall? World, Sports, Business, Technology.",
            "Among World, Sports, Business, and Technology, which best captures the topic of this article?",
            "Classify this news report into the appropriate category: World, Sports, Business, Technology",
            "Which category best fits the topic of this article? World, Sports, Business, Technology",
            "To which category does this news article's topic belong: World, Sports, Business, Technology"
        ]
    },
    "subj": {
        "classes": ["objective", "subjective"],
        "references": [
            "Is the subjectivity of this text objective or subjective?",
            "How would the subjectivity of this sentence be described? Objective, Subjective.",
            "Is this sentence objective or subjective in nature?",
            "In terms of subjectivity, is this sentence objective or subjective?",
            "How would you describe the subjectivity of this sentence: objective or subjective?",
            "Is the nature of this text's subjectivity objective or subjective?",
            "Classify the sentence based on its expression: objective, subjective",
            "Is this sentence factual or opinionated: objective, subjective",
            "Is this sentence based on facts or personal feelings: objective, subjective",
            "Determine if this sentence presents facts or opinions: objective, subjective"
        ]
    },
    "trec-qc": {
        "classes": ["description", "entity", "abbreviation", "human", "location", "number"],
        "references": [
            "Is the question asking about an entity, a description, an abbreviation, an expression, a human, a location, or a number?",
            "Which one of the following options would the answer to this be?\nDescription, Entity, Abbreviation, Expression, Human, Location, Number",
            "What type of thing is the question asking about?\nDescription, Entity, Abbreviation, Expression, Human, Location, Number",
            "classify the answer of this question. is it an entity, a description, an abbreviation, an expression, a human, a location, or a number?",
            "What type is the answer to this question: entity, description, abbreviation, expression, human, location, or number?",
            "How would you classify the answer from the following options?\nDescription, Entity, Abbreviation, Expression, Human, Location, Number",
            "Choose the category that best fits the answer:\nDescription, Entity, Abbreviation, Expression, Human, Location, Number",
            "Is the question seeking information about an entity, a description, an abbreviation, an expression, a human, a location, or a number?",
            "Does the question pertain to an entity, a description, an abbreviation, an expression, a human, a location, or a number?"
        ]
    }
}


def calculate_eval_metrics(elicited_text, bench_data):
    """Calculates Class Rate, ROUGE-1, and F1 Score as defined by the InSPEcT methodology."""
    # Calculate Class Rate (Recall)
    clean_text = elicited_text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = set(clean_text.split())
    
    classes_count = sum(1 for c in bench_data["classes"] if c.lower() in words)
    class_rate = classes_count / len(bench_data["classes"]) if bench_data["classes"] else 0.0
    
    # Calculate Precision and F1 Score
    precision = classes_count / len(words) if words else 0.0
    f1_score = 2 * (precision * class_rate) / (precision + class_rate) if (precision + class_rate) > 0 else 0.0
    
    # Calculate ROUGE1
    def remove_stopwords(t):
        t_clean = t.translate(str.maketrans('', '', string.punctuation)).lower()
        return " ".join([w for w in t_clean.split() if w not in STOP_WORDS])
    
    clean_pred = remove_stopwords(elicited_text)
    max_rouge1 = 0.0
    
    for ref in bench_data["references"]:
        clean_ref = remove_stopwords(ref)
        score = ROUGE_SCORER.score(clean_ref, clean_pred)['rouge1'].fmeasure
        if score > max_rouge1:
            max_rouge1 = score
            
    return class_rate, max_rouge1, f1_score


def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect_soft_prompts_dir", type=str, default="./inspect_soft_prompts_peft_random_16bit")
    parser.add_argument("--results_save_dir", type=str, default="./inspect_results/inspect_soft_prompts_peft_random_16bit")
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--peft", action="store_true", help="Use PEFT style way of loading soft prompts")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    INSPECT_SOFT_PROMPTS_DIR = args.inspect_soft_prompts_dir
    NUM_TOKENS = args.num_tokens
    LOAD_LIKE_PEFT = args.peft
    RESULTS_SAVE_DIR = args.results_save_dir

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Loading Training accuracy stats
    TRAINING_STATS_DF = pd.read_csv(os.path.join(INSPECT_SOFT_PROMPTS_DIR, "accuracy_stats.csv"))

    # ┌───────────────────────────────────────────────┐
    # │              INSPECT MODEL PREP               │
    # └───────────────────────────────────────────────┘
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {MODEL_NAME} for InSPEcT...")
    inspect_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    inspect_model.eval()



    # ┌───────────────────────────────────────────────┐
    # │         PERFORM INSPECT ON SOFT PROMPTS       │
    # └───────────────────────────────────────────────┘

    # List to hold the summary of best metrics across all inspect datasets
    summary_results = []

    for root, dirs, files in os.walk(INSPECT_SOFT_PROMPTS_DIR):
        if len(dirs) > 0:
            for soft_prompt_dir in dirs:
                # Extract the Dataset Name
                dataset_name = '_'.join(soft_prompt_dir.split('_')[:-1])

                # Construct soft prompt path
                soft_prompt_path = os.path.join(root, soft_prompt_dir, 'softprompt.pt')

                if LOAD_LIKE_PEFT:
                    soft_prompt = torch.load(soft_prompt_path, map_location = "cpu", weights_only = True)

                else:
                    # Load the saved state dict
                    # weights_only=True is a PyTorch security best practice for loading tensors
                    state_dict = torch.load(soft_prompt_path, map_location = "cpu", weights_only = True)
                    
                    # Extract the prompt embeddings
                    soft_prompt = state_dict['prompt_embeddings']       # (1, soft_prompt_len, embed_dim)
                    soft_prompt = soft_prompt.squeeze(0)                # (soft_prompt_len, embed_dim)


                print("-" * 100)
                print(f"Performing InSPEcT using soft prompts trained on {dataset_name}")
                print("-" * 100 + "\n")

                # Get Elicited Text using InSPEcT Technique
                inspect_elicited_results = elicit_description_using_inspect_technique(
                    model=inspect_model,
                    tokenizer=tokenizer,
                    num_tokens=NUM_TOKENS,
                    soft_prompt=soft_prompt,
                    dataset_name="REPLACE_ME",
                    # layer_combinations=BEST_PATCHES,
                    layer_combinations=ALL_LAYER_COMBINATIONS,
                    target_prompt_type='few_shot'
                )

                # Evaluate InSPEcT Model
                bench_data = INSPECT_BENCHMARKS[dataset_name.lower()]
                for i in range(len(inspect_elicited_results)):
                    output_text = str(inspect_elicited_results[i]['output'])
                    class_rate, rouge1, f1_score = calculate_eval_metrics(output_text, bench_data)
                    inspect_elicited_results[i]['class_rate'] = class_rate
                    inspect_elicited_results[i]['rouge1_score'] = rouge1
                    inspect_elicited_results[i]['f1_score'] = f1_score

                # Find the row with the highest ROUGE-1 score
                max_rouge1_row = max(inspect_elicited_results, key=lambda x: x['rouge1_score'])

                # Find the row with the highest Class Rate
                max_classrate_row = max(inspect_elicited_results, key=lambda x: x['class_rate'])

                # Find the row with the highest F1 Score
                max_f1_row = max(inspect_elicited_results, key=lambda x: x['f1_score'])

                # Retrieve the training stats for this dataset
                training_stats_df = TRAINING_STATS_DF[TRAINING_STATS_DF["dataset_id"].str.contains(dataset_name)]

                # Save Elicitations using InSPEcT for this dataset
                os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

                df = pd.DataFrame(inspect_elicited_results)
                df.to_csv(f'{RESULTS_SAVE_DIR}/{dataset_name}_elicitations.csv', index=False)

                result_entry = {
                    "dataset": dataset_name,
                    "val_accuracy": training_stats_df['val_accuracy'].iloc[0] if len(training_stats_df) > 0 else None,
                    "max_class_rate": round(max_classrate_row['class_rate'], 4),
                    "max_rouge1": round(max_rouge1_row['rouge1_score'], 4),
                    "max_f1_score": round(max_f1_row['f1_score'], 4),
                    "max_class_rate_src_layer": max_classrate_row['source_layer'],
                    "max_class_rate_tgt_layer": max_classrate_row['target_layer'],
                    "max_rouge1_src_layer": max_rouge1_row['source_layer'],
                    "max_rouge1_tgt_layer": max_rouge1_row['target_layer'],
                    "max_f1_src_layer": max_f1_row['source_layer'],
                    "max_f1_tgt_layer": max_f1_row['target_layer'],
                }

                summary_results.append(result_entry)

        else:
            break

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = f"{RESULTS_SAVE_DIR}/inspect_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSaved master summary with best metrics to: {summary_csv_path}")
