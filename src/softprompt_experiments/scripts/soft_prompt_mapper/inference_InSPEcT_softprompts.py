import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.softprompt_experiments.InSPEcT_utils import elicit_description_using_inspect_technique, BEST_PATCHES, ALL_LAYER_COMBINATIONS
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


def calculate_inspect_metrics(elicited_text, bench_data):
    """Calculates Class Rate and ROUGE-1 as defined by the InSPEcT methodology."""
    # Calculate Class Rate
    clean_text = elicited_text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = set(clean_text.split())
    
    classes_count = sum(1 for c in bench_data["classes"] if c.lower() in words)
    class_rate = classes_count / len(bench_data["classes"]) if bench_data["classes"] else 0.0
    
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
            
    return class_rate, max_rouge1


def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect_soft_prompts_dir", type=str, default="./inspect_soft_prompts")
    parser.add_argument("--lora_dir", type=str, default="./mapper_lora_weights/DoD_3_5k_1616")
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--inspect", action="store_true", help="Run InSPEcT technique for comparison")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    INSPECT_SOFT_PROMPTS_DIR = args.inspect_soft_prompts_dir
    LORA_DIR = args.lora_dir
    NUM_TOKENS = args.num_tokens
    DATASET_NAME = LORA_DIR.split('/')[-1]

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


    # ┌───────────────────────────────────────────────┐
    # │                 LORA MODEL PREP               │
    # └───────────────────────────────────────────────┘
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)

    inspect_model = None
    if args.inspect:
        print(f"Loading inspect model {MODEL_NAME}...")
        inspect_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
        inspect_model.eval()

    print(f"Loading LoRA adapters from {LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()


    # ┌───────────────────────────────────────────────┐
    # │ PERFORM INFERENCE USING INSPECT SOFT PROMPTS  │
    # └───────────────────────────────────────────────┘

    # List to hold the summary of best metrics across all inspect datasets
    summary_results = []

    for root, dirs, files in os.walk(INSPECT_SOFT_PROMPTS_DIR):
        if len(dirs) > 0:
            for soft_prompt_dir in dirs:
                # Extract the Dataset Name
                dataset_name = '_'.join(soft_prompt_dir.split('_')[:-1])

                # Match dataset to benchmarks
                benchmark_key = None
                for key in INSPECT_BENCHMARKS.keys():
                    if key in dataset_name.lower():
                        benchmark_key = key
                        break

                print("-" * 100)
                print(f"Performing Inference using soft prompts trained on {dataset_name}")
                print("-" * 100)

                # Construct soft prompt path
                soft_prompt_path = os.path.join(root, soft_prompt_dir, 'softprompt.pt')

                # # Load the saved state dict
                # # weights_only=True is a PyTorch security best practice for loading tensors
                state_dict = torch.load(soft_prompt_path, map_location = "cpu", weights_only = True)
                
                # Extract the prompt embeddings. 
                # The SoftPrompt class saves it as shape (1, soft_prompt_len, embed_dim).
                soft_prompt = state_dict['prompt_embeddings']

                # Add batch dimension to the soft prompt
                inputs_embeds = soft_prompt.to(DEVICE, dtype = DTYPE)               # (1, soft_prompt_len, embed_dim)

                # Create an attention mask of 1s for the soft_prompt_len tokens
                attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=DEVICE) # (1, soft_prompt_len)

                # Generate the discrete text
                # Using greedy decoding (temperature=0.0)
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                    do_sample=False, 
                    pad_token_id=tokenizer.eos_token_id
                )

                # Decode the generated token IDs back into an English string
                pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                # Print out the Stats
                print(f"Mapper Model Predictions: {pred_text}\n\n")

                # Evaluate Mapper Model
                if benchmark_key:
                    bench_data = INSPECT_BENCHMARKS[benchmark_key]
                    mapper_class_rate, mapper_rouge1 = calculate_inspect_metrics(pred_text, bench_data)
                    
                    # Print out the classes of this dataset
                    print(f"Classes for {dataset_name}: {bench_data['classes']}")


                    print(f"Mapper Class Rate: {mapper_class_rate:.2f}")
                    print(f"Mapper Max ROUGE-1: {mapper_rouge1:.2f}\n")
                else:
                    print("Could not map dataset name to InSPEcT benchmarks for Mapper evaluation.\n")

                # Initialize InSPEcT results
                inspect_best_classrate_row = None
                inspect_best_rouge1_row = None

                if args.inspect:
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
                        layer_combinations=ALL_LAYER_COMBINATIONS,
                        target_prompt_type='few_shot'
                    )

                    # Evaluate InSPEcT Model
                    if benchmark_key:
                        bench_data = INSPECT_BENCHMARKS[benchmark_key]
                        for i in range(len(inspect_elicited_results)):
                            output_text = str(inspect_elicited_results[i]['output'])
                            class_rate, rouge1 = calculate_inspect_metrics(output_text, bench_data)

                            inspect_elicited_results[i]['class_rate'] = class_rate
                            inspect_elicited_results[i]['rouge1_score'] = rouge1

                    # Find the row with the highest ROUGE-1 score
                    inspect_best_rouge1_row = max(inspect_elicited_results, key=lambda x: x['rouge1_score'])

                    # Find the row with the highest Class Rate
                    inspect_best_classrate_row = max(inspect_elicited_results, key=lambda x: x['class_rate'])

                    # Save Elicitations using InSPEcT for this dataset
                    elicitation_save_dir = f"./inspect_results/{DATASET_NAME}/inspect_soft_prompts"
                    os.makedirs(elicitation_save_dir, exist_ok=True)

                    df = pd.DataFrame(inspect_elicited_results)
                    df.to_csv(f'{elicitation_save_dir}/{dataset_name}_elicitations.csv', index=False)

                # Save the summary comparing Mapper vs InSPEcT Best
                result_entry = {
                    "dataset": dataset_name,
                    "mapper_class_rate": round(mapper_class_rate, 4) if benchmark_key else None,
                    "mapper_rouge1": round(mapper_rouge1, 4) if benchmark_key else None,
                    "mapper_elicitation": pred_text,
                }

                if args.inspect and inspect_best_classrate_row and inspect_best_rouge1_row:
                    result_entry.update({
                        "inspect_best_class_rate": round(inspect_best_classrate_row['class_rate'], 4),
                        "inspect_best_class_rate_src_layer": inspect_best_classrate_row['source_layer'],
                        "inspect_best_class_rate_tgt_layer": inspect_best_classrate_row['target_layer'],
                        "inspect_best_class_rate_elicitation": inspect_best_classrate_row['output'],
                        "inspect_best_rouge1": round(inspect_best_rouge1_row['rouge1_score'], 4),
                        "inspect_best_rouge1_src_layer": inspect_best_rouge1_row['source_layer'],
                        "inspect_best_rouge1_tgt_layer": inspect_best_rouge1_row['target_layer'],
                        "inspect_best_rouge1_elicitation": inspect_best_rouge1_row['output']
                    })

                summary_results.append(result_entry)

        else:
            break

    if summary_results:
        summary_save_dir = f"./inspect_results/{DATASET_NAME}/inspect_soft_prompts"
        os.makedirs(summary_save_dir, exist_ok=True)
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = f"{summary_save_dir}/mapper_vs_inspect.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSaved master summary with best metrics to: {summary_csv_path}")
