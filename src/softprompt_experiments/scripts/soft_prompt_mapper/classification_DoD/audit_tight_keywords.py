import argparse
import pickle
import os
import sqlite3
import re
import random

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
    parser.add_argument("--keyword_pickle_path", type=str, default="./datasets/mapper_classification_datasets/keywords_DoD3_10k_2.pkl")
    args, _ = parser.parse_known_args(args_list)

    KEYWORD_PICKLE_PATH = args.keyword_pickle_path
    
    # Regexes
    VALID_ENGLISH_CATEGORY_REGEX = r"^[a-zA-Z0-9\s_\-]+$"
    STRICT_SINGLE_WORD_REGEX = r"^[A-Za-z]+$"

    # InSPEcT Forbidden Evaluation Targets
    FORBIDDEN_DATASETS = [
        {"positive", "negative"},                                                # SST-2
        {"terrible", "bad", "neutral", "good", "great"},                         # SST-5
        {"world", "sports", "business", "technology"},                           # AGNews
        {"objective", "subjective"},                                             # Subj
        {"abbreviation", "entity", "description", "human", "location", "number"} # TREC
    ]

    # Load Existing Keywords
    semantic_keyword_sets = set()
    print(f"Loading keywords from {KEYWORD_PICKLE_PATH} ...")
    try:
        with open(KEYWORD_PICKLE_PATH, 'rb') as f:
            semantic_keyword_sets = pickle.load(f)
        print(f"Successfully loaded {len(semantic_keyword_sets)} total keyword sets.")
    except Exception as e:
        print(f"Error loading existing pickle file: {e}. Exiting ...")
        exit(-1)

    # ---------------------------------------------------------
    # TRACKERS
    # ---------------------------------------------------------
    categories_set = set()
    
    # Category-level issues
    non_english_category_count = 0
    repetitive_category_count = 0
    
    # Class-level issues
    incorrect_class_count = 0
    multi_word_class_count = 0
    forbidden_vocab_hit_count = 0
    target_leakage_count = 0

    # ---------------------------------------------------------
    # AUDIT LOOP
    # ---------------------------------------------------------
    for category, classes in list(semantic_keyword_sets):
        clean_category = category.strip()
        lower_classes = [c.lower() for c in classes]
        lower_classes_set = set(lower_classes)

        # 1. Repetitive Categories
        if clean_category.lower() in categories_set:
            repetitive_category_count += 1
        categories_set.add(clean_category.lower())

        # 2. Non-English / Malformed Categories
        if not re.search(VALID_ENGLISH_CATEGORY_REGEX, clean_category):
            non_english_category_count += 1
            # print(f"Flagged as non-English Category: '{clean_category}'")

        # 3. Exactly 5 Classes (Catches both < 5 squashed duplicates and > 5 hallucinations)
        if len(classes) != 5:
            incorrect_class_count += 1

        # 4. Strict Single-Word Classes Check
        for c in classes:
            if not re.search(STRICT_SINGLE_WORD_REGEX, c):
                multi_word_class_count += 1
                # print(f"Flagged Multi-Word Class: '{c}' in Category '{clean_category}'")
                break # Only count the dataset once

        # 5. Forbidden Baseline Vocab Hit (Overlap >= 2)
        for baseline_set in FORBIDDEN_DATASETS:
            if len(lower_classes_set.intersection(baseline_set)) >= 2:
                forbidden_vocab_hit_count += 1
                print(f"Forbidden Vocab Intersection: {clean_category} -> {lower_classes}")
                break

        # 6. Target Leakage (Category name exists as one of the classes)
        if clean_category.lower() in lower_classes:
            target_leakage_count += 1

    # ---------------------------------------------------------
    # REPORTING
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("AUDIT RESULTS:")
    print("="*50)
    print(f"Total Sets Checked:             {len(semantic_keyword_sets)}")
    print(f"Repetitive Categories:          {repetitive_category_count}")
    print(f"Non-English Categories:         {non_english_category_count}")
    print(f"Not Exactly 5 Classes:          {incorrect_class_count}")
    print(f"Multi-word Classes:             {multi_word_class_count}")
    print(f"Forbidden Vocab Overlaps:       {forbidden_vocab_hit_count}")
    print(f"Target Leakage:                 {target_leakage_count}")
    print("="*50 + "\n")

    # Optional: Calculate how many purely valid datasets you actually have
    total_flags = (repetitive_category_count + non_english_category_count + 
                   incorrect_class_count + multi_word_class_count + target_leakage_count)
    
    print(f"Estimated Clean Datasets available for DoD: ~{len(semantic_keyword_sets) - total_flags}")

    # Convert to list and predictably shuffle for ML reproducibility
    keyword_list = sorted(list(semantic_keyword_sets))
    random.seed(42) # Guarantees the exact same random sequence every run
    random.shuffle(keyword_list)


    print("Printing valid keyword sets:\n")
    valid_keyword_sets = []
    for category, classes in keyword_list:
        clean_category = category.strip().lower()
        lower_classes = [c.lower() for c in classes]
        lower_classes_set = set(lower_classes)

        # 1. Check for exactly 5 classes
        if len(classes) != 5:
            continue
            
        # 2. Check for Target Leakage
        if clean_category in lower_classes:
            continue

        # 3. Check for valid english categories
        if not re.search(VALID_ENGLISH_CATEGORY_REGEX, clean_category):
            continue
            
        # 4. Check for Baseline Data Leakage (>= 2 overlaps)
        is_leak = False
        for baseline_set in FORBIDDEN_DATASETS:
            if len(lower_classes_set.intersection(baseline_set)) >= 2:
                is_leak = True
                break
                
        if is_leak:
            continue
            
        # If it passes all checks, add it to our final generation queue!
        valid_keyword_sets.append((clean_category, classes))

    
    for category, classes in valid_keyword_sets[:5]:
        print(f"{category}: {classes}")

    

    