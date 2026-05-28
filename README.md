# Soft Prompt Interpretability Experiments
## Abstract
Soft prompt tuning is a parameter-efficient method for adapting LLMs to specific tasks, but suffers from a lack of interpretability. Building on recent work on interpreting soft prompts (Ramati et al. 2024), we explore how training a dedicated soft prompt to natural language translation model can yield higher translation quality. In particular, in both quantitative and qualitative comparisons on multiple Datasets of Datasets (DoDs), we demonstrate that our translator produces fluent, accurate verbalizations that outperforms existing training-free methods like InSPEcT. In addition to advancing interpretability, our work suggests a promising downstream application: soft prompts optimized on small, open-source models can be translated into portable text prompts that, when deployed on larger closed-API models, exceed the performance of the original soft prompt and, in some cases, even few-shot learning.

## Command for setting up Locally
Setup Virtual Environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
```

## Project Directory Structure Overview
This overview is mostly for relevant scripts for the Paper: `Learning to Translate from Soft to Hard LLM Prompts`:
```bash
├── datasets <-- Stores Datasets
├── inspect_results  <-- Stores InSPEcT results   
├── mapper_lora_weights <-- Stores trained LoRA Adapters for different Translators
├── pyproject.toml
├── README.md
├── requirements.txt
├── run_experiment.py <-- Driver Script
src/
└── softprompt_experiments
    ├── __init__.py
    ├── InSPEcT_utils.py <-- Util Code for InSPEcT Experiments
    ├── models
    │   ├── __init__.py
    |   | ...
    │   ├── softprompt.py <-- CUSTOM SOFT PROMPT IMPLEMENTATION
    │   | ...
    ├── scripts
    │   ├── soft_prompt_mapper
    │   │   ├── classification_DoD <-- Scripts for Classification DoD Experiments
    │   │   ├── __init__.py
    │   │   └── supernat_instruct_DoD <-- Script for General DoD Experiments
    └── ...
├── trained_soft_prompts <-- Contains Trained Soft prompts
...
```


## Command for reproducing experiments
Example Usage 
```bash
python -m run_experiment --scripts \
  soft_prompt_mapper.supernat_instruct_DoD.train_softprompts \
  soft_prompt_mapper.supernat_instruct_DoD.compile_mapper_dataset \
  soft_prompt_mapper.supernat_instruct_DoD.train_mapper \
  soft_prompt_mapper.supernat_instruct_DoD.test_mapper
```
This will:
1. Train softprompts
2. Compile trained softprompts into a dataset for translator
3. Train the translator
4. Test the translator

You can also call these scripts individually like
```bash
python -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.train_softprompts
```

### Important Note about input / output directory paths as command line args
When running the scripts one by one using the previous command, please ensure that the output directory in the script argument matches the input directory of the next script's argument. For example:
`soft_prompt_mapper.supernat_instruct_DoD.train_softprompts` might save soft prompts trained under directory: `./trained_soft_prompts/General-DoD` (using argument `--save_dir`). So the `--trained_soft_prompts_dir` argument of script `soft_prompt_mapper.supernat_instruct_DoD.compile_mapper_dataset` should be `./trained_soft_prompts/General-DoD`.






## Important Notes
- All our scripts use the word 'mapper' interchangeably with 'translator'
- InSPEcT Implementation is inspired by code from [this repository](https://github.com/danaramati1/InSPEcT).


