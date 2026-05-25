# Soft Prompt Interpretability Experiments
## Abstract

Soft prompting or prompt tuning refers to a parameter efficient tuning technique where the base LLM is frozen and a set of trainable embeddings are prepended to the input sequence. In this work, we demonstrate two findings. First (1): LLMs natively have a basic internal comprehension of soft prompts and are capable of verbalizing explanations of certain soft prompts in natural language, revealing activated concepts. Second (2): these verbalized explanations reveal how soft prompts can implicitly extract and utilize an LLM’s latent scientific knowledge. This indicates that soft prompting can act as a loosely "science informed" learning algorithm, capable of drawing upon the vast quantity of scientific literature ingested by LLMs during its pre-training to inform its predictions. 

## Command for setting up Locally
Setup Virtual Environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
```

## Command for reproducing experiments

Example Usage
```bash
python -m run_experiment --scripts dataset_math_custom softprompt_generator_regression softprompt_verbalizer --save_directory ./datasets/<folder name>
```

This will build the dataset, train the soft prompts, and generate verbalizations. In that order.

## Scripts for Classification DoD


## Scripts for General DoD
