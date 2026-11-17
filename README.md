# Learning to Translate from Soft to Hard LLM Prompts

## Abstract

Soft prompt tuning is a parameter-efficient method for adapting LLMs to specific tasks, but suffers from a lack of interpretability. Building on recent work on interpreting soft prompts (Ramati et al. 2024), we explore how training a dedicated soft prompt to natural language translation model can yield higher translation quality. In particular, in both quantitative and qualitative comparisons on multiple Datasets of Datasets (DoDs), we demonstrate that our translator produces fluent, accurate verbalizations that outperforms existing training-free methods like InSPEcT. In addition to advancing interpretability, our work suggests a promising downstream application: soft prompts optimized on small, open-source models can be translated into portable text prompts that, when deployed on larger closed-API models, exceed the performance of the original soft prompt and, in some cases, even few-shot learning.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
```

- Base model: `meta-llama/Llama-3.1-8B-Instruct` (requires Hugging Face access).
- Scripts that call OpenAI (e.g. the gpt-4o-mini portability experiment) need `OPENAI_API_KEY` in your environment or in a `.env` file in the project root.

## Repository Structure

```bash
├── run_experiment.py                     # Driver script: runs any script under src/softprompt_experiments/scripts
├── src/softprompt_experiments
│   ├── models/softprompt.py              # Custom soft prompt implementation
│   ├── InSPEcT_utils.py                  # InSPEcT baseline utilities
│   └── scripts
│       ├── soft_prompt_mapper
│       │   ├── supernat_instruct_DoD/    # General DoD (Super-NaturalInstructions) experiments
│       │   └── classification_DoD/       # Classification DoD + InSPEcT benchmark experiments
│       ├── comparisons/                  # Task relevance: verbalizations used as prompts (§4.2)
│       └── dpo/                          # Translator++ and DPO post-training (Appendix A.10)
│
│   # Created when running experiments:
├── datasets/                             # Generated DoDs and compiled translator datasets
├── trained_soft_prompts/                 # Trained soft prompts
├── mapper_lora_weights/                  # Trained translator LoRA adapters
├── inspect_results/                      # InSPEcT outputs
└── verbalizations/                       # Translator verbalizations and evaluation results
```

> **Terminology:** the code uses *mapper* and *translator* interchangeably.

## Running Experiments

Every script is run through the driver, using its module path under `softprompt_experiments.scripts`:

```bash
python -m run_experiment --scripts <module.path> [--script-args ...]
```

Several scripts can be chained in one call, and any extra arguments are forwarded to every listed script. The default paths are already chained, so each stage's output directory is the next stage's input. **If you override an output path, pass the same path to the next stage's input argument.**

### General DoD (Super-NaturalInstructions)

```bash
python -m run_experiment --scripts \
  soft_prompt_mapper.supernat_instruct_DoD.train_softprompts \
  soft_prompt_mapper.supernat_instruct_DoD.compile_mapper_dataset \
  soft_prompt_mapper.supernat_instruct_DoD.train_mapper \
  soft_prompt_mapper.supernat_instruct_DoD.test_mapper
```

This pipeline:

1. Trains a 20-token soft prompt for every task.
2. Pairs each soft prompt with its instruction and paraphrases (General-DoD-10x).
3. Trains the translator.
4. Generates verbalizations for the held-out tasks and scores them (ROUGE-L, cosine similarity).

To rebuild the dataset from scratch, use `generate_dataset` → `filter_dataset` → `generate_paraphrasals`. The first step needs a local clone of [allenai/natural-instructions](https://github.com/allenai/natural-instructions).

### Classification DoD

```bash
python -m run_experiment --scripts \
  soft_prompt_mapper.classification_DoD.generate_keywords_for_tight_DoD \
  soft_prompt_mapper.classification_DoD.classification_DoD_generator \
  soft_prompt_mapper.classification_DoD.train_DoD_softprompts \
  soft_prompt_mapper.classification_DoD.compile_mapper_dataset \
  soft_prompt_mapper.classification_DoD.train_mapper \
  soft_prompt_mapper.classification_DoD.inference_mapper_dataset
```

### InSPEcT Baseline

| Setting | Scripts (in order) |
|---|---|
| General DoD | `supernat_instruct_DoD.train_softprompts_peft` → `compile_dataset_for_InSPEcT` → `apply_InSPEcT_on_DoD` → `test_InSPEcT_on_DoD` |
| Classification DoD | `classification_DoD.train_DoD_softprompts_peft` → `compile_mapper_dataset --trained_soft_prompts_dir ./trained_soft_prompts/DoD_3_5k_peft` → `apply_InSPEcT_on_DoD` → `test_InSPEcT_on_DoD` |
| InSPEcT benchmark datasets | `classification_DoD.train_InSPEcT_softprompts_peft` → `apply_InSPEcT` |

To apply our translators to the InSPEcT benchmark soft prompts:
1. Train the soft prompts with `classification_DoD.train_InSPEcT_softprompts`.
2. Run one of the translators on them:
   - `classification_DoD.inference_InSPEcT_softprompts` for the Classification DoD translator.
   - `supernat_instruct_DoD.test_mapper_on_InSPEcT_softprompts` for the General DoD translator (out-of-distribution evaluation).

### Verbalizations as Prompts (Task Relevance)

`comparisons.vllm_pipeline` runs on Llama-3.1-8B-Instruct and `comparisons.pipeline` runs on gpt-4o-mini. Both run the verbalizations as task prompts and score the LLM's outputs with Task ROUGE-L.

## Acknowledgements

Our InSPEcT implementation builds on code from [danaramati1/InSPEcT](https://github.com/danaramati1/InSPEcT).
