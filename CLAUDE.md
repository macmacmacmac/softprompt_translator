# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) or Antigravity CLI when working with code in this repository.

## Workflow

[CLAUDE CODE only] When implementing plan changes (or any adhoc code changes), always delegate the implementation to a Sonnet 5 Agent.

Scripts that require GPU access should not be runtime-verified by Claude / AGY CLI — the user tests them on the WPI SLURM cluster. A `python -m py_compile` syntax check is sufficient verification.

## Scope

The actively maintained code lives in two places — focus there and ignore the rest unless asked:
- `src/softprompt_experiments/scripts/soft_prompt_mapper/supernat_instruct_DoD/` — experiment scripts for the General DoD (Dataset-of-Datasets) pipeline
- `src/softprompt_experiments/models/` — custom model wrappers (SoftPrompt, SquishyPrompt, LoRa, OpenAI_Model)

## What this project is

Research code for "Learning to Translate from Soft to Hard LLM Prompts": train per-task soft prompts on a frozen Llama-3.1-8B-Instruct, then train a LoRA-adapted "mapper" (a.k.a. "translator" — the terms are used interchangeably everywhere) that translates soft prompt embeddings back into natural-language instructions. InSPEcT is the training-free baseline being compared against.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
```

Requires `OPENAI_API_KEY` in the environment for scripts that use `OpenAI_Model`. HF datasets are pulled from the `SoftPromptTranslator/` hub namespace.

## Running experiments

There are no tests or linters. Everything runs through the driver `run_experiment.py`, which imports each name under `softprompt_experiments.scripts.` and calls its `run(args_list)`:

```bash
python -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.train_softprompts
```

The full General-DoD pipeline, in order:

```bash
python -m run_experiment --scripts \
  soft_prompt_mapper.supernat_instruct_DoD.train_softprompts \
  soft_prompt_mapper.supernat_instruct_DoD.compile_mapper_dataset \
  soft_prompt_mapper.supernat_instruct_DoD.train_mapper \
  soft_prompt_mapper.supernat_instruct_DoD.test_mapper
```

All script-specific flags are passed through `parse_known_args`, so any extra CLI args go to every script listed. On the WPI SLURM cluster, jobs are submitted via `run_job.sh` (loads python/3.12.3 + cuda/12.9.0 modules, activates/creates `.venv`, sets `PYTORCH_ALLOC_CONF=expandable_segments:True`) or `job.sh`.

## Pipeline architecture (supernat_instruct_DoD)

Each stage writes to a directory that the next stage reads via a matching CLI arg — when overriding defaults, keep them chained (e.g. `train_softprompts --save_dir X` requires `compile_mapper_dataset --trained_soft_prompts_dir X`):

1. **`train_softprompts.py`** — for each task in the HF dataset, trains a 20-token `SoftPrompt` against frozen Llama-3.1-8B-Instruct (model name hardcoded here) with early stopping on val loss; saves `<save_dir>/<task_name>/softprompt.pt` and appends to `<save_dir>/training_stats.csv`. Skips tasks whose `softprompt.pt` already exists, so it is resumable; `--resume` additionally warm-starts from prior weights.
2. **`compile_mapper_dataset.py`** — pairs each saved soft prompt tensor with its task instruction(s) (train split is exploded over `reduced_instructions` paraphrases + the original instruction) into `train_mapper_dataset.pt` / `val_mapper_dataset.pt` under `--compiled_dataset_dir`.
3. **`train_mapper.py`** — trains the translator: PEFT LoRA (note: the HF `peft` library here, not the custom `models/lora.py`) on Llama, conditioning on the frozen soft prompt embeddings prepended to the target instruction text. Saves adapters via `save_pretrained` to `<lora_save_dir>/<dataset_name>/<model_name>`.
4. **`test_mapper.py`** — loads the LoRA adapters with `PeftModel.from_pretrained`, generates verbalizations for val soft prompts, scores with ROUGE, writes `verbalizations.json` next to the adapters.

Supporting scripts in the same folder: `generate_dataset.py` / `filter_dataset.py` / `generate_paraphrasals.py` (dataset construction/augmentation), `train_softprompts_peft.py` (PEFT-based soft prompt variant used by the InSPEcT track), `compile_dataset_for_InSPEcT.py` / `apply_InSPEcT_on_DoD.py` / `test_InSPEcT_on_DoD.py` (InSPEcT baseline), `save_few_shot_examples.py`, `compute_cosine_similarity_matrix.py`.

### Script conventions

- Every script exposes `run(args_list)` and does its own `argparse` with `parse_known_args(args_list)` — the driver errors if `run` is missing.
- All scripts share `--seed` default 47, prepend soft prompt embeddings to input text embeddings, mask soft prompt positions in labels with `-100`, and pad the attention mask with 1s for the soft prompt length. `num_tokens` defaults to 20 and must match between training and downstream stages.
- Llama has no pad token; scripts set `tokenizer.pad_token = tokenizer.eos_token`.

## Models (`src/softprompt_experiments/models/`)

- **`softprompt.py` — `SoftPrompt`**: the custom soft prompt implementation (an `nn.Module` whose only parameter is `prompt_embeddings`). The wrapped model/tokenizer/embeddings are attached via `object.__setattr__` deliberately, so they are *not* registered as child modules and stay out of `state_dict()`/`parameters()`. `forward()` returns `[1, num_tokens, embed_dim]`; callers `.expand()` it across the batch. Saved with `save_softprompt(dir)` → `dir/softprompt.pt` containing `prompt_embeddings`, `initial_tokens`, `initial_embeddings`, `num_tokens`. Also provides interpretability utilities (`get_prompt_logits`, `get_parsability`, nearest-token projection).
- **`squishyprompt.py` — `SquishyPrompt`**: subclass of `SoftPrompt` that adds a parsability regularization term (`CE + lambd * parsability`) to the loss.
- **`lora.py` — `LoRa` / `LoRALinear`**: a from-scratch LoRA that wraps only `q_proj`/`v_proj` linears, with a `disable_lora` context manager for generating from the base model. Note this is separate from the HF `peft` library, which is what `train_mapper.py`/`test_mapper.py` actually use — don't confuse the two; their checkpoint formats are incompatible (`lora.pt` vs PEFT `save_pretrained` directories).
- **`openai_model.py` — `OpenAI_Model`**: thin chat-completions wrapper used for LLM-as-judge scoring; parses a `Rating: <n>` from the response and returns `-1.0` on parse failure.
