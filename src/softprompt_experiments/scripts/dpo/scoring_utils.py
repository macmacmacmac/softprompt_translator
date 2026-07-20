import os
import contextlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from openai import OpenAI, RateLimitError
from tqdm import tqdm
from typing import List, Dict, Tuple
import evaluate
from vllm import LLM, SamplingParams


# ┌───────────────────────────────────────────────┐
# │                PROMPT TEMPLATES               │
# └───────────────────────────────────────────────┘
SYS_PROMPT = """You are a helpful assistant. Follow the task exactly."""

USR_PROMPT_TEMPLATE = """{task_prompt}

Input:
{input}

Output:
"""


FULL_PROMPT_TEMPLATE = SYS_PROMPT + USR_PROMPT_TEMPLATE


# ┌───────────────────────────────────────────────┐
# │                MODULE GLOBALS                 │
# └───────────────────────────────────────────────┘
# ROUGE-L Scorer
ROUGE_METRIC = None

# Set by init_scoring()
SCORE_FN = None
SCORE_MODEL_NAME = None
SCORE_BACKEND = None
SCORE_BATCH_SIZE = None
SCORE_MAX_NEW_TOKENS = None
SHARED_SCORE_MODEL = None
OPENAI_CONCURRENCY = None
DEVICE = None
DTYPE = None
OPENAI_CLIENT = None
score_model = None
score_tokenizer = None


# ┌───────────────────────────────────────────────┐
# │              INITIALIZATION                   │
# └───────────────────────────────────────────────┘
def init_scoring(
    score_fn,
    score_model_name,
    score_batch_size,
    score_max_new_tokens,
    openai_concurrency,
    device,
    dtype,
    translator_model,
    lora_model_name,
    use_vllm,
    vllm_gpu_memory_utilization,
):
    """
    Initialise every module-level global the scoring helpers depend on.
    Called once from the main script's run() after CLI-arg parsing.
    """
    global SCORE_FN, SCORE_MODEL_NAME, SCORE_BACKEND
    global SCORE_BATCH_SIZE, SCORE_MAX_NEW_TOKENS, OPENAI_CONCURRENCY
    global DEVICE, DTYPE
    global OPENAI_CLIENT, score_model, score_tokenizer, SHARED_SCORE_MODEL

    SCORE_FN = score_fn
    SCORE_MODEL_NAME = score_model_name
    
    # Route to vllm if requested, compatible, and using ROUGE-L
    # (LOGPROB already shares weights efficiently via HF, no vllm needed)
    if use_vllm and get_score_backend(score_model_name) == "hf" and SCORE_FN == "ROUGE-L":
        SCORE_BACKEND = "vllm"
    else:
        SCORE_BACKEND = get_score_backend(score_model_name)
        
    SCORE_BATCH_SIZE = score_batch_size
    SCORE_MAX_NEW_TOKENS = score_max_new_tokens
    OPENAI_CONCURRENCY = openai_concurrency
    DEVICE = device
    DTYPE = dtype

    # ── Validate ──
    if SCORE_FN not in ("ROUGE-L", "LOGPROB"):
        print(f"Unsupported Score Function: {SCORE_FN}!")
        exit(1)

    # ── Backend-specific setup ──
    if SCORE_BACKEND == "openai":
        # LOGPROB needs raw logits over the full vocab, which the OpenAI API doesn't expose
        if SCORE_FN != "ROUGE-L":
            raise ValueError(
                f"OpenAI score model `{SCORE_MODEL_NAME}` only supports the "
                f"ROUGE-L score function, got: {SCORE_FN}"
            )
        OPENAI_CLIENT = init_openai_client()

    elif SCORE_BACKEND == "vllm":
        print(f"Loading score model {SCORE_MODEL_NAME} with vLLM (gpu_memory_utilization={vllm_gpu_memory_utilization})...")
        # Ensure we use bfloat16 for vllm if DTYPE is bfloat16, else let it auto-detect or use float32
        vllm_dtype = "bfloat16" if DTYPE == torch.bfloat16 else "float32"
        score_model = LLM(
            model=SCORE_MODEL_NAME, 
            dtype=vllm_dtype,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            enforce_eager=True, # Saves CUDA graph memory
        )
        SHARED_SCORE_MODEL = False

    else:
        score_tokenizer = AutoTokenizer.from_pretrained(SCORE_MODEL_NAME)

        # Do this only if score model is from Llama family
        score_tokenizer.pad_token = score_tokenizer.eos_token

        if SCORE_FN == "ROUGE-L":
            # Batched decoder-only generation needs left padding so every prompt sits
            # flush against its continuation (Llama defaults to right padding; the
            # LOGPROB path pads manually and is unaffected either way)
            score_tokenizer.padding_side = "left"

        if SCORE_MODEL_NAME == lora_model_name:
            # Avoid loading a second bf16 copy of the same base model — reuse the
            # translator's underlying weights, with the LoRA adapter disabled during scoring
            print("Score model matches translator base model — sharing weights (LoRA disabled during scoring).")
            score_model = translator_model
            SHARED_SCORE_MODEL = True
        else:
            print(f"Loading score model {SCORE_MODEL_NAME}...")
            score_model = AutoModelForCausalLM.from_pretrained(SCORE_MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
            score_model.eval()
            SHARED_SCORE_MODEL = False


# ┌───────────────────────────────────────────────┐
# │            OPENAI HELPER METHODS              │
# └───────────────────────────────────────────────┘
def init_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please add `OPENAI_API_KEY` inside a .env file in project root")

    return OpenAI(
        api_key=api_key,
        max_retries=5,
    )


def prompt_openai_model(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 5,
    **kwargs
):
    defaults = {
        "model": model_name,
    }
    params = {**defaults, **kwargs}
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                **params,
            )

            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            print(
                f"Rate limited. Retrying in {wait_time}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_time)
    return ""


def generate_outputs_concurrently(
    client: OpenAI,
    model_name: str,
    translations: List[str],
    train_instances: List[Dict],
    sys_prompt: str,
    usr_prompt_template: str,
    concurrency: int
) -> List[List[str]]:
    # Build the full (translation x train_instance) job list up front so all
    # OpenAI calls can be fired concurrently instead of one blocking call at a time.
    # Each job carries its own fully-formatted system/user prompt plus the (i, j)
    # coordinates needed to place its result back into y_hat in the right slot.
    jobs = []
    for i, translation in enumerate(translations):
        # Prep system prompt based on hard prompt (translation)
        for j, instance in enumerate(train_instances):
            user_prompt = usr_prompt_template.format(task_prompt=translation, input=instance["input"])
            jobs.append((i, j, sys_prompt, user_prompt))

    # Preallocate y_hat[i][j] so results can be written back out of order as
    # futures complete, regardless of scheduling.
    y_hat = [[None] * len(train_instances) for _ in translations]

    # The OpenAI client is thread-safe and these calls are I/O-bound (waiting on
    # the network), so a thread pool -- not a process pool -- is the right tool
    # here: it parallelizes the waiting without paying for GIL-bound work.
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_coords: Dict[Future[str], Tuple[int, int]] = {
            executor.submit(prompt_openai_model, client, model_name, system_prompt, user_prompt): (i, j)
            for (i, j, system_prompt, user_prompt) in jobs
        }

        for future in tqdm(as_completed(future_to_coords), total=len(jobs), desc="Scoring", leave=False):
            i, j = future_to_coords[future]
            # .result() re-raises any exception from the worker (e.g. exhausted
            # retries in prompt_openai_model), matching the previous fail-fast behavior
            y_hat[i][j] = future.result()

    return y_hat


# ┌───────────────────────────────────────────────┐
# │              SCORING FUNCTIONS                │
# └───────────────────────────────────────────────┘
def get_score_backend(score_model_name: str) -> str:
    # HF model ids contain a `/` (e.g. `meta-llama/Llama-3.1-8B-Instruct`) -> run locally.
    # Otherwise (e.g. `gpt-4o-mini`) -> use the OpenAI API.
    return "hf" if "/" in score_model_name else "openai"


def get_scores(
        translations: List[str],
        train_instances: List[Dict]
    ) -> torch.Tensor:
    if SCORE_FN == "ROUGE-L":
        return get_rougeL_scores(
            translations,
            train_instances
        )
    elif SCORE_FN == "LOGPROB":
        return get_logprob_scores(
            translations,
            train_instances
        )
    raise ValueError(f"Unsupported score function: {SCORE_FN}")


def get_rougeL_scores(
        translations: List[str],
        train_instances: List[Dict]
    ) -> torch.Tensor:
    global ROUGE_METRIC
    if ROUGE_METRIC is None:
        ROUGE_METRIC = evaluate.load("rouge")

    y = [t["output"] for t in train_instances]

    # Produce y_hat[i][j] = score model's output for (translation i, train instance j),
    # either with the local HF score model, vLLM, or via concurrent OpenAI API calls
    if SCORE_BACKEND == "hf":
        y_hat = generate_outputs_locally(translations, train_instances)
    elif SCORE_BACKEND == "vllm":
        y_hat = generate_outputs_vllm(translations, train_instances)
    else:
        y_hat = generate_outputs_concurrently(
            OPENAI_CLIENT,
            SCORE_MODEL_NAME,
            translations,
            train_instances,
            SYS_PROMPT,
            USR_PROMPT_TEMPLATE,
            OPENAI_CONCURRENCY,
        )

    # Compute ROUGE-L between y and y_hat per translation, in original translation order
    rougeL_scores = [
        ROUGE_METRIC.compute(predictions=y_hat[i], references=y)["rougeL"]
        for i in range(len(translations))
    ]
    return torch.tensor(rougeL_scores)


def generate_outputs_vllm(
        translations: List[str],
        train_instances: List[Dict]
    ) -> List[List[str]]:

    # Build prompts and job mapping
    jobs = []
    prompts = []
    for i, translation in enumerate(translations):
        for j, instance in enumerate(train_instances):
            prompt = FULL_PROMPT_TEMPLATE.format(task_prompt = translation, input = instance["input"])
            jobs.append((i, j))
            prompts.append(prompt)

    y_hat = [[None] * len(train_instances) for _ in translations]

    # Deterministic generation
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=SCORE_MAX_NEW_TOKENS,
    )
    
    # vLLM handles batching internally
    outputs = score_model.generate(prompts, sampling_params, use_tqdm=True)

    for (i, j), output in zip(jobs, outputs):
        y_hat[i][j] = output.outputs[0].text.strip()

    return y_hat


def generate_outputs_locally(
        translations: List[str],
        train_instances: List[Dict]
    ) -> List[List[str]]:
    # Build the full (translation x train_instance) job list up front, mirroring the
    # OpenAI path: each job carries its fully-formatted prompt plus the (i, j)
    # coordinates needed to place its result back into y_hat in the right slot.
    jobs = []
    for i, translation in enumerate(translations):
        for j, instance in enumerate(train_instances):
            # Same raw prompt format the LOGPROB score path uses
            prompt = FULL_PROMPT_TEMPLATE.format(task_prompt = translation, input = instance["input"])
            jobs.append((i, j, prompt))

    # Preallocate y_hat[i][j] so results can be written back via each job's coordinates
    y_hat = [[None] * len(train_instances) for _ in translations]

    # Generate in micro-batches of SCORE_BATCH_SIZE to bound peak memory, with the
    # LoRA adapter disabled when the score model shares weights with the translator
    scoring_ctx = score_model.disable_adapter() if SHARED_SCORE_MODEL else contextlib.nullcontext()
    with scoring_ctx:
        for start in tqdm(range(0, len(jobs), SCORE_BATCH_SIZE), desc="Scoring", leave=False):
            chunk = jobs[start:start + SCORE_BATCH_SIZE]
            chunk_prompts = [prompt for (_, _, prompt) in chunk]

            # score_tokenizer.padding_side is set to "left" at load time — batched
            # decoder-only generation needs every prompt flush against its continuation,
            # which also gives all rows the same padded prompt length for slicing below.
            inputs = score_tokenizer(chunk_prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Deterministic (greedy) decoding
            with torch.no_grad():
                gen_ids = score_model.generate(
                    **inputs,
                    max_new_tokens = SCORE_MAX_NEW_TOKENS,
                    do_sample = False,
                    pad_token_id = score_tokenizer.eos_token_id,
                )

            # Drop the (uniform, left-padded) prompt span, keep only the continuations
            outputs = score_tokenizer.batch_decode(gen_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Write results back into y_hat via each job's coordinates
            for (i, j, _), output in zip(chunk, outputs):
                y_hat[i][j] = output.strip()

    return y_hat


def get_logprob_scores(
        translations: List[str],
        train_instances: List[Dict]
    ) -> torch.Tensor:
    logprob_scores = []
    y = [t["output"] for t in train_instances]
    for translation in translations:
        # Create Prompts for all instances within a task
        prompts = [FULL_PROMPT_TEMPLATE.format(task_prompt = translation, input = t["input"]) for t in train_instances]

        # Log prob of each true output given its (translation + input) prompt, using the score model
        # (batched into a single forward pass across all instances in the task)
        instance_logprobs = get_logprob_of_output_given_prompt(score_model, score_tokenizer, prompts, y)

        # Average over instances so translations are compared on the same per-instance scale
        logprob_scores.append(instance_logprobs.mean().item())

    return torch.tensor(logprob_scores)


def get_logprob_of_translation_given_soft_prompt(
        model: PeftModel,
        tokenizer: AutoTokenizer,
        translation: str,
        soft_prompt_embeds: torch.Tensor
    ) -> float:
    # Tokenize the translation (append EOS so the model learns/expects an explicit stop token)
    inputs = tokenizer(translation + tokenizer.eos_token, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Embed token ids via the model's word embedding layer (works through PeftModel too)
    translation_embeds = model.get_input_embeddings()(input_ids).to(DTYPE)

    # Prepare labels
    labels = input_ids.clone()

    # Move soft prompt embeds to DEVICE and DTYPE and add batch_dim if necessary
    soft_prompt_embeds = soft_prompt_embeds.to(device=DEVICE, dtype=DTYPE)
    if soft_prompt_embeds.dim() == 2:
        soft_prompt_embeds = soft_prompt_embeds.unsqueeze(0)

    # Get num of soft tokens
    soft_prompt_len = soft_prompt_embeds.shape[1]

    # Prepend the soft prompt embeddings to the sequence embeddings
    inputs_embeds = torch.cat([soft_prompt_embeds, translation_embeds], dim=1)

    # Pad attention mask with 1s for the soft prompt span
    soft_prompt_mask = torch.ones((1, soft_prompt_len), dtype=attention_mask.dtype, device=DEVICE)
    full_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)

    # Pad labels with -100 for the soft prompt span so no loss/log-prob is attributed to it
    soft_prompt_labels = torch.full((1, soft_prompt_len), -100, dtype=labels.dtype, device=DEVICE)
    full_labels = torch.cat([soft_prompt_labels, labels], dim=1)

    # Perform forward pass (no `labels=` kwarg — we need raw logits to hand-compute per-token log-probs)
    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask)
        logits = outputs.logits                                 # (batch_size, seq_len, vocab_size)

    # Sum log-probs over the valid (non -100) sequence positions -> log P(translation | soft_prompt)
    total_log_prob = _sum_sequence_logprob(logits, full_labels)
    return total_log_prob.item()


def get_logprob_of_output_given_prompt(
        model: AutoModelForCausalLM | PeftModel,
        tokenizer,
        prompts: List[str],
        outputs: List[str]
    ) -> torch.Tensor:
    # Tokenize prompt and output separately and concatenate ids -- this makes the
    # prompt/output boundary exact by construction (no cross-junction BPE merging
    # like we'd risk by tokenizing the concatenated string, and no second
    # measurement pass needed to find where the prompt span ends)
    batch_input_ids = []
    batch_labels = []
    for prompt, output in zip(prompts, outputs):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

        row_ids = prompt_ids + output_ids
        row_labels = [-100] * len(prompt_ids) + output_ids   # mask the prompt span, score only the output (+ EOS)

        batch_input_ids.append(row_ids)
        batch_labels.append(row_labels)

    # Score in micro-batches of SCORE_BATCH_SIZE rows instead of one giant forward pass —
    # keeps peak memory (activations + fp32 logits over the full vocab) bounded regardless
    # of how many train instances a task has
    per_row_scores = []
    scoring_ctx = model.disable_adapter() if SHARED_SCORE_MODEL else contextlib.nullcontext()
    with scoring_ctx:
        for start in range(0, len(batch_input_ids), SCORE_BATCH_SIZE):
            chunk_ids = batch_input_ids[start:start + SCORE_BATCH_SIZE]
            chunk_labels = batch_labels[start:start + SCORE_BATCH_SIZE]

            # Manually right-pad this chunk to its own max sequence length
            max_len = max(len(ids) for ids in chunk_ids)
            input_ids, attention_mask, labels = [], [], []
            for ids, lbls in zip(chunk_ids, chunk_labels):
                pad_len = max_len - len(ids)
                input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
                labels.append(lbls + [-100] * pad_len)

            input_ids = torch.tensor(input_ids, device=DEVICE)              # (chunk_size, seq_len)
            attention_mask = torch.tensor(attention_mask, device=DEVICE)    # (chunk_size, seq_len)
            labels = torch.tensor(labels, device=DEVICE)                    # (chunk_size, seq_len)

            # Single forward pass over this chunk (no `labels=` kwarg — we hand-compute log-probs)
            with torch.no_grad():
                outputs_hf = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs_hf.logits                           # (chunk_size, seq_len, vocab_size)

            # Per row: sum of log P(token | preceding tokens) over the scored (non -100) positions
            summed_log_probs = _sum_sequence_logprob(logits, labels)                # (chunk_size,)

            # Count scored tokens per row; [:, 1:] mirrors the shift inside _sum_sequence_logprob
            # (position 0 is never scored — nothing precedes it)
            token_counts = (labels[:, 1:] != -100).sum(dim=-1)                      # (chunk_size,)

            # Average per token, so instances with longer outputs aren't systematically more negative
            per_row_scores.append(summed_log_probs / token_counts)

    # Concatenate chunk results back into a single (num_rows,) tensor, preserving input order
    return torch.cat(per_row_scores, dim=0)


def _sum_sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Shift so that logits at position i predict the token at position i+1
    shift_logits = logits[:, :-1, :]            # (batch_size, seq_len - 1, vocab_size)
    shift_labels = labels[:, 1:]                # (batch_size, seq_len - 1)

    # Fused cross-entropy = -log P(label); ignore_index zeroes the -100 positions
    # without materializing a full-vocab log-prob tensor like log_softmax would
    loss = F.cross_entropy(
        shift_logits.flatten(0, 1).float(),
        shift_labels.flatten(),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)

    # Sum log-probs over the valid (non -100) sequence positions
    return -loss.sum(dim=-1)   # (batch_size,)
