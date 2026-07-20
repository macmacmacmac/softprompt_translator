import os
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import time
from typing import List, Dict, Tuple

from openai import OpenAI, RateLimitError
from tqdm import tqdm


# ┌───────────────────────────────────────────────┐
# │                 HELPER METHODS                │
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
