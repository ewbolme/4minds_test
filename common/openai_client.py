import os
import time

import openai as _openai
from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def supports_temperature(model_name: str) -> bool:
    """Return False for o-series reasoning models that reject the temperature parameter."""
    no_temp_prefixes = ("o1", "o3", "o4")
    no_temp_exact = {"gpt-5-nano"}
    return (
        not any(model_name.startswith(p) for p in no_temp_prefixes)
        and model_name not in no_temp_exact
    )


def cached_tokens(completion) -> int:
    """
    Return the number of prompt tokens served from the KV cache.
    Returns 0 if the model or response does not report cache usage.
    Caching is automatic for supported models on prompts ≥ 1024 tokens;
    cached tokens are billed at ~50% of the normal input rate.
    """
    try:
        return completion.usage.prompt_tokens_details.cached_tokens or 0
    except AttributeError:
        return 0


def call_with_retry(fn, max_retries: int = 3, base_delay: float = 5.0):
    """
    Call fn() and retry on transient errors with exponential backoff.
    Backoff schedule (default): 5s → 10s → 20s before giving up.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (
            _openai.APIConnectionError,
            _openai.APITimeoutError,
            _openai.BadRequestError,
            _openai.RateLimitError,
        ) as exc:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(
                f"  [retry {attempt + 1}/{max_retries}] {type(exc).__name__}: {exc} "
                f"— retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
