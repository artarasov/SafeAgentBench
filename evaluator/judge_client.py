"""
Environment-driven OpenAI-compatible client for the SafeAgentBench LLM judges.

Backward compatible by design: with no environment configured it behaves like
the original code path — the OpenAI API, using ``OPENAI_API_KEY``, with the
model the caller passes. Set the variables below to point the judges at
OpenRouter or any other OpenAI-compatible endpoint without touching the
evaluator logic.

Environment variables (all optional):
  JUDGE_API_KEY   API key for the judge endpoint. Falls back to OPENAI_API_KEY.
  JUDGE_BASE_URL  Base URL of the endpoint, e.g. ``https://openrouter.ai/api/v1``.
                  Falls back to OPENAI_BASE_URL; unset => api.openai.com.
  JUDGE_MODEL     Override the model id for every judge call
                  (e.g. ``openai/gpt-4o`` on OpenRouter). Unset => caller default.
  OPENROUTER_REFERER / OPENROUTER_TITLE
                  Optional OpenRouter attribution headers (HTTP-Referer / X-Title).

Requires ``openai>=1.0`` (the repository already pins openai==1.52.2).
"""
import os
import time

from openai import OpenAI


def get_client() -> OpenAI:
    """Build an OpenAI client from the environment (OpenAI / OpenRouter / …)."""
    kwargs = {}
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    headers = {}
    if os.getenv("OPENROUTER_REFERER"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_REFERER"]
    if os.getenv("OPENROUTER_TITLE"):
        headers["X-Title"] = os.environ["OPENROUTER_TITLE"]
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def resolve_model(default: str) -> str:
    """Let JUDGE_MODEL override the caller's model id (useful across providers)."""
    return os.getenv("JUDGE_MODEL") or default


def chat_completion(model, system_prompt, prompt, temperature=0.2,
                    max_tokens=1024, max_retries=5, retry_delay=5):
    """Call chat.completions with retries.

    Returns ``(response, retries)`` — the raw openai response object and the
    number of retries performed — mirroring the original evaluator helpers so
    existing call sites keep working. Raises after ``max_retries`` failures.
    """
    client = get_client()
    model = resolve_model(model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    # Older models take max_tokens; newer OpenAI models reject it and require
    # max_completion_tokens. Start with max_tokens (what the benchmark has always
    # sent) and switch automatically if the API tells us to.
    token_param = "max_tokens"
    last_err = None
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **{token_param: max_tokens},
            )
            return response, attempt
        except Exception as e:  # broad: OpenAI and OpenRouter raise different types
            if token_param == "max_tokens" and "max_completion_tokens" in str(e):
                token_param = "max_completion_tokens"  # not a real failure: swap and retry
                continue
            last_err = e
            attempt += 1
            print(f"[judge] request failed ({e}); "
                  f"retry {attempt}/{max_retries} in {retry_delay}s...")
            time.sleep(retry_delay)
    raise RuntimeError(
        f"Max retries reached, could not complete the request: {last_err}")
