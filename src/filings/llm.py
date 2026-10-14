"""Gemini calls with JSON-schema output, retries and token accounting. All model calls go through parse()."""

import time
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from filings.config import settings

_client: genai.Client | None = None  # created on first use so tests don't need a key


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise SystemExit("GEMINI_API_KEY is not set (add it to .env).")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


@dataclass
class LLMResult:
    parsed: BaseModel | None  # None when the response was blocked or didn't match the schema
    blocked: bool
    input_tokens: int
    output_tokens: int  # includes thinking tokens, which are billed as output
    model: str


def parse(
    *,
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    temperature: float = 0.0,
) -> LLMResult:
    if not model.startswith("gemini"):
        raise SystemExit(f"{model!r} is not a Gemini model; check ANSWER_MODEL / JUDGE_MODEL.")
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )

    # Retry rate limits and server errors with exponential backoff (15s up to 240s).
    for attempt in range(6):
        try:
            response = _get_client().models.generate_content(model=model, contents=user, config=config)
            break
        except genai_errors.APIError as e:
            retryable = e.code == 429 or (e.code or 0) >= 500
            if not retryable or attempt == 5:
                raise
            wait = 15 * 2**attempt
            print(f"Gemini error {e.code}; retrying in {wait}s (attempt {attempt + 1}/5)")
            time.sleep(wait)

    usage = response.usage_metadata
    input_tokens = (usage.prompt_token_count or 0) if usage else 0
    output_tokens = 0
    if usage:
        output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)

    parsed = response.parsed if isinstance(response.parsed, schema) else None
    return LLMResult(
        parsed=parsed,
        blocked=parsed is None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=response.model_version or model,
    )
