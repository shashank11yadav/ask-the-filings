"""
llm.py - the ONE place in the project that talks to the Gemini API.

Every other file (answering questions, generating eval questions, judging answers)
calls `parse()` from here. Keeping it in one file means that if you ever switch
models or providers again, you only change this file.

What `parse()` does:
  1. Sends a "system instruction" (the rules) + a "user" message (the actual task) to Gemini.
  2. Forces Gemini to reply in JSON that matches a Pydantic class you pass in
     (so we get a Python object back, not free text we'd have to parse ourselves).
  3. Returns that object plus how many tokens were used, so we can compute cost.
"""

import os
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from filings.config import settings

# We create the client lazily (on first use) so that importing this file in tests doesn't need a key.
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise SystemExit(
                "GEMINI_API_KEY is missing. Add a line like GEMINI_API_KEY=your-key to .env "
                "in the project folder (no quotes or spaces around =)."
            )
        # Pass the key from .env explicitly.
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


@dataclass
class LLMResult:
    """What we hand back to the rest of the code after one Gemini call."""

    parsed: BaseModel | None  # the reply as a Pydantic object, or None if Gemini blocked/failed
    blocked: bool  # True if Gemini refused to answer (e.g. safety filter)
    input_tokens: int  # tokens we sent (prompt) - you pay for these
    output_tokens: int  # tokens Gemini wrote, INCLUDING its hidden "thinking" tokens - you pay for these too
    model: str  # which model actually answered


def parse(
    *,
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    temperature: float = 0.0,
) -> LLMResult:
    """Ask Gemini a question and get back an instance of `schema`.

    model       - e.g. "gemini-3.8-flash" (set in .env)
    system      - the rules / role for the model
    user        - the actual content: retrieved excerpts + the question
    schema      - a Pydantic class describing the JSON shape we want back
    temperature - 0.0 = as deterministic as possible (good for evals, so runs are comparable)
    """
    # Catch a common setup mistake early: a non-Gemini model name left over in .env.
    if not model.startswith("gemini"):
        # Terminal environment variables override .env, so say which one the value came from.
        from_shell = [k for k in ("ANSWER_MODEL", "JUDGE_MODEL") if os.environ.get(k) == model]
        if from_shell:
            where = (
                f"It comes from your terminal environment ({', '.join(from_shell)}), which overrides .env.\n"
                f"Fix: run `unset {' '.join(from_shell)}` and remove any `export {from_shell[0]}=...` line "
                "from ~/.zshrc, then open a new terminal."
            )
        else:
            where = "It comes from .env: set ANSWER_MODEL and JUDGE_MODEL to gemini-3.8-flash (or delete those lines)."
        raise SystemExit(f"Model {model!r} is not a Gemini model. {where}")
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        # These two lines are "structured output": Gemini must return JSON matching `schema`,
        # and the SDK automatically converts it into a `schema` object at `response.parsed`.
        response_mime_type="application/json",
        response_schema=schema,
    )

    # Long runs (question generation, evals) can hit Gemini's rate limit (HTTP 429 "too many
    # requests") or a temporary server error (5xx). Instead of crashing, wait and try again:
    # 15s, 30s, 60s, 120s, 240s. Other errors (bad key, bad model name) are raised immediately.
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

    # usage_metadata tells us how many tokens were used. Fields can be None, so default to 0.
    usage = response.usage_metadata
    input_tokens = (usage.prompt_token_count or 0) if usage else 0
    output_tokens = 0
    if usage:
        output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)

    # If the safety filter blocked the prompt or the answer, `parsed` will be empty.
    parsed = response.parsed if isinstance(response.parsed, schema) else None
    return LLMResult(
        parsed=parsed,
        blocked=parsed is None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=response.model_version or model,
    )
