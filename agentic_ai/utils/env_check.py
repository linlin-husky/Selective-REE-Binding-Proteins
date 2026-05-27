"""Environment + OpenAI connectivity helpers (Week 1, Block 1)."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

_DEFAULT_MODEL = "gpt-4o-mini"
_PING_PROMPT = "Reply with exactly: connection ok"


def load_api_key() -> str:
    """
    Loads the OpenAI API key from a .env file or the environment.
    return : The validated, whitespace-stripped API key as a string.
    raises : RuntimeError if the key is missing, empty, or still set to
             the placeholder value.
    """
    load_dotenv()

    key = os.getenv("OPENAI_API_KEY", "").strip()

    if not key or key.startswith("sk-replace"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Edit .env and paste your real key."
        )

    return key


def ping_openai(api_key: str = None, model: str = _DEFAULT_MODEL) -> str:
    """
    Sends a minimal prompt to the OpenAI Chat Completions API to verify
    end-to-end connectivity.
    @param api_key: The OpenAI API key used to authenticate the client.
    @param model: The model identifier to invoke (default: gpt-4o-mini).
    return : The model's text response, stripped of surrounding whitespace.
    """
    if api_key is None:
        api_key = ""

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _PING_PROMPT}],
        max_tokens=20,
    )

    return (response.choices[0].message.content or "").strip()
