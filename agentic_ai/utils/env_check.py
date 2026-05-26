"""Environment + OpenAI connectivity helpers (Week 1, Block 1)."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

_DEFAULT_MODEL = "gpt-4o-mini"
_PING_PROMPT = "Reply with exactly: connection ok"


def load_api_key() -> str:
    """Load OPENAI_API_KEY from .env or the environment."""
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("sk-replace"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Edit .env and paste your real key."
        )
    return key


def ping_openai(api_key: str, model: str = _DEFAULT_MODEL) -> str:
    """Send a minimal prompt and return the model's text response."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _PING_PROMPT}],
        max_tokens=20,
    )
    return (response.choices[0].message.content or "").strip()
