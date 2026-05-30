"""Tests for the environment + OpenAI connectivity helpers
(Week 1 Block 4.5).

load_api_key() is pure logic and fully tested here. ping_openai()
makes a real API call so it is only smoke-tested as an opt-in.
"""
from __future__ import annotations

import pytest

from agentic_ai.utils.env_check import load_api_key


# ---------------------------------------------------------------------------
# load_api_key: success path
# ---------------------------------------------------------------------------

def test_load_api_key_returns_value_from_environment(monkeypatch):
    """
    Verifies that a valid key in the environment is returned as-is.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-valid-key-1234567890")

    key = load_api_key()

    assert key == "sk-test-valid-key-1234567890"


def test_load_api_key_strips_surrounding_whitespace(monkeypatch):
    """
    Verifies that whitespace (a common copy-paste artifact) is
    stripped from the returned key.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "  sk-test-key-with-spaces  \n")

    key = load_api_key()

    assert key == "sk-test-key-with-spaces"


# ---------------------------------------------------------------------------
# load_api_key: failure paths
# ---------------------------------------------------------------------------

def test_load_api_key_raises_when_key_missing(monkeypatch):
    """
    Verifies that a missing OPENAI_API_KEY raises a clear RuntimeError.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Stub load_dotenv so it does not silently populate the key from a
    # local .env file during the test.
    monkeypatch.setattr(
        "agentic_ai.utils.env_check.load_dotenv",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        load_api_key()


def test_load_api_key_raises_when_key_empty(monkeypatch):
    """
    Verifies that an empty string key is rejected. Catches the common
    failure mode where the .env file has the line but no value.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(
        "agentic_ai.utils.env_check.load_dotenv",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        load_api_key()


def test_load_api_key_raises_when_key_whitespace_only(monkeypatch):
    """
    Verifies that a whitespace-only key is rejected after stripping.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "   \n\t  ")
    monkeypatch.setattr(
        "agentic_ai.utils.env_check.load_dotenv",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        load_api_key()


def test_load_api_key_raises_when_key_is_placeholder(monkeypatch):
    """
    Verifies that a key still set to the 'sk-replace...' placeholder
    is rejected. Catches the failure mode where someone copied the
    .env.example template without filling in their real key.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-replace-with-your-real-key")
    monkeypatch.setattr(
        "agentic_ai.utils.env_check.load_dotenv",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        load_api_key()


def test_load_api_key_error_message_guides_user_to_env_file(monkeypatch):
    """
    Verifies the error message mentions the .env file so users know
    where to set the key. Small but valuable for first-time setup.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "agentic_ai.utils.env_check.load_dotenv",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError) as exc_info:
        load_api_key()

    assert ".env" in str(exc_info.value)
