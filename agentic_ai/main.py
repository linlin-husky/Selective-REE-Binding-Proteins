"""Entry point for the REE-binding agentic pipeline.

Week 1 / Block 1: minimal smoke test confirming OpenAI API connectivity.
Block 4 will replace main()'s body with the CrewAI orchestration call.
"""
from __future__ import annotations

import sys

from agentic_ai.utils.env_check import load_api_key, ping_openai


def main() -> int:
    """
    Runs the Block 1 connectivity smoke test by loading the API key and
    issuing a single round-trip request to the OpenAI Chat Completions API.
    return : Shell-style exit code (0 on success, 1 on configuration error).
    """
    try:
        api_key = load_api_key()
    except RuntimeError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 1

    reply = ping_openai(api_key)
    print(f"OpenAI says: {reply}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
    