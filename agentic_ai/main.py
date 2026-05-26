"""Entry point for the REE-binding agentic pipeline.

Week 1 / Block 1: minimal smoke test confirming OpenAI API connectivity.
"""
from __future__ import annotations

import sys

from agentic_ai.utils.env_check import load_api_key, ping_openai


def main() -> int:
    """Run the Block 1 smoke test. Returns a shell-style exit code."""
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
