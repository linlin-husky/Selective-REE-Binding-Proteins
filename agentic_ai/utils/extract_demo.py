"""Structured-extraction proof-of-concept (Week 1, Block 2).

Demonstrates that an LLM can be forced to return data conforming to the
`ProteinData` schema by using OpenAI's structured-output parsing. This
is a manual stand-in for what the CrewAI researcher agent will do in
Block 4.

Run with: python -m agentic_ai.utils.extract_demo
"""
from __future__ import annotations
import sys
from openai import OpenAI
from agentic_ai.schemas import ProteinData
from agentic_ai.utils.env_check import load_api_key

_EXTRACTION_MODEL = "gpt-4o-mini"

# Synthetic abstract used as a controlled test input. Block 3 will
# replace this with real paragraphs from downloaded papers.
_SAMPLE_ABSTRACT = """
We characterized a novel Lanmodulin variant isolated from
Methylobacterium extorquens AM1. The protein, whose full sequence is
MAEKGITSEELEELKEAFRLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTID
FPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREAD
IDGDGQVNYEEFVQMMTAK, was shown to bind Neodymium ions with high
selectivity. Isothermal titration calorimetry measurements yielded
a dissociation constant of 2.4e-12 M, consistent with the picomolar
affinities reported for wild-type LanM.
"""

_EXTRACTION_PROMPT = (
    "Extract the Lanmodulin protein record from the following abstract. "
    "Return exactly one record. If any required field is not explicitly "
    "stated in the text, do not invent a value.\n\n"
    f"Abstract:\n{_SAMPLE_ABSTRACT}"
)


def extract_protein_record(api_key: str = None, text: str = None) -> ProteinData:
    """
    Asks the LLM to extract a single ProteinData record from the supplied
    text, using OpenAI's structured-output parsing to guarantee schema
    conformance.
    @param api_key: The OpenAI API key used to authenticate the client.
    @param text: The scientific abstract or paragraph to extract from.
    return : A validated ProteinData instance.
    raises : ValueError if the LLM returns no parseable record.
    """
    if api_key is None:
        api_key = ""
    if text is None:
        text = _EXTRACTION_PROMPT

    client = OpenAI(api_key=api_key)

    completion = client.beta.chat.completions.parse(
        model=_EXTRACTION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert computational biologist extracting "
                    "structured data from scientific text. Follow the "
                    "schema exactly. Do not invent values.\n\n"
                    "CRITICAL: When extracting numeric values like dissociation "
                    "constants, preserve the full scientific notation including "
                    "the exponent. A value reported as '2.4e-12 M' must be "
                    "extracted as the float 2.4e-12, NOT 2.4. Picomolar Kd values "
                    "are typically in the range 1e-15 to 1e-9."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format=ProteinData,
    )

    record = completion.choices[0].message.parsed
    if record is None:
        raise ValueError("LLM returned no parseable ProteinData record.")

    return record


def main() -> int:
    """
    Runs the structured-extraction demo and prints the parsed record.
    return : Shell-style exit code (0 on success, 1 on error).
    """
    try:
        api_key = load_api_key()
    except RuntimeError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 1

    record = extract_protein_record(api_key=api_key)

    print("Extracted ProteinData record:")
    print(f"  protein_sequence : {record.protein_sequence[:40]}... "
          f"({len(record.protein_sequence)} residues)")
    print(f"  microbial_origin : {record.microbial_origin}")
    print(f"  target_REE       : {record.target_REE}")
    print(f"  binding_affinity : {record.binding_affinity:.2e} M")

    return 0


if __name__ == "__main__":
    sys.exit(main())
