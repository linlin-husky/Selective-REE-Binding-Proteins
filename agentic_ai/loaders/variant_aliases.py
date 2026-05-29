"""Variant name normalization and classifier for the literature corpus.

This module handles two distinct concerns kept separate by design:

  1. Normalization (`normalize_variant_name`): collapses surface-level
     variations of the same name (case, whitespace, missing tags,
     orphan mutations) into a canonical token. Pure string conditioning,
     no biology.

  2. Classification (added in subsequent commits): maps the canonical
     token to its construct_type, parent_scaffold, and (where relevant)
     a MOESM3 ortholog index for joining. This is the biology layer.

The normalizer runs first. The classifier consumes its output.

Run as a script for a quick smoke test:
    python -m agentic_ai.loaders.variant_aliases
"""
from __future__ import annotations

import re
from typing import Optional

# Organism prefixes that should be capitalized in the canonical form.
# The literature uses '<Organism>-LanM' with a capitalized prefix
# (Mex-LanM, Hans-LanM, Melba-LanM). Lowercase forms like 'mex-LanM'
# are typos to be corrected.
_KNOWN_ORGANISM_PREFIXES = {"mex", "hans", "melba"}

# Bare mutation codes that appear without their parent name in some
# papers. Map them to the canonical <parent>(<mutation>) form.
# Currently just R100K (the Mattocks 2023 Hans-LanM monomeric variant);
# future entries should be added here with a comment citing the paper.
_BARE_MUTATIONS_TO_PARENTED = {
    "R100K": "Hans-LanM(R100K)",  # Mattocks et al. 2023, Nature
}

# Tags appended to the C- or N-terminus for purification or matrix
# conjugation. These do not change the binding behavior of the core
# protein, so we strip them in normalization. If a downstream user
# cares about tagged-vs-untagged distinctions, they can add a
# `notes` field on the variant rather than diverging the variant_id.
_STRIPPABLE_TAGS_SUFFIXES = (
    "-Cys",
    "-GSGC",
    "(GSGC)",
    "-SpyTag",
    "-SpyTag3",
    "-His6",
    "-His10",
)

# Pre-compiled regex for ortholog-ID detection: `o-36`, `O36`, `o36`,
# `O-36` etc. all normalize to the lowercase, hyphenated form.
_ORTHOLOG_ID_PATTERN = re.compile(r"^[Oo]-?(\d+)$")


def normalize_variant_name(raw: str = None) -> Optional[str]:
    """
    Collapses surface-level variations of a variant identifier into a
    canonical string token that the classifier can look up.
    Normalization order matters: rules are applied in the sequence
    below so later rules can assume earlier ones have run.
    @param raw: The variant identifier as it appears in the source
                text (e.g. from a CrewAI agent extraction).
    return : The canonical form as a string, or None if the input is
             empty, whitespace-only, or otherwise unusable.
    """
    if raw is None:
        return None

    # Rule 6: Whitespace and empty handling (first, so later rules can
    # assume non-empty input).
    cleaned = raw.strip()
    if not cleaned:
        return None

    # Rule 4: Ortholog ID canonicalization.
    # 'o36', 'O-36', 'O36', 'o-36' all -> 'o-36'.
    ortholog_match = _ORTHOLOG_ID_PATTERN.match(cleaned)
    if ortholog_match:
        return f"o-{ortholog_match.group(1)}"

    # Rule 5: Strip purification/conjugation tags. Done before other
    # transformations so 'mex-LanM-Cys' becomes 'Mex-LanM' cleanly.
    cleaned = _strip_known_tags(cleaned)

    # Rule 3: Parent bare mutation codes.
    # 'R100K' -> 'Hans-LanM(R100K)'.
    if cleaned in _BARE_MUTATIONS_TO_PARENTED:
        return _BARE_MUTATIONS_TO_PARENTED[cleaned]

    # Rule 2: Internal whitespace handling for known shorthand.
    # '4 P2A' -> '4P2A', 'WT LanM' -> 'WT-LanM'.
    cleaned = _collapse_known_internal_whitespace(cleaned)

    # Rule 1: Organism prefix capitalization.
    # 'mex-LanM' -> 'Mex-LanM'.
    cleaned = _capitalize_organism_prefix(cleaned)

    return cleaned


def _strip_known_tags(text: str) -> str:
    """
    Removes any of the known purification/conjugation tag suffixes
    from the end of a variant name. Applied iteratively in case a
    name has multiple stacked tags.
    @param text: The variant string after initial whitespace strip.
    return : The string with any trailing known tags removed.
    """
    changed = True
    while changed:
        changed = False
        for suffix in _STRIPPABLE_TAGS_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)].rstrip("-")
                changed = True
                break
    return text


def _collapse_known_internal_whitespace(text: str) -> str:
    """
    Removes internal whitespace in cases known to be typos. We do NOT
    blindly strip all internal whitespace because some real names
    contain spaces meaningfully (none in the current corpus, but we
    leave the door open).
    @param text: The variant string.
    return : The string with known typo-spaces fixed.
    """
    # '4 P2A' -> '4P2A' (and similar shorthand)
    text = re.sub(r"^(\d)\s+([A-Z]\d+[A-Z])$", r"\1\2", text)

    # 'WT LanM' -> 'WT-LanM'
    text = re.sub(r"^WT\s+LanM$", "WT-LanM", text)

    return text


def _capitalize_organism_prefix(text: str) -> str:
    """
    Capitalizes the organism prefix of '<organism>-LanM' style names
    when the prefix is in our known set.
    Example: 'mex-LanM' -> 'Mex-LanM'.
    @param text: The variant string.
    return : The string with the organism prefix capitalized if it
             matched a known organism, otherwise unchanged.
    """
    parts = text.split("-", 1)
    if len(parts) != 2:
        return text

    prefix, rest = parts
    if prefix.lower() in _KNOWN_ORGANISM_PREFIXES:
        return f"{prefix.capitalize()}-{rest}"

    return text
