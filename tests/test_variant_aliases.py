"""Tests for the variant name normalizer (Week 1 Block 4.4b)."""
from __future__ import annotations

import pytest

from agentic_ai.loaders.variant_aliases import normalize_variant_name


# ---------------------------------------------------------------------------
# Rule 6: empty / whitespace / None handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_normalize_returns_none_for_empty_input(raw):
    """
    Verifies that any falsy or whitespace-only input normalizes to
    None, signaling 'no usable name' to the classifier.
    """
    assert normalize_variant_name(raw) is None


def test_normalize_strips_outer_whitespace_from_real_names():
    """
    Verifies that real names with surrounding whitespace are stripped
    but otherwise preserved.
    """
    assert normalize_variant_name("  Mex-LanM  ") == "Mex-LanM"


# ---------------------------------------------------------------------------
# Rule 1: organism prefix capitalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("mex-LanM", "Mex-LanM"),
    ("hans-LanM", "Hans-LanM"),
    ("melba-LanM", "Melba-LanM"),
])
def test_normalize_capitalizes_lowercase_organism_prefix(raw, expected):
    """
    Verifies that lowercase organism prefixes in <organism>-LanM names
    are capitalized to canonical form.
    """
    assert normalize_variant_name(raw) == expected


@pytest.mark.parametrize("name", ["Mex-LanM", "Hans-LanM", "Melba-LanM"])
def test_normalize_preserves_already_canonical_names(name):
    """
    Verifies that canonical names pass through unchanged.
    """
    assert normalize_variant_name(name) == name


def test_normalize_preserves_wt_lanm_distinction():
    """
    Verifies that 'WT-LanM' (a legitimate distinct name, not a typo)
    is preserved rather than collapsed.
    """
    assert normalize_variant_name("WT-LanM") == "WT-LanM"


def test_normalize_leaves_unknown_organism_prefixes_unchanged():
    """
    Verifies that the capitalization rule only fires for organisms in
    our known set. 'foo-LanM' should NOT become 'Foo-LanM' (we want
    novel names to surface to the classifier as-is for inspection).
    """
    assert normalize_variant_name("foo-LanM") == "foo-LanM"


# ---------------------------------------------------------------------------
# Rule 2: internal whitespace in shorthand
# ---------------------------------------------------------------------------

def test_normalize_collapses_internal_space_in_mutation_shorthand():
    """
    Verifies that '4 P2A' (typo with internal space) becomes '4P2A'.
    """
    assert normalize_variant_name("4 P2A") == "4P2A"


def test_normalize_converts_wt_space_lanm_to_canonical_form():
    """
    Verifies that 'WT LanM' becomes 'WT-LanM'.
    """
    assert normalize_variant_name("WT LanM") == "WT-LanM"


# ---------------------------------------------------------------------------
# Rule 3: bare mutation parenting
# ---------------------------------------------------------------------------

def test_normalize_parents_bare_r100k_to_hans_lanm():
    """
    Verifies that the bare 'R100K' mutation code is parented to
    'Hans-LanM(R100K)' since R100K is the canonical Hans-LanM
    monomeric variant in the literature.
    """
    assert normalize_variant_name("R100K") == "Hans-LanM(R100K)"


def test_normalize_preserves_already_parented_mutations():
    """
    Verifies that 'Hans-LanM(R100K)' (already canonical) is unchanged.
    """
    assert normalize_variant_name("Hans-LanM(R100K)") == "Hans-LanM(R100K)"


@pytest.mark.parametrize("name", ["4D9A", "4D9H", "4D9M", "4D9N", "4P2A"])
def test_normalize_preserves_4d9x_shorthand(name):
    """
    Verifies that the 4D9X EF-hand shorthand notation (LanM-specific
    mutation convention from the Elsevier MD paper) passes through
    unchanged. These don't fit the <parent>(<mutation>) pattern.
    """
    assert normalize_variant_name(name) == name


# ---------------------------------------------------------------------------
# Rule 4: ortholog ID canonicalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("o-36", "o-36"),
    ("o36", "o-36"),
    ("O-36", "o-36"),
    ("O36", "o-36"),
    ("o-621", "o-621"),
    ("O-180", "o-180"),
])
def test_normalize_canonicalizes_ortholog_id_capitalization_and_hyphen(raw, expected):
    """
    Verifies that ortholog IDs in any capitalization or hyphen variant
    collapse to the canonical lowercase-hyphenated form matching
    MOESM3.
    """
    assert normalize_variant_name(raw) == expected


def test_normalize_does_not_match_ortholog_pattern_on_engineered_names():
    """
    Verifies that the ortholog pattern is strict enough not to false-
    positive on engineered constructs that happen to start with 'O'.
    """
    assert normalize_variant_name("Ortho-construct") == "Ortho-construct"


# ---------------------------------------------------------------------------
# Rule 5: tag stripping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Mex-LanM-Cys", "Mex-LanM"),
    ("Hans-LanM-Cys", "Hans-LanM"),
    ("o-36(GSGC)", "o-36"),
    ("Mex-LanM-SpyTag", "Mex-LanM"),
    ("Hans-LanM-SpyTag3", "Hans-LanM"),
    ("Mex-LanM-His6", "Mex-LanM"),
    ("Mex-LanM-His10", "Mex-LanM"),
])
def test_normalize_strips_purification_and_conjugation_tags(raw, expected):
    """
    Verifies that purification/conjugation tag suffixes are stripped
    so tagged constructs join cleanly to their untagged equivalents.
    """
    assert normalize_variant_name(raw) == expected


def test_normalize_strips_multiple_stacked_tags():
    """
    Verifies that names with multiple stacked tags (e.g. SpyTag plus
    His10) get all tags stripped.
    """
    assert normalize_variant_name("Mex-LanM-His10-SpyTag") == "Mex-LanM"


def test_normalize_combines_tag_strip_with_capitalization():
    """
    Verifies that tag stripping composes correctly with organism
    prefix capitalization: 'mex-LanM-Cys' -> 'Mex-LanM'.
    """
    assert normalize_variant_name("mex-LanM-Cys") == "Mex-LanM"

