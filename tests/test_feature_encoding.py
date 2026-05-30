"""Tests for feature encoding utilities (Week 2 Block 2.2c)."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from agentic_ai.features.encoding import (
    align_columns_to_schema,
    build_feature_dataframe,
    encode_categorical_features,
    fit_encoder,
    transform_features,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_canonical_lanm_features() -> dict:
    """
    Builds a feature dict that mimics Mex-LanM's output from
    compute_ef_hand_features: 4 motifs, all positions populated.
    """
    return {
        "ef1_motif_pos0": "P", "ef1_motif_pos2": "K", "ef1_motif_pos5": "T",
        "ef2_motif_pos0": "P", "ef2_motif_pos2": "K", "ef2_motif_pos5": "T",
        "ef3_motif_pos0": "P", "ef3_motif_pos2": "N", "ef3_motif_pos5": "T",
        "ef4_motif_pos0": "P", "ef4_motif_pos2": "N", "ef4_motif_pos5": "T",
        "ef_hand_count": 4,
        "ef_hand_mean_spacing": 24.333,
        "ef_hand_spacing_stdev": 0.471,
        "ef_hand_span_fraction": 0.646,
        "has_four_ef_hands": 1,
    }


def _make_zero_motif_features() -> dict:
    """
    Builds a feature dict mimicking a sequence with zero EF-hand
    motifs: all per-motif features None, aggregates undefined.
    """
    return {
        "ef1_motif_pos0": None, "ef1_motif_pos2": None, "ef1_motif_pos5": None,
        "ef2_motif_pos0": None, "ef2_motif_pos2": None, "ef2_motif_pos5": None,
        "ef3_motif_pos0": None, "ef3_motif_pos2": None, "ef3_motif_pos5": None,
        "ef4_motif_pos0": None, "ef4_motif_pos2": None, "ef4_motif_pos5": None,
        "ef_hand_count": 0,
        "ef_hand_mean_spacing": None,
        "ef_hand_spacing_stdev": None,
        "ef_hand_span_fraction": None,
        "has_four_ef_hands": 0,
    }


# ---------------------------------------------------------------------------
# build_feature_dataframe
# ---------------------------------------------------------------------------

def test_build_returns_empty_dataframe_for_empty_input():
    """
    Verifies that empty/None input returns an empty DataFrame rather
    than crashing.
    """
    assert build_feature_dataframe([]).empty
    assert build_feature_dataframe(None).empty


def test_build_returns_one_row_per_input_dict():
    """
    Verifies the basic shape contract: N input dicts -> N rows.
    """
    df = build_feature_dataframe([
        _make_canonical_lanm_features(),
        _make_zero_motif_features(),
    ])
    assert len(df) == 2


def test_build_preserves_all_17_feature_columns():
    """
    Verifies that the raw DataFrame has the full feature schema (12
    categorical + 5 numeric = 17 columns). Pins the column contract.
    """
    df = build_feature_dataframe([_make_canonical_lanm_features()])
    assert len(df.columns) == 17


def test_build_preserves_categorical_strings_unchanged():
    """
    Verifies that the build step does NOT encode or transform values;
    that's encode_categorical_features' job. Categorical columns
    should still be strings.
    """
    df = build_feature_dataframe([_make_canonical_lanm_features()])
    assert df.iloc[0]["ef1_motif_pos0"] == "P"
    assert df.iloc[0]["ef3_motif_pos2"] == "N"


def test_build_converts_none_to_nan_via_pandas():
    """
    Verifies that None values become NaN in the DataFrame (pandas
    default behavior). encode_categorical_features handles the NaN ->
    MISSING substitution later.
    """
    df = build_feature_dataframe([_make_zero_motif_features()])
    # pandas converts None to NaN in numeric and object columns
    assert pd.isna(df.iloc[0]["ef_hand_mean_spacing"])


# ---------------------------------------------------------------------------
# encode_categorical_features — output shape and column contract
# ---------------------------------------------------------------------------

def test_encode_raises_on_empty_input():
    """
    Verifies that empty/None input raises rather than returning an
    empty DataFrame. Encoding empty data is almost always a bug.
    """
    with pytest.raises(ValueError, match="empty"):
        encode_categorical_features(df=pd.DataFrame())
    with pytest.raises(ValueError, match="empty"):
        encode_categorical_features(df=None)


def test_encode_preserves_row_count():
    """
    Verifies that encoding does not drop or duplicate rows.
    """
    df = build_feature_dataframe([
        _make_canonical_lanm_features(),
        _make_zero_motif_features(),
    ])
    encoded_df, _ = encode_categorical_features(df=df)
    assert len(encoded_df) == 2


def test_encode_returns_fitted_encoder():
    """
    Verifies that the second return value is a OneHotEncoder instance
    that has been fit (i.e. has the categories_ attribute populated).
    """
    df = build_feature_dataframe([_make_canonical_lanm_features()])
    _, encoder = encode_categorical_features(df=df)

    assert isinstance(encoder, OneHotEncoder)
    assert hasattr(encoder, "categories_")
    assert len(encoder.categories_) > 0


# ---------------------------------------------------------------------------
# encode_categorical_features — MISSING handling
# ---------------------------------------------------------------------------

def test_encode_creates_missing_columns_for_absent_motifs():
    """
    Verifies that None values in categorical columns produce explicit
    MISSING-suffixed columns rather than being dropped or NaN-filled.
    The 12 per-motif positions each get a MISSING indicator.
    """
    df = build_feature_dataframe([_make_zero_motif_features()])
    encoded_df, _ = encode_categorical_features(df=df)

    missing_cols = [c for c in encoded_df.columns if c.endswith("_MISSING")]
    assert len(missing_cols) == 12


def test_encode_missing_indicator_is_set_for_absent_motifs():
    """
    Verifies that for a zero-motif row, every MISSING indicator
    column has value 1.0 (not 0.0).
    """
    df = build_feature_dataframe([_make_zero_motif_features()])
    encoded_df, _ = encode_categorical_features(df=df)

    for col in encoded_df.columns:
        if col.endswith("_MISSING"):
            assert encoded_df.iloc[0][col] == 1.0


def test_encode_residue_column_set_for_canonical_motifs():
    """
    Verifies that for a Mex-LanM-like row, the correct residue
    one-hot is active (e.g. ef1_motif_pos0_P = 1.0) and no MISSING
    indicators are active.
    """
    df = build_feature_dataframe([_make_canonical_lanm_features()])
    encoded_df, _ = encode_categorical_features(df=df)

    # ef1_motif_pos0 = P, so the P column should be 1.0
    assert encoded_df.iloc[0]["ef1_motif_pos0_P"] == 1.0
    # No MISSING indicators should fire
    missing_active = [
        c for c in encoded_df.columns
        if c.endswith("_MISSING") and encoded_df.iloc[0][c] == 1.0
    ]
    assert missing_active == []


# ---------------------------------------------------------------------------
# encode_categorical_features — numeric passthrough
# ---------------------------------------------------------------------------

def test_encode_preserves_numeric_aggregate_columns():
    """
    Verifies that the 5 aggregate numeric columns pass through with
    their original names and values, not transformed.
    """
    df = build_feature_dataframe([_make_canonical_lanm_features()])
    encoded_df, _ = encode_categorical_features(df=df)

    for col in ["ef_hand_count", "ef_hand_mean_spacing",
                "ef_hand_spacing_stdev", "ef_hand_span_fraction",
                "has_four_ef_hands"]:
        assert col in encoded_df.columns


def test_encode_preserves_nan_in_numeric_aggregates():
    """
    Verifies that NaN values in numeric aggregates remain NaN through
    encoding. XGBoost handles NaN natively at split time, so we
    should NOT replace it with a sentinel.
    """
    df = build_feature_dataframe([_make_zero_motif_features()])
    encoded_df, _ = encode_categorical_features(df=df)

    assert pd.isna(encoded_df.iloc[0]["ef_hand_mean_spacing"])
    assert pd.isna(encoded_df.iloc[0]["ef_hand_spacing_stdev"])
    assert pd.isna(encoded_df.iloc[0]["ef_hand_span_fraction"])


def test_encode_preserves_numeric_values_for_canonical_input():
    """
    Verifies that numeric aggregates retain their original float/int
    values through encoding.
    """
    df = build_feature_dataframe([_make_canonical_lanm_features()])
    encoded_df, _ = encode_categorical_features(df=df)

    assert encoded_df.iloc[0]["ef_hand_count"] == 4
    assert encoded_df.iloc[0]["has_four_ef_hands"] == 1
    assert abs(encoded_df.iloc[0]["ef_hand_mean_spacing"] - 24.333) < 0.01


# ---------------------------------------------------------------------------
# encode_categorical_features — using a pre-fitted encoder
# ---------------------------------------------------------------------------

def test_encode_with_prefit_encoder_uses_it_instead_of_refitting():
    """
    Verifies that when a fitted_encoder is passed, the function uses
    it for transform rather than fitting a new one. Critical for
    Week 4 inference: the train-time encoder must drive test-time
    encoding.
    """
    train_df = build_feature_dataframe([_make_canonical_lanm_features()])
    _, encoder = encode_categorical_features(df=train_df)

    test_df = build_feature_dataframe([_make_zero_motif_features()])
    encoded_test_df, returned_encoder = encode_categorical_features(
        df=test_df, fitted_encoder=encoder,
    )

    # The returned encoder should be the same object
    assert returned_encoder is encoder


def test_encode_with_prefit_encoder_handles_unknown_residues():
    """
    Verifies that handle_unknown='ignore' on the encoder gracefully
    handles a test-time residue not seen during training. The unknown
    residue should not create a new column or crash.
    """
    train_df = build_feature_dataframe([_make_canonical_lanm_features()])
    _, encoder = encode_categorical_features(df=train_df)

    # Test sample has 'Y' at a position where only 'P' was seen
    unknown_features = _make_canonical_lanm_features()
    unknown_features["ef1_motif_pos0"] = "Y"  # Tyrosine, not in training
    test_df = build_feature_dataframe([unknown_features])

    encoded, _ = encode_categorical_features(df=test_df, fitted_encoder=encoder)
    # Should not crash; ef1_motif_pos0_P should be 0 (no other indicator set)
    assert encoded.iloc[0]["ef1_motif_pos0_P"] == 0.0


# ---------------------------------------------------------------------------
# fit_encoder + transform_features
# ---------------------------------------------------------------------------

def test_fit_encoder_returns_a_fitted_encoder():
    """
    Verifies the convenience wrapper returns a fit encoder.
    """
    df = build_feature_dataframe([_make_canonical_lanm_features()])
    encoder = fit_encoder(df=df)

    assert isinstance(encoder, OneHotEncoder)
    assert hasattr(encoder, "categories_")


def test_transform_features_requires_encoder():
    """
    Verifies that calling transform_features without an encoder raises
    a clear error rather than silently producing wrong results.
    """
    df = build_feature_dataframe([_make_canonical_lanm_features()])
    with pytest.raises(ValueError, match="fitted_encoder"):
        transform_features(df=df, fitted_encoder=None)


def test_transform_features_uses_provided_encoder():
    """
    Verifies that transform_features applies the encoder consistently.
    """
    train_df = build_feature_dataframe([_make_canonical_lanm_features()])
    encoder = fit_encoder(df=train_df)

    test_df = build_feature_dataframe([_make_zero_motif_features()])
    encoded_test = transform_features(df=test_df, fitted_encoder=encoder)

    # Should have the same columns as train encoding
    train_encoded, _ = encode_categorical_features(df=train_df)
    assert list(encoded_test.columns) == list(train_encoded.columns)


# ---------------------------------------------------------------------------
# align_columns_to_schema
# ---------------------------------------------------------------------------

def test_align_columns_raises_on_missing_args():
    """
    Verifies that both df and schema_columns are required.
    """
    with pytest.raises(ValueError, match="schema_columns"):
        align_columns_to_schema(df=pd.DataFrame(), schema_columns=None)
    with pytest.raises(ValueError, match="schema_columns"):
        align_columns_to_schema(df=None, schema_columns=["a", "b"])


def test_align_columns_fills_missing_columns_with_zero():
    """
    Verifies that columns present in the schema but missing from df
    are added with all-zero values. This handles the case where
    test-time data lacks some residues seen during training.
    """
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    schema = ["a", "b", "c", "d"]

    aligned = align_columns_to_schema(df=df, schema_columns=schema)

    assert list(aligned.columns) == schema
    assert aligned.iloc[0]["c"] == 0
    assert aligned.iloc[0]["d"] == 0


def test_align_columns_drops_extra_columns():
    """
    Verifies that columns in df but not in the schema are dropped.
    This handles the case where test-time data has a new residue
    not seen at training time.
    """
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "extra": [99.0]})
    schema = ["a", "b"]

    aligned = align_columns_to_schema(df=df, schema_columns=schema)

    assert list(aligned.columns) == schema
    assert "extra" not in aligned.columns


def test_align_columns_preserves_column_order():
    """
    Verifies that the output column order matches schema_columns
    exactly. XGBoost requires consistent column ordering between
    training and inference.
    """
    df = pd.DataFrame({"c": [1.0], "a": [2.0], "b": [3.0]})
    schema = ["a", "b", "c"]

    aligned = align_columns_to_schema(df=df, schema_columns=schema)

    assert list(aligned.columns) == ["a", "b", "c"]
