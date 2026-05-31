"""Tests for evaluation metrics (Week 3 Block 3.1)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agentic_ai.models.metrics import (
    compute_all_metrics,
    macro_per_element_spearman,
    macro_per_variant_spearman,
    mae,
    r2,
    rmse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_perfect_predictions_df() -> pd.DataFrame:
    """A small DataFrame where predictions exactly match truth."""
    rng = np.random.default_rng(seed=42)
    rows = []
    for variant_id in [f"v{i}" for i in range(5)]:
        for element in ["Lanthanum", "Cerium", "Neodymium", "Samarium"]:
            value = float(rng.uniform(0, 1))
            rows.append({
                "variant_id":     variant_id,
                "target_element": element,
                "value":          value,
                "predicted":      value,
            })
    return pd.DataFrame(rows)


def _make_random_predictions_df() -> pd.DataFrame:
    """A small DataFrame where predictions are random (unrelated to truth)."""
    rng = np.random.default_rng(seed=42)
    rows = []
    for variant_id in [f"v{i}" for i in range(5)]:
        for element in ["Lanthanum", "Cerium", "Neodymium", "Samarium"]:
            rows.append({
                "variant_id":     variant_id,
                "target_element": element,
                "value":          float(rng.uniform(0, 1)),
                "predicted":      float(rng.uniform(0, 1)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RMSE, MAE, R²
# ---------------------------------------------------------------------------

def test_rmse_is_zero_for_perfect_predictions():
    df = _make_perfect_predictions_df()
    assert rmse(df) == 0.0


def test_mae_is_zero_for_perfect_predictions():
    df = _make_perfect_predictions_df()
    assert mae(df) == 0.0


def test_r2_is_one_for_perfect_predictions():
    df = _make_perfect_predictions_df()
    assert math.isclose(r2(df), 1.0, abs_tol=1e-9)


def test_rmse_is_positive_for_imperfect_predictions():
    df = pd.DataFrame({
        "variant_id":     ["v1", "v1"],
        "target_element": ["La", "Ce"],
        "value":          [0.5, 0.5],
        "predicted":      [0.3, 0.7],
    })
    assert rmse(df) > 0


def test_mae_matches_hand_computed_value():
    """A 4-row toy example with known absolute errors."""
    df = pd.DataFrame({
        "variant_id":     ["v1"] * 4,
        "target_element": ["La", "Ce", "Nd", "Sm"],
        "value":          [1.0, 2.0, 3.0, 4.0],
        "predicted":      [1.5, 1.5, 3.5, 3.5],
    })
    # Absolute errors: 0.5, 0.5, 0.5, 0.5 -> mean = 0.5
    assert math.isclose(mae(df), 0.5)


# ---------------------------------------------------------------------------
# Per-element Spearman
# ---------------------------------------------------------------------------

def test_per_element_spearman_is_one_for_perfect_predictions():
    df = _make_perfect_predictions_df()
    result = macro_per_element_spearman(df)
    assert math.isclose(result["macro"], 1.0, abs_tol=1e-9)


def test_per_element_spearman_breakdown_has_one_value_per_element():
    df = _make_perfect_predictions_df()
    result = macro_per_element_spearman(df)
    assert set(result["per_element"].keys()) == {
        "Lanthanum", "Cerium", "Neodymium", "Samarium",
    }


def test_per_element_spearman_skips_elements_with_fewer_than_two_variants():
    """An element appearing in only 1 row cannot have rank correlation."""
    df = pd.DataFrame({
        "variant_id":     ["v1", "v1", "v2"],
        "target_element": ["La", "Ce", "Ce"],
        "value":          [0.1, 0.2, 0.3],
        "predicted":      [0.1, 0.2, 0.3],
    })
    result = macro_per_element_spearman(df)
    assert "La" in result["skipped_elements"]
    assert "Ce" in result["per_element"]


def test_per_element_spearman_skips_elements_with_constant_values():
    """If true or predicted is constant, Spearman is undefined."""
    df = pd.DataFrame({
        "variant_id":     ["v1", "v2", "v3"],
        "target_element": ["La", "La", "La"],
        "value":          [0.5, 0.5, 0.5],  # constant
        "predicted":      [0.1, 0.2, 0.3],
    })
    result = macro_per_element_spearman(df)
    assert "La" in result["skipped_elements"]


# ---------------------------------------------------------------------------
# Per-variant Spearman
# ---------------------------------------------------------------------------

def test_per_variant_spearman_is_one_for_perfect_predictions():
    df = _make_perfect_predictions_df()
    result = macro_per_variant_spearman(df)
    assert math.isclose(result["macro"], 1.0, abs_tol=1e-9)


def test_per_variant_spearman_skips_variants_with_fewer_than_two_elements():
    df = pd.DataFrame({
        "variant_id":     ["v1", "v2", "v2"],
        "target_element": ["La", "La", "Ce"],
        "value":          [0.1, 0.2, 0.3],
        "predicted":      [0.1, 0.2, 0.3],
    })
    result = macro_per_variant_spearman(df)
    assert "v1" in result["skipped_variants"]
    assert "v2" in result["per_variant"]


# ---------------------------------------------------------------------------
# compute_all_metrics convenience wrapper
# ---------------------------------------------------------------------------

def test_compute_all_metrics_returns_all_five_keys():
    df = _make_perfect_predictions_df()
    result = compute_all_metrics(df)
    assert set(result.keys()) == {
        "rmse", "mae", "r2",
        "per_element_spearman", "per_variant_spearman",
    }


def test_compute_all_metrics_handles_random_predictions_without_crashing():
    """Random predictions should not raise; numbers should be plausible
    (RMSE > 0, Spearman near zero but not strictly)."""
    df = _make_random_predictions_df()
    result = compute_all_metrics(df)
    assert result["rmse"] > 0
    assert result["mae"] > 0
    assert abs(result["per_element_spearman"]["macro"]) < 0.8
