"""Evaluation metrics for the REE selectivity predictor (Week 3 Block 3.1).

Implements five metrics aligned with the project's validation policy:

  Headline metric:
    macro_per_element_spearman  - Per-REE rank correlation across
                                  test-set variants, averaged across
                                  REEs. Answers: "Can the model rank
                                  candidate proteins for a target REE?"

  Supporting metrics:
    macro_per_variant_spearman  - Per-variant rank correlation across
                                  REEs, averaged across variants.
                                  Answers: "Can the model recover a
                                  protein's selectivity profile?"
    rmse, mae, r2               - Row-level regression metrics.

All functions accept pandas DataFrames with required columns
'variant_id', 'target_element', 'value' (true), and 'predicted'.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(df: pd.DataFrame) -> float:
    """
    Row-level root-mean-square error between true and predicted values.
    @param df: DataFrame with 'value' and 'predicted' columns.
    return : float, RMSE in the same units as the target.
    """
    return float(np.sqrt(mean_squared_error(df["value"], df["predicted"])))


def mae(df: pd.DataFrame) -> float:
    """
    Row-level mean absolute error between true and predicted values.
    @param df: DataFrame with 'value' and 'predicted' columns.
    return : float, MAE in the same units as the target.
    """
    return float(mean_absolute_error(df["value"], df["predicted"]))


def r2(df: pd.DataFrame) -> float:
    """
    Row-level coefficient of determination.
    @param df: DataFrame with 'value' and 'predicted' columns.
    return : float, R^2 (1.0 = perfect, 0.0 = mean baseline, negative
             = worse than mean).
    """
    return float(r2_score(df["value"], df["predicted"]))


def macro_per_element_spearman(df: pd.DataFrame) -> Dict:
    """
    Headline metric. For each REE in the test set, rank variants by
    observed and predicted score and compute Spearman correlation;
    return the unweighted mean across REEs.

    Elements with fewer than 2 variants in the test set are skipped
    from the average (Spearman is undefined for n<2). The per-element
    breakdown is returned alongside the macro value for inspection.
    @param df: DataFrame with 'variant_id', 'target_element', 'value',
               'predicted' columns.
    return : Dict with keys 'macro' (float), 'per_element' (dict
             mapping element_name to float), and 'skipped_elements'
             (list of elements skipped for n<2).
    """
    per_element = {}
    skipped = []

    for element, group in df.groupby("target_element"):
        if len(group) < 2:
            skipped.append(element)
            continue
        rho, _ = spearmanr(group["value"], group["predicted"])
        # NaN can occur when one side has zero variance
        if np.isnan(rho):
            skipped.append(element)
            continue
        per_element[element] = float(rho)

    if not per_element:
        return {"macro": float("nan"), "per_element": {}, "skipped_elements": skipped}

    macro = float(np.mean(list(per_element.values())))
    return {
        "macro": macro,
        "per_element": per_element,
        "skipped_elements": skipped,
    }


def macro_per_variant_spearman(df: pd.DataFrame) -> Dict:
    """
    Supporting metric. For each variant in the test set, rank REEs by
    observed and predicted score and compute Spearman correlation;
    return the unweighted mean across variants.

    Variants with fewer than 2 REEs in the test set are skipped.
    @param df: DataFrame with 'variant_id', 'target_element', 'value',
               'predicted' columns.
    return : Dict with keys 'macro' (float), 'per_variant' (dict),
             and 'skipped_variants' (list).
    """
    per_variant = {}
    skipped = []

    for variant, group in df.groupby("variant_id"):
        if len(group) < 2:
            skipped.append(variant)
            continue
        rho, _ = spearmanr(group["value"], group["predicted"])
        if np.isnan(rho):
            skipped.append(variant)
            continue
        per_variant[variant] = float(rho)

    if not per_variant:
        return {"macro": float("nan"), "per_variant": {}, "skipped_variants": skipped}

    macro = float(np.mean(list(per_variant.values())))
    return {
        "macro": macro,
        "per_variant": per_variant,
        "skipped_variants": skipped,
    }


def compute_all_metrics(df: pd.DataFrame) -> Dict:
    """
    Convenience wrapper that computes all five metrics in one call.
    @param df: DataFrame with 'variant_id', 'target_element', 'value',
               'predicted' columns.
    return : Dict with keys 'rmse', 'mae', 'r2',
             'per_element_spearman', 'per_variant_spearman'.
    """
    return {
        "rmse":                  rmse(df),
        "mae":                   mae(df),
        "r2":                    r2(df),
        "per_element_spearman":  macro_per_element_spearman(df),
        "per_variant_spearman":  macro_per_variant_spearman(df),
    }
