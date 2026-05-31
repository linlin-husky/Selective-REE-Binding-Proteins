
"""Baseline XGBoost training pipeline (Week 3 Block 3.1).

Loads the Block 2.5 feature matrix, fits an XGBoost regressor with
sensible default hyperparameters and early stopping on a held-out
validation slice (carved from training only, grouped by variant_id
and stratified by selectivity_cluster), then evaluates on the
preexisting test split.

Persists:
  models/xgb_baseline.pkl                  - the trained model
  models/xgb_baseline_metrics.json         - test-set metrics
  models/xgb_baseline_feature_importance.csv - importance scores

The 10% early-stopping holdout is carved from training data only;
the test set is never touched until final evaluation.

CLI smoke test:
    python -m agentic_ai.models.train
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBRegressor

from agentic_ai.features.build_matrix import get_feature_columns
from agentic_ai.models.metrics import compute_all_metrics

# Baseline hyperparameters — sensible starting values, not literal
# sklearn defaults. These will be re-tuned in Block 3.2.
_BASELINE_HYPERPARAMS = {
    "n_estimators":          500,
    "learning_rate":          0.05,
    "max_depth":              6,
    "subsample":              0.8,
    "colsample_bytree":       0.8,
    "objective":              "reg:squarederror",
    "tree_method":            "hist",
    "random_state":           42,
    "eval_metric":            "rmse",
    "early_stopping_rounds":  20,
}

# Output paths
_MODEL_PATH = Path("models/xgb_baseline.pkl")
_METRICS_PATH = Path("models/xgb_baseline_metrics.json")
_IMPORTANCE_PATH = Path("models/xgb_baseline_feature_importance.csv")
_FEATURE_MATRIX_PATH = Path("data/processed/ml_ready_features.parquet")
_FEATURE_SCHEMA_PATH = Path("data/processed/ml_ready_features_schema.json")

def train_baseline(
    matrix_path: Path = None,
    early_stop_fraction: float = 0.10,
    random_state: int = 42,
) -> dict:
    """
    Trains the baseline XGBoost regressor and evaluates it on the
    held-out test set.
    @param matrix_path: Path to the Block 2.5 parquet file. Defaults
                        to data/processed/ml_ready_features.parquet.
    @param early_stop_fraction: Fraction of training data to hold out
                                for early stopping. The holdout is
                                grouped by variant_id and stratified
                                by selectivity_cluster.
    @param random_state: Seed for reproducibility.
    return : Dict with the trained model, evaluation metrics, and
             feature importance series.
    """
    if not 0 < early_stop_fraction < 0.5:
        raise ValueError(
            f"early_stop_fraction must be in (0, 0.5), got {early_stop_fraction}"
        )
    if matrix_path is None:
        matrix_path = _FEATURE_MATRIX_PATH

    df = pd.read_parquet(matrix_path)
    feature_cols = json.loads(_FEATURE_SCHEMA_PATH.read_text())
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Feature matrix is missing schema columns: {sorted(missing)}"
        )

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    fit_df, early_stop_df = _carve_early_stopping_split(
        train_df,
        early_stop_fraction=early_stop_fraction,
        random_state=random_state,
    )

    # Verify no variant overlap between any of the three splits
    _assert_no_variant_overlap(fit_df, early_stop_df, test_df)

    X_fit = fit_df[feature_cols]
    y_fit = fit_df["value"]
    X_es = early_stop_df[feature_cols]
    y_es = early_stop_df["value"]

    hyperparams = {**_BASELINE_HYPERPARAMS, "random_state": random_state}
    model = XGBRegressor(**hyperparams)
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_es, y_es)],
        verbose=False,
    )

    # Predict on the held-out test set
    X_test = test_df[feature_cols]
    test_df["predicted"] = model.predict(X_test)

    metrics = compute_all_metrics(test_df)

    importance_series = pd.Series(
        model.feature_importances_,
        index=feature_cols,
        name="importance",
    ).sort_values(ascending=False)

    return {
        "model":              model,
        "metrics":            metrics,
        "feature_importance": importance_series,
        "n_fit":              len(fit_df),
        "n_early_stop":       len(early_stop_df),
        "n_test":             len(test_df),
        "n_fit_variants":          fit_df["variant_id"].nunique(),
        "n_early_stop_variants":   early_stop_df["variant_id"].nunique(),
        "n_test_variants":         test_df["variant_id"].nunique(),
        "best_iteration":     getattr(model, "best_iteration", None),
        "hyperparameters":    hyperparams,
    }


def _carve_early_stopping_split(
    train_df: pd.DataFrame,
    early_stop_fraction: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits training data into a fit set and an early-stopping holdout.
    Uses StratifiedGroupKFold to preserve variant grouping and cluster
    balance, then takes one fold as the holdout. The fold count is
    chosen so that one fold is approximately early_stop_fraction of
    the data.
    @param train_df: The full training DataFrame.
    @param early_stop_fraction: Target fraction for the holdout.
    @param random_state: Seed for reproducibility.
    return : Tuple of (fit_df, early_stop_df). Variants do not overlap
             between the two.
    """
    n_splits = max(2, int(round(1.0 / early_stop_fraction)))

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fit_idx, es_idx = next(splitter.split(
        train_df,
        y=train_df["selectivity_cluster"],
        groups=train_df["variant_id"],
    ))

    return (
        train_df.iloc[fit_idx].copy(),
        train_df.iloc[es_idx].copy(),
    )


def _assert_no_variant_overlap(*frames: pd.DataFrame) -> None:
    """
    Verifies that none of the provided DataFrames share variant_id
    values. Catches splitting bugs that would produce inflated metrics.
    @param frames: Two or more DataFrames with 'variant_id' columns.
    raises : AssertionError if any two frames share a variant_id.
    """
    variant_sets = [set(df["variant_id"]) for df in frames]
    for i in range(len(variant_sets)):
        for j in range(i + 1, len(variant_sets)):
            overlap = variant_sets[i] & variant_sets[j]
            if overlap:
                raise AssertionError(
                    f"Variant overlap detected between split {i} and "
                    f"split {j}: {len(overlap)} variants in common, "
                    f"e.g. {list(overlap)[:5]}"
                )


def persist_artifacts(result: dict) -> None:
    """
    Writes the trained model, metrics JSON, and feature importance
    CSV to the models/ directory.
    @param result: The dict returned by train_baseline.
    """
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(_MODEL_PATH, "wb") as fh:
        pickle.dump(result["model"], fh)

    # Strip non-JSON-serializable bits (numpy types) from metrics
    metrics_for_json = _make_json_safe(result["metrics"])
    metrics_for_json["n_fit"] = result["n_fit"]
    metrics_for_json["n_early_stop"] = result["n_early_stop"]
    metrics_for_json["n_test"] = result["n_test"]
    metrics_for_json["n_fit_variants"] = result["n_fit_variants"]
    metrics_for_json["n_early_stop_variants"] = result["n_early_stop_variants"]
    metrics_for_json["n_test_variants"] = result["n_test_variants"]
    metrics_for_json["best_iteration"] = result["best_iteration"]
    metrics_for_json["hyperparameters"] = result["hyperparameters"]

    with open(_METRICS_PATH, "w") as fh:
        json.dump(metrics_for_json, fh, indent=2)

    result["feature_importance"].to_csv(_IMPORTANCE_PATH, header=True)

def _make_json_safe(obj):
    """
    Recursively converts numpy scalars to Python primitives for JSON
    serialization.
    """
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main() -> int:
    """
    CLI smoke test: train, evaluate, persist, print summary.
    """
    print("Loading feature matrix and training baseline XGBoost...")
    result = train_baseline()

    metrics = result["metrics"]
    pes = metrics["per_element_spearman"]
    pvs = metrics["per_variant_spearman"]

    print()
    print(f"=== Training summary ===")
    print(f"  n_fit:           {result['n_fit']:,}")
    print(f"  n_early_stop:    {result['n_early_stop']:,}")
    print(f"  n_test:          {result['n_test']:,}")
    print(f"  best_iteration:  {result['best_iteration']}")
    print()
    print(f"=== Test-set metrics ===")
    print(f"  HEADLINE: per-element Spearman (macro): "
          f"{pes['macro']:.4f}")
    print(f"  per-variant Spearman (macro):           "
          f"{pvs['macro']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R^2:  {metrics['r2']:.4f}")
    print()
    if pes["skipped_elements"]:
        print(f"  Skipped elements (per-element Spearman): "
              f"{pes['skipped_elements']}")
    if pvs["skipped_variants"]:
        print(f"  Skipped variants (per-variant Spearman): "
              f"{len(pvs['skipped_variants'])} variants")
    print()
    print(f"=== Per-element Spearman breakdown ===")
    for element, rho in sorted(
        pes["per_element"].items(), key=lambda x: -x[1],
    ):
        print(f"  {element:<16} {rho:+.4f}")
    print()
    print(f"=== Top 15 features by importance ===")
    print(result["feature_importance"].head(15).to_string())

    persist_artifacts(result)
    print()
    print(f"=== Artifacts persisted ===")
    print(f"  {_MODEL_PATH}")
    print(f"  {_METRICS_PATH}")
    print(f"  {_IMPORTANCE_PATH}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
