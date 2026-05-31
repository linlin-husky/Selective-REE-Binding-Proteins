I have not modified any files. Here is the revised document.

# Block 3.5: Inference Contract for Week 4 Deployment

## Deployment Decision

The baseline XGBoost regressor from Block 3.1 is the deployment model.

The pre-specified headline metric was macro per-element Spearman because
the operational task is ranking proteins for a target REE.

| Model | Per-element Spearman | RMSE | R² |
|---|---:|---:|---:|
| Baseline | **0.5565** | 0.1282 | 0.8511 |
| Tuned | 0.5524 | **0.1253** | **0.8578** |

Randomized hyperparameter tuning modestly improved regression error but
did not improve the ranking objective. The baseline model is therefore
retained for deployment.

The difference is small and should not be interpreted as evidence that
the baseline is statistically superior. The decision follows the
pre-declared selection criterion.

The Block 3.2 tuned model and its audit trail remain methodology
artifacts, not deployment artifacts.

## Artifact Inventory

All paths are repository-relative.

### Required for Inference

| Path | Role |
|---|---|
| `models/xgb_baseline.json` | Trained model in native XGBoost JSON format; preferred for deployment portability |
| `data/processed/ml_ready_features_schema.json` | Ordered list of 128 model-input columns |
| `data/processed/ml_ready_features_encoder.pkl` | Fitted one-hot encoder for EF-hand motif positions; fitted on training variants only |

### Reference Only

| Path | Role |
|---|---|
| `models/xgb_baseline.pkl` | Original Block 3.1 model artifact; retained for reproducibility |
| `models/xgb_baseline_metrics.json` | Held-out test-set evaluation |
| `models/xgb_baseline_feature_importance.csv` | Block 3.1 feature-importance ranking |
| `models/xgb_tuned.json` | Block 3.2 tuned model; retained as methodology evidence |
| `models/xgb_tuning_results.csv` | Randomized-search audit trail |

## Inference API

Week 4 should implement one reusable function outside the Streamlit UI:

```python
predict_profile(sequence: str) -> pd.DataFrame
```

It should return one row per REE with at least:

```text
target_element
predicted_normalized_logD
rank
confidence_note
```

The Streamlit layer should format this structured output rather than
reimplement feature engineering.

## Inference Pipeline

Given one candidate LanM sequence, `predict_profile()` must execute:

### 1. Sequence Validation

- Accept a non-empty amino-acid sequence string.
- Trim outer whitespace and normalize to uppercase.
- Reject non-canonical amino-acid characters.
- Warn when a sequence falls outside the MOESM3 length distribution.
- Describe substantially out-of-range sequences as unsupported rather
  than biologically invalid.

The supported-range warning threshold should be derived from the MOESM3
sequence-length distribution before the UI is finalized.

### 2. Sequence Feature Engineering

Reuse the existing feature modules exactly:

- `agentic_ai/features/sequence_features.py`
  - 15 basic physicochemical descriptors
- `agentic_ai/features/ef_hand_motifs.py`
  - regex-detected motifs matching `[A-Z]D[A-Z]DG[A-Z]`
  - 12 categorical motif-position features
  - 5 aggregate EF-hand architecture features

Missing motif positions are emitted as `None` and represented as missing
data during encoding. Do not introduce a new sentinel value.

### 3. Per-REE Feature Construction

Construct 15 rows per sequence, one for each MOESM3 target:

```text
La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Y
```

Attach five physicochemical features to each row using
`agentic_ai/features/ree_features.py`:

```text
atomic_number
oxidation_state
ionic_radius_pm_cn8
charge_density_z_per_pm3
is_lanthanide
```

The lookup table supports additional elements, but deployment output is
restricted to the 15 REEs represented in the model's training target.

### 4. Encoder Application

- Load `data/processed/ml_ready_features_encoder.pkl`.
- Apply the fitted encoder used during training.
- Never fit a new encoder during inference.
- Preserve the exact feature-building logic used in Block 2.5.

### 5. Schema Validation

- Load `data/processed/ml_ready_features_schema.json`.
- Verify that all 128 expected model-input columns are present.
- Detect unexpected columns and schema drift.
- Reorder columns to match the persisted schema before prediction.
- Fail clearly when the inference feature schema is incompatible.

### 6. Model Prediction

- Load `models/xgb_baseline.json`.
- Call `model.predict()` on the 15 schema-aligned rows.
- Return predicted normalized `logD` values.
- Sort results in descending order and assign ranks.

## Regression Tests for Week 4

Before wiring the Streamlit UI:

1. Run `predict_profile()` for known MOESM3 variants.
2. Confirm that outputs match direct predictions generated from the
   persisted feature matrix.
3. Verify that JSON-model predictions match the original pickle-model
   predictions within numerical tolerance.
4. Test sequences with lowercase letters and outer whitespace.
5. Test rejected inputs containing non-canonical residues.
6. Test sequences with missing detected EF-hand motifs.
7. Test schema-drift failure behavior.

## Output Interpretation

The model predicts MOESM3 normalized on-resin `logD`. It does not
predict in-solution binding affinity.

Required display language:

> Predicted normalized on-resin selectivity profile across 15 REEs.
> Higher values indicate higher predicted preference in the on-resin
> assay regime of Diep et al. (2026). These values are not equivalent
> to in-solution binding affinities.

Avoid:
- “Binding affinity”
- “Best REE for binding by this protein”
- “Predicted Kd”
- “Affinity ranking”
- “REE binding strength”

## Confidence Notes

### Samarium

Display a prominent note:

> Sm ranking confidence is reduced. Approximately 44% of training
> orthologs reach the Sm normalized maximum, limiting discriminative
> ranking signal.

### Neodymium

Document in the About section:

> Nd also exhibits substantial target saturation. Approximately 48% of
> training orthologs have Nd normalized `logD` values greater than or
> equal to 0.95.

The Sm limitation is more severe because approximately 82% of training
orthologs have Sm values greater than or equal to 0.95.

## Model Provenance

Display in an About section:

- Model: XGBoost regressor
- Source dataset: MOESM3 from Diep et al. (2026)
- Full MOESM3 matrix: 616 variants × 15 REEs = 9,240 rows
- Model-selection training split: 492 variants × 15 REEs = 7,380 rows
- Baseline fit subset: 442 variants × 15 REEs = 6,630 rows
- Early-stopping subset: 50 variants × 15 REEs = 750 rows
- Held-out test set: 124 variants × 15 REEs = 1,860 rows
- Test per-element Spearman: 0.557
- Test per-variant Spearman: 0.938
- Test R²: 0.851
- Model commit hash: embed at deployment time

## Known Limitations

1. **LanM ortholog scope.** The model was trained on LanM orthologs from
   MOESM3. Predictions for engineered chelators, fusion proteins, point
   mutants, or substantially different proteins are unvalidated.

2. **On-resin assay regime.** The model predicts normalized on-resin
   `logD`, not in-solution binding affinity. Block 3.4 found partial
   directional concordance with important rank-order differences.

3. **Reduced Sm ranking confidence.** Sm target compression limits
   across-variant ranking performance.

4. **No prediction uncertainty.** The app provides point estimates only.
   Future work could evaluate ensemble-based or quantile-regression
   uncertainty estimates.

5. **No true external validation yet.** Genuine out-of-distribution
   validation remains a future research extension.

## Block 3.4 Context

The exploratory cross-assay analysis for Mex-LanM (`o-621`) found
moderate directional concordance between MOESM3 normalized on-resin
`logD` and literature in-solution affinity:

```text
Spearman rho = 0.609
p = 0.200
n = 6 REEs
```

For Hans-LanM (`o-180`), both assays identify Dy as the
lowest-preference REE among the three examined elements, but they differ
in La versus Nd ordering.

These results support careful output labeling. They do not constitute
external model validation.

## Future External Validation

A genuine out-of-distribution validation study would require:

1. Variants absent from the model's training data.
2. Sequence data for those variants.
3. Scientifically interpretable measurements.

Compatible on-resin normalized `logD` measurements would provide the
most direct evaluation. Other endpoints may support secondary
cross-assay analyses if interpreted separately.

## Handoff to Week 4

Week 4 inherits:
- a JSON deployment model;
- the ordered feature schema;
- the fitted categorical encoder;
- a defined reusable inference API;
- required output language;
- confidence notes and known limitations;
- regression-test requirements.

Week 4 contributes:
- implementation of `predict_profile()`;
- Streamlit sequence input and visualization;
- model-provenance and limitations displays;
- optional batch prediction support.

## Conclusion

Block 3.5 closes Week 3. The baseline model is the deployment artifact
because it retained the strongest value for the pre-specified ranking
metric. Week 4 can now implement the inference pipeline and Streamlit
interface while preserving the scientific boundaries established during
model evaluation.
```