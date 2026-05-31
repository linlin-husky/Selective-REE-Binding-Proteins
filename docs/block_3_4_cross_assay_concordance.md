# Block 3.4: Exploratory Cross-Assay Concordance Analysis

## Scope Statement

This block is an exploratory cross-assay analysis, not external
validation. Both variants compared here (`o-180` Hans-LanM and `o-621`
Mex-LanM) are in the model's training split. Running the final model
against them would test in-sample agreement rather than generalization.

This block asks:

> Do MOESM3 on-resin normalized `logD` profiles align directionally
> with published in-solution Kd-derived affinity measurements for the
> same proteins?

This is a comparison of data sources, not an evaluation of model
performance. It informs how the Streamlit application must describe its
outputs.

## Quantities Compared

| Source | Quantity | Physical regime |
|---|---|---|
| MOESM3 | normalized `logD`, scaled to the variant maximum | on-resin distribution |
| Literature | `-log10(Kd)` | in-solution affinity |

These quantities are related but not interchangeable. On-resin behavior
includes effects of immobilization and inter-REE competition in mixture.
In-solution Kd measures binding affinity for an individual REE.

## o-621 (Mex-LanM): Six-REE Comparison

Literature affinity profile from `elsevier_2001-0370_2025`:

| REE | Kd (M) | `-log10(Kd)` | Literature rank |
|---|---:|---:|---:|
| Lanthanum | 5.3e-12 | 11.28 | 1 (tied) |
| Neodymium | 5.3e-12 | 11.28 | 1 (tied) |
| Samarium | 6.6e-12 | 11.18 | 3 |
| Gadolinium | 1.0e-11 | 11.00 | 4 |
| Yttrium | 1.7e-11 | 10.77 | 5 |
| Terbium | 2.1e-11 | 10.68 | 6 |

MOESM3 normalized `logD` profile for the same six REEs:

| REE | normalized `logD` | MOESM3 rank |
|---|---:|---:|
| Samarium | 1.000 | 1 |
| Neodymium | 0.957 | 2 |
| Gadolinium | 0.736 | 3 |
| Lanthanum | 0.704 | 4 |
| Terbium | 0.619 | 5 |
| Yttrium | 0.255 | 6 |

**Exploratory Spearman correlation**: `rho = +0.6088`,
`p = 0.200`, `n = 6`.

The positive coefficient indicates moderate directional concordance,
but the small profile does not support a strong inferential claim. This
is an exploratory statistical comparison, not confirmatory evidence.

The profiles show partial agreement with important rank differences:

- Both assays place Nd and Sm near the high-preference end.
- Both assays place Y and Tb near the low-preference end.
- Literature places La among the strongest binders, whereas MOESM3
  places La fourth among the six selected REEs.
- The assays reverse the relative ordering of Y and Tb.

## o-180 (Hans-LanM): Three-REE Qualitative Comparison

Literature affinity profile from `nature_s41586-023-05945-5`:

| REE | Kd (M) | `-log10(Kd)` | Literature rank |
|---|---:|---:|---:|
| Lanthanum | 6.8e-11 | 10.17 | 1 |
| Neodymium | 9.1e-11 | 10.04 | 2 |
| Dysprosium | > 3e-7 | < 6.52 | 3 (censored) |

MOESM3 normalized `logD` profile for the same three REEs:

| REE | normalized `logD` | MOESM3 rank |
|---|---:|---:|
| Neodymium | 0.952 | 1 |
| Lanthanum | 0.617 | 2 |
| Dysprosium | 0.202 | 3 |

With `n = 3` and Dy represented by a censored bound, a formal
correlation is not appropriate.

The two assays agree on the major directional finding: Dy is the
lowest-preference REE among the three. They disagree on the La versus Nd
ordering. Literature affinity places La above Nd, whereas MOESM3
normalized `logD` places Nd above La.

## Interpretation

The two assays show partial directional concordance with important
rank-order differences. This is plausibly explained by the distinct
experimental quantities and regimes. These comparisons do not
establish assay error.

### On-Resin Versus In-Solution Conditions

MOESM3's SpyCI-LAMBS assay measures REE distribution under continuous
flow over an immobilized variant in the presence of all 15 REEs in
mixture. Literature Kd values report individual REE binding behavior in
solution. A protein may therefore display different detailed rankings
under the two regimes.

### Normalization Context

MOESM3 normalizes each variant's `logD` profile relative to that
variant's maximum-logD REE. For o-621, Sm anchors the normalized profile
at `1.0`.

Normalization compression is important when ranking Sm across
different variants, as documented in Block 3.3. However, dividing values
within a single variant by the same maximum preserves their internal
ordering. Normalization does not itself cause the within-profile
rank-order differences observed in this block.

## Implications for Streamlit Deployment

The model predicts normalized on-resin `logD`, the quantity directly
present in the training data. It does not predict in-solution binding
affinity.

Recommended language:

> Predicted normalized on-resin selectivity profile across 15 REEs.
> Higher values indicate higher predicted preference in the on-resin
> assay regime of Diep et al. (2026). These values are not equivalent to
> in-solution binding affinities.

Avoid:
- "Binding affinity for this REE"
- "Best REE for binding by this protein"
- "Predicted Kd"
- "Affinity ranking"

## Requirements for Future External Validation

Genuine out-of-distribution validation would require:

1. Variants absent from the model's training data.
2. Sequence data for those variants.
3. Scientifically interpretable experimental measurements.

Ideally, external measurements would use compatible on-resin normalized
`logD`. Other endpoints may support secondary cross-assay validation if
interpreted separately.

This combination is not currently present in the literature corpus.

## Conclusion

For Mex-LanM (`o-621`), literature affinity and MOESM3 normalized
`logD` rankings show moderate positive cross-assay concordance across
six REEs (`rho = 0.609`, `p = 0.200`, `n = 6`). For Hans-LanM
(`o-180`), the assays agree that Dy is the lowest-preference REE among
the three examined elements but disagree on La versus Nd ordering.

This is descriptive cross-assay evidence, not external validation.
Model outputs must be described as predicted normalized on-resin
selectivity profiles, not as predicted binding affinities.