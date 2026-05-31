# Literature Validation Policy (Week 3 Block 3.4)

## Three-Tier Validation Framework

### Tier 1: Strict Primary Quantitative
Filter:
```python
canonical = df[
    (df["is_molar_affinity_candidate"] == True)
    & (df["value_source_type"] == "primary")
    & (df["source_paper"] != "nature_d41586-023-01739")
    & (df["target_element"] != "Calcium")
]
```
Use for: Headline rank correlation metrics.
Variants with MOESM3 sequences available: o-180.

### Tier 2: Curated Reference Profiles
Include `cited_from_earlier_work` rows where the source is internally consistent.
Use for: Richer o-621 analysis using elsevier_2001-0370_2025 compiled values.

### Tier 3: Qualitative Directional Checks
Include censored bounds, sparse records, and provenance notes.

## Locked Profiles

### o-621 (curated reference, elsevier_2001-0370_2025)
Quantitative rank check across 6 REEs:
- Lanthanum:  5.3e-12 M (affinity_score 11.28)
- Neodymium:  5.3e-12 M (affinity_score 11.28)
- Samarium:   6.6e-12 M (affinity_score 11.18)
- Gadolinium: 1.0e-11 M (affinity_score 11.00)
- Yttrium:    1.7e-11 M (affinity_score 10.77)
- Terbium:    2.1e-11 M (affinity_score 10.68)
Expected ordering: La = Nd > Sm > Gd > Y > Tb

### o-180 (primary, nature_s41586-023-05945-5)
Qualitative directional check:
- La (6.8e-11 M) > Nd (9.1e-11 M) >> Dy (>3e-7 M; censored bound)

## Known Issues for Future Curation

1. Hans-LanM(R100K) `0.5 pM` La value may be inconsistent with the
   primary paper's text (twofold weaker than wild-type Hans-LanM's
   68 pM). Manual verification needed before use.
2. wild-type S. oneidensis: `Kd` and `Kd1` are distinct measurements
   (probably two binding sites). Do not merge.
3. EF1-EF4 Europium duplicates: Possibly different experimental methods
   (ITC vs TRLFS). Review source paper before aggregation.
4. nature_d41586-023-01739 commentary article: Retain in corpus for
   provenance, exclude from numerics.
5. o-180 Dy: Source reports `Kd_app > 0.3 µM` (censored lower bound).
   Currently stored as 3e-7 M exact. Annotate before quantitative use.

## Scope Reality

The model can only predict for variants present in MOESM3
(sequence available). That limits quantitative validation to o-621
and o-180. Other Tier 1 rows (EF peptides, RF series, S. oneidensis
wild-type) require sequence enrichment as a future task before
model evaluation.
