# Project summary — satellite crop classification in Brazil

> **Status (2026-07-08):** The promoted models (trained on `features_v8`, held-out 0.983 acc for
> 5-crop) scored **84.3%** in the first production field test — the gap traced to a planting-year
> shift (test fields were 65% 2024, training is 99.9% 2025; see
> `FIELD_TEST_5CROP_ANALYSIS_AND_PLAN.md`). A new training set (`features_v9`, sampled from the
> SICOR rural-credit KML registry) is **fully extracted: 16,553 clean rows** — safrinha SOJA
> supply 26×, scarce crops doubled, first 2026-season and 2024-CAFE data. Merged training pool
> (v8+v9) = 22,727 fields. Next: merge + retrain with the safrinha up-weight swept down, evaluated
> against the held-out test *and* the 250-KML production benchmark.

## Objective

Classify **which crop is planted in a given field**, in Brazil, using only satellite data
(Sentinel-1/2) and a KML polygon + planting date — no site visit, no farmer-reported label at
inference time. Today the classifier covers **7 crops**: ARROZ, AVEIA, CAFE, FEIJAO, MILHO, SOJA,
TRIGO, and ships as three variants (5/6/7-crop) trading crop coverage for accuracy (see
`best_models/README.md`). The end goal is a model reliable enough across Brazil's growing regions
and crop-years to run in production, with a built-in abstain gate for out-of-distribution fields
rather than a confident wrong guess.

## How the pipeline works

```
KML polygon + planting date
        │
        ▼
Sentinel Hub Statistical API  (one request per signal, per field)
        │   server-side cloud masking + zonal stats (mean/p10/p90)
        ▼
Hybrid time grid, anchored at planting date:
  • dekadal (P10D), planting−15d .. +275d  (29 bins — covers CAFE's ~270d cycle)
  • fine    (P5D),  flowering → maturity   (18 bins — the AVEIA/TRIGO senescence window)
        │
        ▼
features.db (SQLite)  — one row per field, ~1,485 columns
        │
        ▼
XGBoost training (train_xgboost_v7.py) → xgboost_crop_classifier.json + abstain_policy.json
```

`phenology_feature_pipeline_v7.py` issues these requests with field-level concurrency behind a
shared rate limiter, is resume-safe (crash-safe, skips already-extracted `field_id`s), and
auto-adapts pixel resolution to each field's polygon size.

## What data we retrieve

Per field, from Sentinel Hub:
- **8 optical indices** (Sentinel-2 L2A): NDVI, NDWI, EVI, NDRE, CIRE, MTCI, PSRI, NDMI — covering
  greenness, water content, red-edge chlorophyll, and senescence/dryness.
- **4 SAR channels** (Sentinel-1): VV, VH, CR (cross-ratio), RVI — canopy structure/moisture, useful
  under cloud cover and for crops that look similar optically (e.g. AVEIA vs TRIGO).
- For each signal/time-bin: **mean, p10, p90** (server-side, cloud-masked).
- Bookkeeping: `field_id`, `crop_label`, `planting_date`, `area_hectares`, `latitude`, `longitude`,
  plus data-quality counters (`dekads_covered`, `fine_covered`, `interpolated`).

Source labels/polygons come from KML libraries of Brazilian field boundaries with crop +
planting/harvest dates: `culturas/` (the original library) and, since 2026-07-07, direct exports
from the **SICOR rural-credit registry** (`extract_kml_sicor.py` → `kml_sicor_2025/` 242k +
`kml_sicor_2026/` 66k files — same registry/ID space as `culturas/`). See
`best_models/datasets/DATASET_ANALYSIS.md` for exactly which subset (region, size, count, year)
has been extracted into each training set, and `SICOR_DATA_PLAN.md` for the sampling strategy.

## How we build the model

- **Algorithm:** XGBoost multiclass (`multi:softprob`), one classifier per crop-count variant
  (5/6/7 crops), same features and pipeline, differing only by which crops are excluded at load.
- **Feature selection:** a quick XGBoost pass keeps the top ~60% of columns by gain.
- **Class balance + targeted up-weighting:** inverse-frequency sample weights correct crop
  imbalance; second-season ("safrinha") SOJA is additionally up-weighted since it's the
  historically hardest sub-population (see `SECOND_SEASON_SOJA_RESULTS.md`).
- **Validation:** weighted 5-fold stratified out-of-fold CV (weights applied to train folds only,
  scoring unweighted so rare classes count), plus a **season-stratified diagnostic** that checks
  SOJA↔MILHO confusion and safrinha recall specifically — global accuracy alone hides that failure
  mode.
- **Abstain gate:** from OOF probabilities, sweep a max-softmax-probability threshold and pick the
  lowest one that keeps covered accuracy ≥ target; below threshold the model returns
  `NAO_CLASSIFICAVEL` instead of guessing.
- **Held-out test:** a fully separate field set (`features_v8_test.db`) never seen during training
  or feature selection, evaluated as the final reported accuracy/F1.
- **Production benchmark:** the 250-KML field-test set (`best_models/kml_test_sample_250.zip`,
  mostly 2024 plantings) is excluded from all training and re-run on every candidate model — it
  measures the out-of-year condition that in-year held-out tests miss.

Full detail: `STATUS_AND_ROADMAP.md` (architecture + roadmap), `best_models/README.md` (model
results), `best_models/datasets/DATASET_ANALYSIS.md` (data composition).
