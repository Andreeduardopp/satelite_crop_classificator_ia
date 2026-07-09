# Best crop-classifier models (5 / 6 / 7 cultures) — v10

Flagship XGBoost dense models, one per crop count, **promoted 2026-07-09** (the v10
recent-years generation; supersedes v8). Each folder is a full copy of its training run
(model + selected features + calibrated abstain policy + metrics + evals + plots).

All three train on **`features_v10_train`** — a balanced, **2025/2026-only** pool
(16,623 fields: 2,500/crop, AVEIA 1,623; zero 2024 rows; composition + kept ids in
`src/data/features_v10_train/build_manifest.json`, built by `scripts/build_v10_train.py`).
Design and full results: [`../PLAN_V10_2025_2026.md`](../PLAN_V10_2025_2026.md).

> ⚠️ **A model is bound to its `selected_features.json`.** At inference, extract features with
> **`phenology_feature_pipeline_v7.py`** (29-dekad extended grid), anchored at the field's
> planting date, then reindex to the model's own `selected_features.json`.

## The three models

| Folder | Cultures | bench2026 (gate) | Held-out v8 | Source run |
|---|---|--:|--:|---|
| `5_culturas_no_aveia/` | ARROZ, FEIJAO, MILHO, SOJA, TRIGO | **0.940** (n=200) | 0.981 (n=973) | `20260709_075941_dense_5crop_v10_ssw1` |
| `6_culturas_no_aveia/` | + CAFE | **0.956** (n=250) | 0.977 (n=1173) | `20260709_091712_dense_6crop_v10_ssw1` |
| `7_culturas_cafe/` | + AVEIA (all 7) | **0.956** (n=250) | 0.955 (n=1373) | `20260709_113112_dense_7crop_v10_ssw1` |

All trained by `train_xgboost_v7.py` (weighted 5-fold OOF CV, class balance, **safrinha
up-weight 1×** — real safrinha data replaced the old 6× crutch, global abstain gate) on the same
DB, differing only by `--exclude-crops`. v8 promoted models scored **0.850/0.856/0.864** on the
same bench2026 — the v10 gain is +9–10 pts where it counts: fields the models never saw, from
the season production will actually serve.

## Per-class recall

| Culture | bench2026 (5/6/7-crop) | held-out v8 (5/6/7-crop) |
|---|---|---|
| ARROZ | 0.94 / 0.94 / 0.94 | 0.99 / 0.99 / 0.99 |
| FEIJAO | 0.96 / 0.96 / 0.96 | 0.99 / 0.99 / 0.99 |
| MILHO | 0.92 / 0.94 / 0.94 | 0.94 / 0.95 / 0.95 |
| SOJA | 0.94 / 0.94 / 0.94 | 0.98 / 0.98 / 0.98 |
| CAFE | — / **1.00** / **1.00** | — / 0.95 / 0.95 |
| TRIGO | *December* | 1.00 / 1.00 / 0.94 |
| AVEIA | *December* | — / — / 0.88 |

Residual bench2026 errors: safrinha SOJA↔MILHO (3+3 swaps) and 3 ARROZ→SOJA. No model ever
hallucinated a winter crop (TRIGO/AVEIA) on the summer benchmark fields. 7-crop AVEIA↔TRIGO
on held-out: 35 swaps (v8: 38) — unchanged until the December two-winter batch.

## ⚠️ Serving requirements (mandatory)

Two production-blocking lessons are baked into these models — the serving side MUST implement
both before `predict_proba`:

1. **inf→NaN guard.** Real fields can produce `±inf` in SAR columns (zero backscatter → −inf dB:
   radar shadow / standing water; e.g. `CAFE_520664544-1` reproduces it on every extraction).
   XGBoost **aborts** on inf input — one such field takes down the request. NaN, by contrast,
   is handled as missing (it's how the models were trained, via the same guard in
   `train_xgboost_v7._load`).
2. **Calibrated abstain threshold.** Each folder's `abstain_policy.json` now ships the
   bench2026-calibrated `threshold` (**0.80** for 5-crop, **0.90** for 6/7-crop — see the
   `calibration` block inside). The old OOF-derived 0.30 is a no-op out of distribution: in the
   original field test it abstained on **zero** of 39 errors.

```python
import numpy as np, xgboost as xgb

X = features_row.reindex(columns=selected_features)      # missing cols -> NaN, exact order
X = X.replace([np.inf, -np.inf], np.nan)                 # (1) inf guard — REQUIRED
model = xgb.XGBClassifier(); model.load_model("xgboost_crop_classifier.json")
proba = model.predict_proba(X)[0]
if proba.max() >= policy["threshold"]:                   # (2) calibrated gate — REQUIRED
    return classes[proba.argmax()]                       # classes: metrics.json -> "classes"
return "NAO_CLASSIFICAVEL"
```

Operational effect at these thresholds (measured on bench2026): 5-crop covers 96% of requests
at 0.953 covered accuracy; 6/7-crop cover ~93% at ~0.978. Also log per request: planting date,
`dekads_covered`/`fine_covered`, lat/lon and the full probability vector (needed for the next
recalibration).

## The benchmark (promotion gate)

**`kml_test_2026`** — 50/crop from the most recent completed season, frozen 2026-07-09
(`src/data/kml_test_2026/`, manifest with field_ids; extracted at `src/data/features_test_2026/`).
SOJA/MILHO/FEIJAO planted Jan–Mar 2026; ARROZ Oct 2025–Feb 2026 (2025/26 rice season); CAFE
Aug–Oct 2025 (270-day cycle). **TRIGO/AVEIA are sampled+frozen but mid-season — extracted after
2026-12-01.** Run: `python src/model/eval_bench2026.py <run_dir>`. Bar: ≥0.95 covered accuracy
at ≥0.85 coverage. Every training sampler must exclude this manifest's field_ids.

The old 2024-heavy `kml_test_sample_250` benchmark is retired (extracted DB, script, and the
superseded v9 sweep runs deleted 2026-07-09; the zip is kept only for sampler exclusions —
that story, including the −14 pt year-shift discovery, is in
`../FIELD_TEST_5CROP_ANALYSIS_AND_PLAN.md` and `../PLAN_V10_2025_2026.md` §5).

## Files in each folder

| File | What it is |
|---|---|
| `xgboost_crop_classifier.json` | the trained XGBoost model |
| `selected_features.json` | ordered feature list the model expects |
| `abstain_policy.json` | **bench2026-calibrated** threshold (+ OOF sweep + calibration block) |
| `metrics.json` | training metrics, class order, config, OOF season diagnostic |
| `test_eval/test_metrics.json` | bench2026 result (the official gate eval) |
| `test_eval_v8heldout/test_metrics.json` | regression eval on the old v8 held-out set |
| `bench2026_eval/bench_metrics.json` | bench2026 with confusion matrix + abstain sweep |
| `best_params.json` | XGBoost hyperparameters used |
| `confusion_matrix.png`, `feature_importance.png`, `abstain_curve.png` | training diagnostics |

## Caveats

- **TRIGO/AVEIA are single-winter (2025) models** and untested on 2026 winter until the December
  batch lands. Don't promise wheat/oats accuracy for the 2026 winter season yet; the abstain gate
  is the net. December also brings the v11 retrain (two winters + full 350-field benchmark).
- **Safrinha SOJA — resolved.** 1,176 real safrinha fields in training (up-weight relaxed to 1×);
  bench2026 safrinha SOJA recall 0.94, held-out 1.00.
- **Regional** — still Sul-dominated (CAFE: Sudeste). Out-of-region remains out-of-distribution;
  expansion from `culturas/` is the standing roadmap item.
- **CAFE's bench number is same-season** (train and test both plantio Aug–Oct 2025) — read its
  1.00 as in-distribution excellence, not year-generalization (perennial crop; acceptable).
- **2024 fields are out of scope by design** (recent-years decision, 2026-07-09): v10 keeps
  bench250-2024 TRIGO at ~0.74 recall. Production traffic is 2026/27-season, where bench2026 is
  the representative test.
