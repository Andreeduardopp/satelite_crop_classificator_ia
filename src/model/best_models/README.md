# Best crop-classifier models (5 / 6 / 7 cultures)

Flagship XGBoost dense models, one per crop count. Each folder is a full copy of its training
run (model + selected features + abstain policy + metrics + plots). All three were re-extracted
and retrained on **`features_v8`** — the resolution-bug-fixed data (2026-07-07).

> ⚠️ **A model is bound to its `selected_features.json`.** At inference, extract features with
> **`phenology_feature_pipeline_v7.py`** (all three models now use the same 29-dekad extended grid),
> anchored at the field's planting date, then reindex to the model's own `selected_features.json`.

## The three models

| Folder | Cultures | Held-out test | Source run |
|---|---|---|---|
| `5_culturas_no_aveia/` | ARROZ, FEIJAO, MILHO, SOJA, TRIGO | **0.983** acc / 0.982 F1 (n=973) | `20260707_014558_dense_5crop_v8_no_aveia_no_cafe` |
| `6_culturas_no_aveia/` | + CAFE | **0.979** acc / 0.978 F1 (n=1173) | `20260707_014951_dense_6crop_v8_no_aveia` |
| `7_culturas_cafe/` | + AVEIA (all 7) | **0.952** acc / 0.952 F1 (n=1373) | `20260707_013253_dense_7crop_v8_resfix` |

All trained by `train_xgboost_v7.py` (weighted 5-fold OOF CV, class balance, safrinha-SOJA up-weight,
global abstain gate, season-stratified diagnostic) on the same DBs, differing only by
`--exclude-crops` (5-crop excludes AVEIA + CAFE; 6-crop excludes AVEIA; 7-crop excludes none).

**The ladder is deliberate:** AVEIA↔TRIGO is the whole residual error budget, so dropping AVEIA lets
the 5-/6-crop models reach ~0.98 (and TRIGO F1 1.00). Deployments that don't need oats should use them.

> 📂 **Full status, dataset provenance/variability, and the upgrade plan are in
> [`../STATUS_AND_ROADMAP.md`](../STATUS_AND_ROADMAP.md).** The exact training/test data is snapshotted
> byte-for-byte in [`datasets/`](datasets/) (`features_v8.db`, `features_v8_test.db`) with checksums +
> per-crop/date/geo stats in [`datasets/MANIFEST.json`](datasets/MANIFEST.json).

## The pipeline

`phenology_feature_pipeline_v7.py` issues one Sentinel-Hub Statistical-API request per source and
aggregates the crop calendar server-side on a hybrid grid: **dekadal (P10D)** over `planting−15 .. +275 d`
(`N_DEKADS = 29`, covers CAFE's ~270-day cycle) + **fine (P5D)** over flowering→maturity (the
AVEIA/TRIGO senescence window). Schema ≈ 1485 columns. `resx/resy` are in **degrees** (the 2026-07-07
fix — the old metres value collapsed every field to ~1 pixel; see STATUS_AND_ROADMAP Part I §2).

## Per-class recall (held-out test)

| Culture | 5-crop | 6-crop | 7-crop |
|---|---|---|---|
| ARROZ | 0.98 | 0.98 | 0.98 |
| FEIJAO | 0.99 | 0.99 | 1.00 |
| MILHO | 0.98 | 0.96 | 0.96 |
| SOJA | 0.96 | 0.96 | 0.96 |
| TRIGO | **1.00** | **1.00** | 0.89 |
| CAFE | — | 0.98 | 0.97 |
| AVEIA | — | — | 0.92 |

TRIGO is perfect in the 5-/6-crop models because AVEIA (its one hard confuser) is absent; adding
AVEIA back in the 7-crop model restores the AVEIA↔TRIGO pair (0.90/0.90 F1, 38 swaps — down from 52
before the resolution fix). CAFE separates cleanly (F1 0.97).

## Using a model (inference sketch)

1. Extract features with `phenology_feature_pipeline_v7.py`, anchored at the field's planting date.
2. Reindex the feature row to the model's `selected_features.json` (add any missing column as NaN, in
   that exact order).
3. `model = xgb.XGBClassifier(); model.load_model("xgboost_crop_classifier.json")`, then
   `proba = model.predict_proba(X)`.
4. Apply `abstain_policy.json`: predict the arg-max crop only if its probability ≥ the policy
   `threshold`, else return `NAO_CLASSIFICAVEL`. Class order is in `metrics.json` → `classes`.

`evaluate()` in `train_xgboost_v7.py` is a complete worked example of steps 2–4.

## Files in each folder

| File | What it is |
|---|---|
| `xgboost_crop_classifier.json` | the trained XGBoost model |
| `selected_features.json` | ordered feature list the model expects |
| `abstain_policy.json` | probability threshold + coverage/accuracy sweep for the abstain gate |
| `metrics.json` | full training metrics, class order, config, OOF season diagnostic |
| `test_eval/test_metrics.json` | held-out test accuracy / macro-F1 / per-class recall |
| `best_params.json` | XGBoost hyperparameters used |
| `confusion_matrix.png`, `feature_importance.png`, `abstain_curve.png` | diagnostic plots |

## Caveats

- **Safrinha (second-season) SOJA** — still limited (~144 mostly-safrinha SOJA KMLs were never in
  the matched field set), so held-out safrinha recall is ~0.63. **SOJA→MILHO confusion is 0** in all
  three models, so the economic risk is contained. Recovery plan in `../STATUS_AND_ROADMAP.md` (Part II,
  Obstacle 3).
- **Regional / single-year** — trained on ~2025 South-Brazil fields; out-of-region/year is
  out-of-distribution (abstain gate is the safety net). See STATUS_AND_ROADMAP Part I §3.
