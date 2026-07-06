# Best crop-classifier models (5 / 6 / 7 cultures)

Flagship XGBoost dense models, one per crop count. Each folder is a full copy of its training
run (model + selected features + abstain policy + metrics + plots).

> ⚠️ **A model is bound to the feature schema it was trained on.** At inference you MUST extract
> features with the *same pipeline* listed below and pass them through the model's own
> `selected_features.json`. The 5- and 6-culture models use the v6 grid (17 dekads, +155 d); the
> 7-culture model uses the extended v7 grid (29 dekads, +275 d). They are **not** interchangeable —
> feeding v6 features to the 7-crop model (or vice-versa) will silently misclassify.

## The three models

| Folder | Cultures | Held-out test | Feature pipeline | Feature DB (train) |
|---|---|---|---|---|
| `5_culturas_no_aveia/` | ARROZ, FEIJAO, MILHO, SOJA, TRIGO | **0.976** acc / 0.976 F1 (n=957) | `phenology_feature_pipeline_v6.py` | `features_v6` (17 dekads, +155 d) |
| `6_culturas_arroz/` | ARROZ, AVEIA, FEIJAO, MILHO, SOJA, TRIGO | **0.941** acc / 0.941 F1 (n=1149) | `phenology_feature_pipeline_v6.py` | `features_v6` (17 dekads, +155 d) |
| `7_culturas_cafe/` | ARROZ, AVEIA, **CAFE**, FEIJAO, MILHO, SOJA, TRIGO | **0.936** acc / 0.937 F1 (n=1310) | `phenology_feature_pipeline_v7.py` | `features_v6_ext` (29 dekads, +275 d) |

All three were trained by `train_xgboost_v7.py` (weighted 5-fold OOF CV, class balance, safrinha-SOJA
up-weight, global abstain gate, season-stratified diagnostic). "v7" is the *trainer* name; the
distinction that matters for inference is the *pipeline* / grid in the table above.

> 📂 **The exact training/test data is documented in [`DATASETS.md`](DATASETS.md)** (provenance,
> feature schema, class/geographic/temporal variability, biases) and snapshotted byte-for-byte in
> [`datasets/`](datasets/) with checksums in [`datasets/MANIFEST.json`](datasets/MANIFEST.json).

## Which pipeline, and why it differs

Both pipelines issue one Sentinel-Hub Statistical-API request per source and aggregate the crop
calendar into time bins server-side, on a hybrid grid: **dekadal (P10D)** over the full season +
**fine (P5D)** over flowering→maturity. The only difference is the dekadal grid length:

- **`phenology_feature_pipeline_v6.py`** — dekadal grid `planting−15 .. +155 d` (`N_DEKADS = 17`).
  Fits every annual. Used by the **5-** and **6-culture** models.
- **`phenology_feature_pipeline_v7.py`** — a thin subclass of v6 that extends the dekadal grid to
  `planting−15 .. +275 d` (`N_DEKADS = 29`) so it covers **CAFE's ~270-day perennial cycle**. Used by
  the **7-culture** model. Because a single classifier can't know the crop at inference, every crop
  is extracted over this longer grid (annuals carry real post-harvest data in the late dekads — no
  train/serve skew). Schema ≈ 1485 columns (under SQLite's ceiling).

## Per-class recall (held-out test)

| Culture | 5-crop | 6-crop | 7-crop |
|---|---|---|---|
| ARROZ | 0.97 | 0.97 | 0.97 |
| FEIJAO | 0.99 | 1.00 | 1.00 |
| MILHO | 0.93 | 0.95 | 0.94 |
| SOJA | 0.98 | 0.98 | 0.96 |
| TRIGO | 0.99 | 0.86 | 0.86 |
| AVEIA | — | 0.89 | 0.86 |
| CAFE | — | — | 0.96 |

TRIGO is near-perfect in the 5-crop model only because AVEIA (its one hard confuser) is absent;
adding AVEIA back in the 6-/7-crop models restores the AVEIA↔TRIGO error pair (~0.86 each), which is
the dominant remaining error budget. CAFE separates cleanly on arrival (F1 0.97): 9/200 misses leak
to annual look-alikes (MILHO 5, SOJA 3, ARROZ 1), 1 false positive.

## Using a model (inference sketch)

1. Extract features for the field with the **pipeline named above** for that model (v6 for 5/6-crop,
   v7 for 7-crop), anchored at the field's planting date.
2. Reindex the feature row to the model's `selected_features.json` (add any missing column as NaN,
   in that exact order).
3. `model = xgb.XGBClassifier(); model.load_model("xgboost_crop_classifier.json")`,
   then `proba = model.predict_proba(X)`.
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

- **7-crop safrinha SOJA regression.** The v7 re-extraction matched only 633/777 SOJA (the ~144
  missing were mostly second-season), dropping held-out safrinha recall to 0.60 (n=15). The critical
  **SOJA→MILHO confusion is still 0** in all three models. See `../STATUS_AND_ROADMAP.md` §5.4 for the
  recovery plan. The 5- and 6-crop models keep the better safrinha recall (0.867).
- Source runs (unchanged) live in `../runs_v7/`: `20260629_111811_dense_5crop_no_aveia`,
  `20260628_075718_dense_6crop_arroz`, `20260630_021450_dense_7crop_cafe`.
