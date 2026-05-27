# Plan: MLflow + FastAPI Serving for Crop Classifier

## Context

The project classifies crops (SOJA, MILHO, FEIJAO, etc.) from Sentinel satellite imagery using an XGBoost + ExtraTrees ensemble trained on phenological features. Currently, training outputs go to flat JSON/PNG files with no experiment tracking, and there's no way to serve predictions via an API. The goal is:

1. **MLflow** for experiment tracking and model registry during training
2. **FastAPI server** that receives KML + planting_date + culture_key, runs the full pipeline (fetch satellite data -> extract features -> predict), and returns the classification

---

## New File Structure

```
src/
  config.py                              # NEW: env vars, paths
  serving/                               # NEW: entire directory
    __init__.py
    app.py                               # FastAPI app + lifespan
    schemas.py                           # Pydantic request/response models
    inference.py                         # Orchestrates: KML -> Sentinel -> features -> predict
    artifacts.py                         # Loads model, selected_features, label_encoder
  model/
    train_xgboost_v3.py                  # MODIFY: add MLflow logging + save ensemble/label_encoder
  pipelines/
    phenology_feature_pipeline_v2.py     # MODIFY: add parse_kml_content() for in-memory KML
  data_ingestion/
    request_sentinel_v1.py               # MODIFY: read credentials from env vars
```

---

## Implementation Steps

### Phase 1: Config + Credential Cleanup

**`src/config.py`** (new) -- Central config reading from env vars:
- `SENTINEL_CLIENT_ID`, `SENTINEL_CLIENT_SECRET`, `SENTINEL_BASE_URL`
- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`
- `ARTIFACTS_DIR` (defaults to `src/model/output_v3`)

**`src/data_ingestion/request_sentinel_v1.py`** (modify) -- `SentinelHubService.__init__` accepts optional `client_id`/`client_secret` params, reads from env vars via config, falls back to current hardcoded values with a deprecation warning.

**`.env.example`** (new) -- Documents required env vars (not committed with secrets).

### Phase 2: MLflow Integration into Training

**`src/model/train_xgboost_v3.py`** (modify):

1. **New artifacts to save** (currently missing):
   - `ensemble_model.joblib` -- full VotingClassifier (currently only XGBoost JSON is saved)
   - `label_encoder.json` -- `{"classes": ["ARROZ", "AVEIA", ...]}` (needed for index->name mapping at inference)

2. **MLflow wrapping** -- wrap `train_and_evaluate()` body in `mlflow.start_run()`:
   - **Log params**: n_folds, optuna_trials, keep_ratio, n_samples, n_features, n_classes, all best hyperparameters (prefixed `xgb_`/`et_`)
   - **Log metrics**: accuracy, f1_macro, f1_weighted for each model (xgboost, extratrees, ensemble) + per-class precision/recall/F1
   - **Log artifacts**: entire output_v3 directory (model, configs, plots)
   - **Register model**: `mlflow.sklearn.log_model(ensemble, registered_model_name="crop_classifier_ensemble")`

3. **Tune functions** return `(best_params, best_value)` instead of just `best_params` so we can log Optuna's best score.

### Phase 3: FastAPI Server

**`src/serving/schemas.py`** -- Pydantic models:
- `PredictionRequest`: `kml_content` (str), `planting_date` (YYYY-MM-DD), `culture_key` (e.g. "SOJA")
- `PredictionResponse`: `predicted_crop`, `confidence`, `probabilities` (all classes), `stages_fetched`, `processing_time_seconds`
- `HealthResponse`, `ErrorResponse`

**`src/serving/artifacts.py`** -- `ModelArtifacts` class:
- Loads `ensemble_model.joblib`, `selected_features.json`, `label_encoder.json` at startup
- `predict(feature_vector)` -> returns predicted class, confidence, all probabilities

**`src/pipelines/phenology_feature_pipeline_v2.py`** (modify) -- Add `parse_kml_content(kml_string)` next to existing `parse_kml_polygon(kml_path)`. Same logic, uses `ET.fromstring()` instead of `ET.parse()`.

**`src/serving/inference.py`** -- Core inference function `run_inference()`:
1. Parse KML content -> polygon coordinates
2. Validate culture_key against `CROP_STAGE_WINDOWS`
3. Create `PhenologyFeaturePipeline` instance, call `_process_field()` to fetch all 6 stages (optical + SAR)
4. Convert to 1-row DataFrame, run `engineer_features()` + `add_null_indicators()` (reused from training)
5. **Align columns** to `selected_features` -- add missing null-indicator cols as 0, missing features as NaN
6. Call `artifacts.predict()`, return result

**`src/serving/app.py`** -- FastAPI app:
- `lifespan`: loads `ModelArtifacts` + `SentinelHubService` at startup
- `GET /health` -- model status + available cultures
- `POST /predict` -- full pipeline prediction

### Phase 4: Dependencies

Add to `requirements.txt`:
```
mlflow>=2.12.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-dotenv>=1.0.0
joblib>=1.4.0
python-multipart>=0.0.9
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Serve ensemble vs XGBoost alone | **Full ensemble** (VotingClassifier) | Soft voting produces better-calibrated probabilities; memory cost minimal for tree models |
| Model serialization | **joblib** | Standard for sklearn; MLflow registry for tracking, joblib for runtime loading |
| Sync vs async endpoint | **Sync** (runs in FastAPI threadpool) | 12 Sentinel API calls take ~30-60s; inherent latency. Async task queue is Phase 2 |
| Feature alignment at inference | Align to `selected_features.json` | Handles missing null-indicator columns; ensures exact column order the model expects |
| Reuse pipeline code | Call `_process_field()` directly | Avoids duplicating 400+ lines of Sentinel fetch/parse logic |

---

## Inference Flow (POST /predict)

```
Client sends: { kml_content, planting_date, culture_key }
                        |
                        v
              parse_kml_content(kml)
              -> polygon coordinates
                        |
                        v
         PhenologyFeaturePipeline._process_field()
           |-- For each of 6 stages (baseline -> maturity):
           |     |-- Compute date window from planting_date + CROP_STAGE_WINDOWS[culture_key]
           |     |-- Fetch Sentinel-2 optical stats (NDVI, NDWI, EVI)
           |     |-- Fetch Sentinel-1 SAR stats (VV, VH, CR, RVI)
           |     |-- Extract: mean, median, std, p10, p90 per index
           |-- Also compute: area_hectares, latitude, longitude, stages_covered
                        |
                        v
              engineer_features(1-row DataFrame)
              -> stage deltas, peak detection, greenup/senescence rates,
                 cross-index ratios, temporal CV, cumulative, etc.
                        |
                        v
              add_null_indicators()
                        |
                        v
              Align to selected_features.json (column selection + ordering)
                        |
                        v
              ensemble.predict_proba() -> probabilities per class
                        |
                        v
Client receives: { predicted_crop, confidence, probabilities[], stages_fetched, processing_time }
```

---

## Error Handling

| Scenario | HTTP Status | Message |
|---|---|---|
| Invalid KML content | 400 | "Could not parse polygon from KML content" |
| Unknown culture_key | 400 | Lists valid keys |
| Invalid date format | 422 | Automatic FastAPI/Pydantic validation |
| All stages return no data | 500 | "No satellite data available for this field/date" |
| Sentinel Hub auth failure | 503 | "Sentinel Hub authentication failed" |
| Model not loaded | 503 | "Model not loaded" |

---

## Feature Alignment at Inference (Critical Detail)

The training script runs `engineer_features()` + `add_null_indicators()` on the full dataset, then selects features by XGBoost gain importance. At inference with 1 row:

1. `engineer_features()` works correctly on a 1-row DataFrame (verified all 11 feature categories)
2. `add_null_indicators()` adds `_is_null` columns for any column that is NULL in this row (100% > 10% threshold)
3. **Alignment step**: For each column in `selected_features.json`:
   - If present in the engineered output: use it
   - If missing and it's a `_is_null` column: add with value `0.0` (column wasn't null)
   - If missing and it's a regular feature: add with value `NaN` (XGBoost handles missing values natively)
4. Select only `selected_features` columns, in the exact order the model expects

---

## Running

```bash
# Training with MLflow
python src/model/train_xgboost_v3.py
mlflow ui  # View experiments at http://localhost:5000

# Serving
set SENTINEL_CLIENT_ID=8103d553-...
set SENTINEL_CLIENT_SECRET=4Ewmc2u...
set ARTIFACTS_DIR=src/model/output_v3
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{...}"
```

---

## Future Enhancements (Phase 2)

- **Async endpoint** (`POST /predict/async` + `GET /predict/status/{task_id}`) for long-running predictions via Celery/Redis
- **Batch endpoint** (`POST /predict/batch`) for multiple fields at once
- **Result caching** -- cache Sentinel Hub responses by coordinates + date range to avoid redundant API calls
- **Docker** -- containerize the server with all dependencies
