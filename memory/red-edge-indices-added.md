---
name: red-edge-indices-added
description: Red-edge + SWIR optical indices added to the feature pipeline to attack AVEIA/TRIGO confusion
metadata:
  type: project
---

Added 5 optical indices to the pipeline to target the [[aveia-trigo-bottleneck]]: **NDRE, CIRE, MTCI** (red-edge chlorophyll), **PSRI** (senescence), **NDMI** (canopy water). Requires Sentinel-2 bands B05/B06/B07/B8A (B11 already fetched).

**Where:** `OPTICAL_INDEX_KEYS` (single source of truth) drives both evalscripts (`EVALSCRIPT_OPTICAL_STATS` = Stats API, `EVALSCRIPT_PROCESS` = raster fallback) and both parsers in `src/pipelines/phenology_feature_pipeline_v5.py`. Unbounded ratios (CIRE/MTCI/PSRI) are clamped server-side in the evalscript. Legacy `src/data_ingestion/request_sentinel_v1.py` evalscript (used by `simsec_pipeline.py`) updated additively. `train_xgboost_v5.py` `INDICES` extended so engineered features (deltas/peaks/drydown) cover the new indices.

**How to apply:** New indices need fresh Sentinel-2 requests (the DB stores only computed stats, never raw bands). Two paths: (a) full re-run into a fresh `--output-dir`, or (b) the efficient **`--mode backfill-optical`** added to `phenology_feature_pipeline_v5.py` — mirrors `backfill_sar`, populates only the 5 new columns on existing rows, skips SAR entirely, and does NOT overwrite NDVI/NDWI/EVI. Tracked by an `optical_backfill_done` flag column. Run: `python src/pipelines/phenology_feature_pipeline_v5.py --mode backfill-optical --output-dir src/data/features_v5`. As of 2026-06-18 the backfill is DONE: all 6226 rows have `optical_backfill_done=1` (NDRE ~89%, PSRI ~98%, NDMI ~97% populated). These indices now feed the [[second-season-soja-fix]] discriminators in `train_xgboost_v6`.
