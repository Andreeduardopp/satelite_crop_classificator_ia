# Phenology Feature Pipeline — Technical Documentation

## What it does

Takes a directory of KML files (one per agricultural field) and produces a single
SQLite table where **each row is one field** with **90 numeric feature columns**
ready for XGBoost crop classification.

```
KML files  ──►  Sentinel Hub API  ──►  Cloud mask + Zonal stats  ──►  features.db
```

---

## How to run

```bash
# From the project root, with the venv active:
python src/pipelines/phenology_feature_pipeline.py
```

The `__main__` block defaults to:
- `root_dir` = `src/data/dataset_split/train`
- `max_per_crop` = 2  (small test — 14 fields total)
- `width / height` = `None` (adaptive resolution from polygon)
- `save_tiffs` = `False`
- `max_workers` = 4 (parallel stage requests)
- `use_stats_api` = `True` (Statistical API with Process API fallback)

To customize:

```python
from src.data_ingestion.request_sentinel_v1 import SentinelHubService
from src.pipelines.phenology_feature_pipeline import PhenologyFeaturePipeline

service  = SentinelHubService()
pipeline = PhenologyFeaturePipeline(
    service,
    output_dir="src/data/features",
    max_workers=4,          # concurrent stage requests per field
    use_stats_api=True,     # prefer Statistical API (lighter, faster)
    request_delay=0.3,      # seconds between fields
)

df = pipeline.process_directory(
    root_dir="src/data/dataset_split/train",
    max_per_crop=None,      # None = process all KMLs
    save_tiffs=False,       # True = also dump GeoTIFFs per stage (forces Process API)
    # width=None, height=None → adaptive resolution from polygon bounds
)
```

Output: `src/data/features/features.db` → table `phenology_features`.

**Resumable**: rerun the same command after a crash and it picks up where it left off —
already-processed `field_id`s are skipped automatically.

---

## API calls per row

Each row (= one field) hits the API once per phenological stage:

| Path              | Calls/field | Response size | Stats computed |
|-------------------|-------------|---------------|----------------|
| Statistical API   | **6**       | ~1 KB JSON    | Server-side    |
| Process API       | **6**       | ~400 KB tar   | Client-side    |

The pipeline tries the Statistical API first (`use_stats_api=True`).
If the account doesn't support it (HTTP 403/404), it falls back to the
Process API automatically for all remaining requests.

```
total_api_calls = 1 (auth, refreshed every 250 s) + N_fields × 6 (stages)
```

### Cost projection for the full dataset

| Subset        | Fields | API calls | Time (sequential) | Time (4 workers) |
|---------------|--------|-----------|-------------------|-------------------|
| Test run      | 14     | 84        | ~2 min            | ~40 s             |
| 50 per crop   | 350    | 2,100     | ~30 min           | ~8 min            |
| 500 per crop  | 3,500  | 21,000    | ~5 h              | ~1.5 h            |
| Full train    | 21,000 | 126,000   | ~30 h             | ~8 h              |
| Full dataset  | 28,000 | 168,000   | ~40 h             | ~11 h             |

**Sentinel Hub free tier** = 30,000 requests/month.
**Exploration tier** = 100,000/month.
Full train (126k) needs a paid plan or splitting across months.

### What each call returns

**Statistical API** (default):
```
Request  ──►  POST /api/v1/statistics
         ◄──  JSON { mean, stDev, percentiles: {10, 50, 90} } per index
              Cloud masking baked into evalscript dataMask
```

**Process API** (fallback, or when `save_tiffs=True`):
```
Request  ──►  POST /api/v1/process
         ◄──  TAR archive with 4 GeoTIFFs:
              ndvi_tiff.tif  (FLOAT32)
              evi_tiff.tif   (FLOAT32)
              ndwi_tiff.tif  (FLOAT32)
              scl_tiff.tif   (UINT8) ← cloud mask
```

Both paths use `mosaickingOrder: leastCC` — the server picks the least-cloudy
acquisition, no separate Catalog search needed.

---

## Output schema

```
phenology_features (95 columns)
├── field_id           TEXT     "SOJA_518366576-1"
├── crop_label         TEXT     "SOJA"
├── planting_date      TEXT     "2024-10-15"
├── area_hectares      REAL     139.50
├── stages_covered     REAL     0–6 (data quality signal)
│
├── NDVI_mean_baseline      REAL  ┐
├── NDVI_median_baseline    REAL  │  6 stages × 3 indices × 5 stats
├── NDVI_std_baseline       REAL  │  = 90 feature columns
├── NDVI_p10_baseline       REAL  │
├── NDVI_p90_baseline       REAL  │  NULL when stage had no cloud-free pixels
├── NDVI_mean_emergence     REAL  │  (XGBoost handles NULL natively)
├── ...                           │
├── EVI_p90_maturity        REAL  ┘
```

Querying from Python:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("src/data/features/features.db")
df = pd.read_sql("SELECT * FROM phenology_features", conn)
conn.close()

X = df.drop(columns=["field_id", "crop_label", "planting_date"])
y = df["crop_label"]
```

---

## Old pipeline vs new pipeline

| Aspect             | `simsec_pipeline.py` (old)              | `phenology_feature_pipeline.py` (new)     |
|--------------------|-----------------------------------------|-------------------------------------------|
| Stages             | Arbitrary day offsets (0,20,35,55…)     | 6 named biological stages per crop        |
| Indices            | NDVI, EVI, NDWI, NDBI                  | NDVI, NDWI, EVI                           |
| Stats per cell     | Median only                             | mean, median, std, p10, p90               |
| Cloud masking      | None                                    | SCL band (Sentinel-2 L2A)                |
| Feature columns    | Variable (depends on offsets)           | Fixed 90                                  |
| Area calculation   | Not computed                            | Geodesic hectares (pyproj)                |
| Quality metric     | None                                    | `stages_covered` (0–6)                    |
| Mosaic strategy    | Server default                          | `leastCC` (least cloud cover)             |
| API calls/field    | 7–8 (variable offsets)                  | 6 (fixed stages)                          |
| API path           | Process API only                        | Statistical API + Process API fallback    |
| Concurrency        | Sequential                              | ThreadPoolExecutor (4 workers/field)      |
| Crash recovery     | Loses all progress                      | Incremental resume (skip existing rows)   |
| Token management   | Single auth at start                    | Auto-refresh before 300 s expiry          |
| Resolution         | Fixed 100×100                           | Adaptive from polygon bounds (~10 m/px)   |
| Output             | SQLite `optimized_phenology_dataset.db` | SQLite `features.db`                      |
| Column naming      | `ndvi_day_55`                           | `NDVI_mean_flowering`                     |

---

## Implemented optimizations

All six optimizations are built into the pipeline. No flags to toggle
except `use_stats_api` and `max_workers`.

### 1. Statistical API (server-side stats)

**`use_stats_api=True` (default)**

Instead of downloading 4 rasters (~400 KB) and computing stats in Python,
the Statistical API returns a ~1 KB JSON with mean, stDev, and percentiles
computed server-side. Cloud masking is baked into the evalscript's `dataMask`
output — cloudy pixels (SCL not in {2,4,5,6,7}) are excluded before the
server computes statistics.

```
Before:  download tar → parse 4 TIFFs → numpy mask → numpy stats
After:   POST JSON → parse 15 numbers from response
```

If the account returns 403/404, the pipeline logs a warning and falls back
to the Process API automatically for all remaining requests.

When `save_tiffs=True`, the Process API is used instead (since we need
the actual raster data on disk).

### 2. Concurrent stage requests (ThreadPoolExecutor)

**`max_workers=4` (default)**

All 6 stages for one field are dispatched to a thread pool simultaneously.
With 4 workers, a field completes in ~2 round-trips instead of 6:

```
Sequential:  S1 → S2 → S3 → S4 → S5 → S6     (6 round-trips)
Parallel:    [S1,S2,S3,S4] → [S5,S6]           (2 round-trips)
```

Token refresh (`_get_token`) is thread-safe via `threading.Lock`.

### 3. Incremental / crash-safe persistence

Each row is committed to SQLite **immediately** after its 6 stages finish.
On restart, existing `field_id`s are loaded from the DB and skipped:

```
Run 1: processes fields 1–500, crashes at 501  → 500 rows in DB
Run 2: loads 500 existing IDs, skips them, continues from 501
```

Maximum data loss on crash: **one field** (the one in-flight).

### 4. Token refresh

`_get_token()` tracks when the token was obtained. If >250 seconds have
elapsed (Sentinel Hub tokens expire at ~300 s), it re-authenticates before
the next API call. This is transparent — no 401 errors mid-run.

### 5. Adaptive resolution

When `width` and `height` are not explicitly passed, the pipeline computes
them from the polygon's bounding box at Sentinel-2 native resolution (~10 m/px):

```
SOJA_518366576-1  (139 ha)  →  122×183 px   (captures field heterogeneity)
MILHO_517259584-1 (2.5 ha)  →   32×33 px    (saves bandwidth on tiny fields)
SOJA_517274165-1  (181 ha)  →  118×221 px
```

Clamped to [32, 2500] (Process API limits). The Statistical API always uses
`resx=10, resy=10` (10 m native), so adaptive resolution only affects the
Process API fallback path.

You can still override with explicit `width=100, height=100` if you want
uniform raster size across all fields.

### 6. Statistical API auto-fallback

If the first Statistical API call returns HTTP 403 or 404 (account doesn't
have access), `use_stats_api` is flipped to `False` automatically. All
subsequent fields use the Process API directly — no manual intervention needed.

---

## Recommended run strategy

```
Phase 1 — Validate (14 fields, ~40 s)
  python src/pipelines/phenology_feature_pipeline.py
  # defaults: max_per_crop=2, use_stats_api=True, max_workers=4

Phase 2 — Small sample (350 fields, ~8 min)
  max_per_crop=50

Phase 3 — Full extraction (21,000 fields)
  max_per_crop=None
  # Resumable — safe to Ctrl+C and restart
  # Estimated: ~8 h with 4 workers + Statistical API
```

After each phase, inspect the quality:

```python
import sqlite3, pandas as pd

conn = sqlite3.connect("src/data/features/features.db")
df = pd.read_sql("SELECT * FROM phenology_features", conn)
conn.close()

print(df["stages_covered"].value_counts().sort_index())
print(df.groupby("crop_label")["stages_covered"].mean())
```

If >30% of rows have `stages_covered < 4`, investigate whether the planting
dates fall in a period with persistent cloud cover for that region.
