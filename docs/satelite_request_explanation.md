# Satellite Request Explanation

## What You Send

A KML file containing a polygon (your field boundary). The filename encodes metadata:
`CROP_fieldId_plantio_DD-MM-YY_colheita_DD-MM-YY.kml`

The pipeline extracts the crop type, field ID, planting date, and harvest date from this naming convention.

---

## What Gets Calculated

For each phenological stage (baseline, emergence, vegetative, flowering, grain_fill, maturity), the pipeline requests **all pixels inside your polygon** from Sentinel Hub for the corresponding time window.

### Optical Indices (Sentinel-2) — computed per pixel

| Index | Formula | What it measures |
|-------|---------|-----------------|
| **NDVI** | (B08 − B04) / (B08 + B04) | Vegetation vigor — ratio of near-infrared (NIR) to red reflectance. Healthy vegetation absorbs red and reflects NIR strongly. Range: −1 to +1, crops typically 0.3–0.9. |
| **EVI** | 2.5 × (B08 − B04) / (B08 + 6×B04 − 7.5×B02 + 1) | Enhanced vegetation index — corrects for atmospheric and soil background effects. Less saturated than NDVI over dense canopy. |
| **NDWI** | (B03 − B08) / (B03 + B08) | Normalized difference water index — measures leaf water content using green vs NIR reflectance. |

### SAR Indices (Sentinel-1 GRD) — computed per pixel

| Index | Formula | What it measures |
|-------|---------|-----------------|
| **VV** | 10 × log₁₀(VV) in dB | Backscatter in vertical-vertical polarization. Sensitive to soil moisture and crop height. |
| **VH** | 10 × log₁₀(VH) in dB | Backscatter in vertical-horizontal polarization. Sensitive to volume scattering (crop biomass). |
| **CR** | VH / VV (linear) | Cross-polarization ratio — indicates crop structure and canopy complexity. |
| **RVI** | 4 × VH / (VV + VH) | Radar vegetation index — proxy for biomass that works through clouds. |

### Cloud Filtering (Optical Only)

Before computing optical statistics, each pixel is checked against the Scene Classification Layer (SCL). Only pixels with these SCL classes are considered valid:

| SCL Code | Class |
|----------|-------|
| 2 | Dark area pixels |
| 4 | Vegetation |
| 5 | Bare soil |
| 6 | Water |
| 7 | Unclassified (low probability cloud) |

Clouds, cloud shadows, cirrus, and snow are excluded. This ensures the vegetation indices reflect actual ground conditions, not atmospheric contamination.

SAR data (Sentinel-1) is not affected by clouds, which is one of its key advantages.

### Expanded SAR Windows (v3)

Sentinel-1 revisits the same location roughly every ~12 days in Brazil, while Sentinel-2 (optical) has a ~5 day revisit. This means short phenological stage windows — especially for fast-cycle crops like FEIJAO (10–20 days per stage) — often contain no SAR acquisition at all.

**The problem (pipeline v2):** Optical and SAR windows were identical. Result:

| Crop | Typical stage window | SAR coverage |
|------|----------------------|--------------|
| CAFE | 30–60 days | ~97% |
| MILHO | 25–30 days | ~41% |
| FEIJAO | 10–20 days | ~6% |

**The solution (pipeline v3):** SAR windows are symmetrically expanded to guarantee a minimum of **24 days** of coverage (>= 1 Sentinel-1 pass). Optical windows remain unchanged.

```
expand_sar_window(day_start, day_end, min_days=24)

Example — FEIJAO emergence (0 → 10 days, 10d span):
  Optical window: day 0 to day 10   (10 days — fine for S2 with 5d revisit)
  SAR window:     day -7 to day 17  (24 days — guarantees >= 1 S1 pass)

Example — CAFE vegetative (30 → 90 days, 60d span):
  Optical window: day 30 to day 90  (60 days — no change)
  SAR window:     day 30 to day 90  (60 days — already >= 24d, no expansion)
```

The expansion is symmetric around the stage midpoint, so SAR data remains centered on the correct phenological period. This accepts a small trade-off — slight temporal overlap between adjacent stages for SAR — in exchange for significantly higher coverage.

Additionally, the v3 pipeline marks each processed row with a `sar_backfill_done` flag, preventing infinite re-requests for fields that genuinely have no SAR coverage.

---

## What You Get Back

The pipeline does **not** return individual pixel values. Instead, it aggregates all valid pixels across the entire polygon into **5 summary statistics**:

| Statistic | Meaning |
|-----------|---------|
| **mean** | Average value across all pixels in your field |
| **median** | 50th percentile — robust to outliers |
| **std** | Standard deviation — spatial variability within the field |
| **p10** | 10th percentile — represents the worst-performing area |
| **p90** | 90th percentile — represents the best-performing area |

**Example:** `NDVI_mean_vegetative = 0.72` means the average NDVI across all pixels in your polygon during the vegetative stage was 0.72.

### Total Output Per Field

| Category | Count |
|----------|-------|
| Optical features | 3 indices × 5 stats × 6 stages = **90** |
| SAR features | 4 indices × 5 stats × 6 stages = **120** |
| Metadata | field_id, crop_label, planting_date, area_hectares, latitude, longitude, stages_covered = **7** |
| **Total columns** | **217** |

Each field produces **one row** in the database representing its full phenological profile.

---

## Pixel Size in the Real World

Sentinel-2 optical bands used in this pipeline (B02, B03, B04, B08) have a native resolution of **10 meters per pixel**. This means each pixel covers a **10m × 10m = 100 m²** area on the ground.

Sentinel-1 SAR (GRD product, IW mode) also delivers data at **10m × 10m** resolution after terrain correction.

For a typical soybean field of 50 hectares (500,000 m²), you get approximately **5,000 pixels** of data per acquisition. The statistics aggregate these thousands of pixel-level measurements into a compact field-level summary.

---

## Why Use the Statistical API Instead of Downloading Images

The pipeline supports two modes: the **Statistical API** (server-side aggregation) and the **Process API** (downloading raster images). The Statistical API is the default, and here is why:

### 1. No Image Transfer = Faster and Cheaper

With the Process API, you download full GeoTIFF rasters (one per index), then compute statistics locally. For a 50 ha field at 10m resolution, that is ~5,000 pixels × 4 bands × 32-bit float = **~80 KB per stage request**. Multiply by hundreds of fields and 6 stages each and the bandwidth adds up.

The Statistical API computes mean, median, std, p10, p90 **on Sentinel Hub's servers** and returns only a small JSON (~1 KB). This is orders of magnitude less data to transfer.

### 2. Server-Side Cloud Masking

With raster downloads, your code must download the SCL band, build a mask, and filter pixels locally. The Statistical API handles cloud masking inside the evalscript on the server — the `dataMask` output tells the server which pixels to include in the aggregation. You get clean statistics without writing pixel-level filtering logic.

### 3. No Local Storage Required

Raster downloads produce thousands of TIFF files that need to be stored, organized, and cleaned up. The Statistical API returns numbers directly — the pipeline writes them straight to the SQLite database. No intermediate files, no disk space concerns.

### 4. Lower Processing Unit (PU) Cost

Sentinel Hub charges Processing Units based on the output size. The Statistical API returns a tiny JSON response, which costs fewer PUs than requesting full-resolution rasters. When processing hundreds of fields across 6 stages, this difference is significant for your monthly PU budget.

### 5. When You Still Want Images

The Process API (raster download) is still available via `save_tiffs=True`. This is useful when you need to:
- Visually inspect a specific field
- Generate spatial maps (e.g., within-field NDVI variability maps)
- Debug unexpected statistical values
- Create figures for reports or publications

For the ML training pipeline, where you only need numerical features, the Statistical API is the better choice.

---

## Comparison with the Legacy Request Code

The original data ingestion (`src_legacy/data_ingestion/processamento_sentinel_Hub.py`) used a fundamentally different approach to retrieve satellite data. Below is a detailed comparison of what changed and why.

### Legacy Approach — How It Worked

```
KML file → extract bounding box (min/max lat/lon)
         → WCS request to Sentinel Hub (download full PNG/TIFF image)
         → save to disk
         → post-hoc cloud analysis on the downloaded image
         → return first image only
```

The old code used the `sentinelhub` Python package with the **WCS (Web Coverage Service)** protocol — a legacy OGC standard that Sentinel Hub still supports but no longer recommends. A single function call (`request_sentinel_hub`) would download one image at a time for a given date.

### Key Differences

#### 1. Bounding Box vs Actual Polygon Geometry

| Aspect | Legacy | Current Pipeline |
|--------|--------|-----------------|
| **Geometry sent** | Bounding box (min/max of coordinates) | Full polygon from KML |
| **Pixels included** | All pixels in the rectangle, including area outside the field | Only pixels inside the polygon boundary |
| **Impact** | Neighboring fields, roads, and water bodies contaminate the statistics | Clean field-level data — no external contamination |

The legacy code extracted the bounding box from the KML:
```python
min_x = min(coord[0] for coord in coords_list)
max_x = max(coord[0] for coord in coords_list)
bbox = BBox(bbox=[min_x, min_y, max_x, max_y], crs=CRS.WGS84)
```

For an irregularly shaped field, the bounding box can include 30–50% of pixels that are outside the actual field boundary. The current pipeline sends the full polygon geometry to Sentinel Hub, which clips at the server side.

#### 2. Image Download vs Server-Side Statistics

| Aspect | Legacy | Current Pipeline |
|--------|--------|-----------------|
| **What is returned** | Full raster image (PNG or TIFF) | JSON with 5 statistics (mean, median, std, p10, p90) |
| **Data volume** | ~100 KB–1 MB per image | ~1 KB JSON per stage |
| **Local processing** | Required (cloud analysis, index computation) | None — statistics computed on server |
| **Storage** | Thousands of image files on disk | Numbers written directly to SQLite |

The legacy code downloaded raw images and saved them to disk:
```python
imagens = wcs_request.get_data(save_data=True)
```

The current pipeline receives pre-computed statistics via the Statistical API — no images, no disk I/O.

#### 3. Cloud Masking Strategy

| Aspect | Legacy | Current Pipeline |
|--------|--------|-----------------|
| **Tile-level filter** | `maxcc=0.60` — rejects tiles with >60% cloud cover | Not needed (pixel-level masking) |
| **Pixel-level masking** | Post-download via `analisar_cobertura_de_nuvens()` function | Server-side via SCL band in evalscript (`dataMask`) |
| **Problem** | A tile with 55% clouds passes the filter; the field might still be 100% cloudy | Each pixel is individually checked — only clear pixels enter the statistics |

The legacy approach had a two-stage problem: first, it could only reject entire tiles above 60% cloud cover. A tile with 55% overall cloud cover could still have the field itself completely under clouds. Second, cloud detection was done *after* downloading the image, wasting bandwidth on unusable data.

The current pipeline applies SCL-based pixel-level masking inside the evalscript on the server. Only pixels classified as clear (SCL classes 2, 4, 5, 6, 7) contribute to the statistics. Cloudy pixels are never counted.

#### 4. Temporal Strategy

| Aspect | Legacy | Current Pipeline |
|--------|--------|-----------------|
| **Time window** | Fixed 5-day window from a single date | Crop-specific phenological stage windows (10–60 days each) |
| **Mosaicking** | Single acquisition (one pass) | `leastCC` mosaicking — best cloud-free composite across the window |
| **Phenological awareness** | None — just "take an image at this date" | 6 stages aligned to planting date and crop growth cycle |

The legacy code requested data for a fixed 5-day window:
```python
data_inicial = data - timedelta(days=5)
```

This meant: if the satellite didn't pass over the field in those 5 days, or if it was cloudy during that pass, you got nothing. No fallback, no composite.

The current pipeline uses windows of 15–60 days (depending on the crop stage), with `leastCC` mosaicking — Sentinel Hub selects the least cloudy acquisition within the window automatically.

#### 5. Bands and Indices

| Aspect | Legacy | Current Pipeline |
|--------|--------|-----------------|
| **Bands** | Pre-configured layer (`BANDAS_RBN` — likely RGB+NIR) | Custom evalscript computing NDVI, EVI, NDWI per pixel |
| **Index computation** | Not done — raw band image returned | Computed server-side inside the evalscript |
| **SAR data** | Not available | Sentinel-1 VV, VH, CR, RVI via separate request |

The legacy code relied on a pre-configured layer in the Sentinel Hub dashboard (`BANDAS_RBN`). The index computation (if any) would need to be done locally after downloading. The current pipeline defines evalscripts that compute NDVI, EVI, and NDWI on the server, and separately fetches SAR indices from Sentinel-1.

#### 6. API Protocol

| Aspect | Legacy | Current Pipeline |
|--------|--------|-----------------|
| **Protocol** | WCS (Web Coverage Service) — OGC legacy standard | Process API + Statistical API (Sentinel Hub REST API) |
| **Authentication** | `sentinelhub` Python package handles it | Direct OAuth2 token management with auto-refresh |
| **Dashboard dependency** | Required custom layer configured in web dashboard | Self-contained evalscripts in code |

WCS is a legacy protocol that Sentinel Hub maintains for backwards compatibility. The Process and Statistical APIs are the modern recommended approach, offering more flexibility (custom evalscripts, polygon clipping, server-side statistics).

#### 7. Error Handling and Resilience

| Aspect | Legacy | Current Pipeline |
|--------|--------|-----------------|
| **Retry logic** | None | Exponential backoff with 3 retries on HTTP 400/429/500/502/503 |
| **Token refresh** | Handled by `sentinelhub` package | Explicit refresh every 250 seconds with thread-safe lock |
| **Crash recovery** | None — restart from scratch | SQLite-based resume — skips already-processed fields |
| **Rate limiting** | None | Configurable `request_delay` between fields + staggered stage requests |

### Summary: What Changed and Why

```
LEGACY                              CURRENT PIPELINE
─────────────────────────           ─────────────────────────
Bounding box                   →    Full polygon geometry
Download full image            →    Server-side statistics (JSON)
Post-download cloud analysis   →    Server-side pixel-level SCL masking
5-day fixed window             →    Crop-specific phenological windows
Single acquisition             →    Least-cloud-cover mosaic
RGB+NIR layer (dashboard)      →    Custom evalscript (NDVI, EVI, NDWI)
No SAR                         →    Sentinel-1 VV, VH, CR, RVI
WCS legacy protocol            →    Process + Statistical REST API
No retry / no resume           →    Exponential backoff + SQLite resume
One image at a time            →    6 stages × 7 indices × 5 stats per field
```

The legacy code was designed for visual inspection — download an image, look at it. The current pipeline is designed for ML feature extraction — get clean numerical features across the full crop growth cycle, at scale, with resilience.
