# Satellite Crop Classification — Architecture & Results

## 1. Project Overview

This project classifies **7 crop types** cultivated in Brazil using satellite remote sensing data:

| Crop | Type | Season | Primary Regions |
|---|---|---|---|
| **SOJA** (Soybean) | Grain legume | Summer (Oct–Mar) | PR, RS, MT, GO |
| **MILHO** (Corn) | Cereal | Summer (Oct–Mar) | GO, MG, PR, MT |
| **ARROZ** (Rice) | Cereal (paddy) | Summer (Oct–Mar) | RS |
| **FEIJAO** (Bean) | Grain legume | Summer/Winter | PR, MG, BA |
| **TRIGO** (Wheat) | Winter cereal | Winter (May–Sep) | PR, RS |
| **AVEIA** (Oats) | Winter cereal | Winter (May–Sep) | PR, RS |
| **CAFE** (Coffee) | Perennial | Year-round | MG, SP, ES |

Each sample is a **farm field** defined by a KML polygon file, with known planting and harvest dates sourced from Brazilian agricultural registries. The pipeline extracts spectral and radar time-series features from Sentinel satellites, then a tree-based ensemble classifier predicts the crop type.

---

## 2. Data Sources

### 2.1 Sentinel-2 (Optical)

- **Satellite:** Sentinel-2 L2A (atmospherically corrected)
- **Resolution:** 10m/pixel
- **Revisit:** ~5 days
- **Bands used:** B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR), SCL (Scene Classification)
- **Cloud masking:** SCL classes 2, 4, 5, 6, 7 are considered clear sky. Cloudy/shadow pixels are excluded before computing statistics.

Three vegetation/water indices are derived per pixel:

| Index | Formula | What it captures |
|---|---|---|
| **NDVI** | (B08 − B04) / (B08 + B04) | Canopy greenness, chlorophyll activity |
| **EVI** | 2.5 × (B08 − B04) / (B08 + 6×B04 − 7.5×B02 + 1) | Enhanced vegetation, less saturation in dense canopy |
| **NDWI** | (B03 − B08) / (B03 + B08) | Leaf water content, irrigation status |

### 2.2 Sentinel-1 (SAR — Synthetic Aperture Radar)

- **Satellite:** Sentinel-1 GRD (Ground Range Detected)
- **Mode:** IW (Interferometric Wide Swath), dual-pol VV+VH
- **Resolution:** ~10m/pixel
- **Revisit:** ~6 days
- **Backscatter correction:** GAMMA0_TERRAIN (terrain-corrected)
- **Key advantage:** Penetrates clouds — zero data loss from weather

Four radar indices are derived:

| Index | Formula | What it captures |
|---|---|---|
| **VV** (dB) | 10 × log10(VV) | Surface roughness, soil moisture |
| **VH** (dB) | 10 × log10(VH) | Volume scattering — crop biomass, canopy density |
| **CR** (Cross-pol ratio) | VH / VV (linear) | Most crop-discriminative SAR feature |
| **RVI** (Radar Vegetation Index) | 4 × VH / (VV + VH) | Biomass proxy, depolarization ratio |

### 2.3 Why both optical + SAR?

Optical indices (NDVI, EVI) capture **what color the crop is** — chlorophyll, greenness, senescence. SAR captures **what structure the crop has** — canopy height, leaf geometry, stem density. Two crops can look identical in NDVI (same greenness) but have different radar signatures because their physical architecture differs.

This is critical for **TRIGO vs AVEIA** (wheat vs oats): both are winter cereals planted in the same region at the same time, with nearly identical spectral curves. But wheat grows taller with denser heads, producing a different VH backscatter profile.

---

## 3. Pipeline Architecture

### 3.1 Data Flow

```
KML polygon files (field boundaries + planting/harvest dates)
    │
    ├── Parse filename → crop_label, field_id, planting_date
    ├── Parse KML XML → polygon coordinates → centroid (lat, lon) + area (ha)
    │
    ▼
Phenological Stage Windows (crop-specific)
    │
    ├── baseline    (−15 to 0 days from planting)
    ├── emergence   (0 to 20 days)
    ├── vegetative  (20 to 50 days)
    ├── flowering   (50 to 80 days)
    ├── grain_fill  (80 to 110 days)
    └── maturity    (110 to 140 days)
    │
    ▼
For each stage: Sentinel Hub API calls
    │
    ├── Statistical API (Sentinel-2) → NDVI/EVI/NDWI mean, median, std, p10, p90
    ├── Statistical API (Sentinel-1) → VV/VH/CR/RVI mean, median, std, p10, p90
    │   (SAR runs independently — no cloud dependency)
    │
    ▼
SQLite Database (one row per field)
    │
    ├── Metadata: field_id, crop_label, planting_date, area_hectares, lat, lon
    ├── Optical:  90 columns (3 indices × 5 stats × 6 stages)
    ├── SAR:     120 columns (4 indices × 5 stats × 6 stages)
    └── Total:   ~215 columns per field
```

### 3.2 Crop-Specific Phenological Windows

Each crop has different growth timing. Using crop-specific stage windows ensures we capture the right phenological moment:

| Stage | SOJA | CAFE | TRIGO |
|---|---|---|---|
| baseline | −15 to 0 d | −15 to 0 d | −15 to 0 d |
| emergence | 0–20 d | 0–30 d | 0–20 d |
| vegetative | 20–55 d | 30–90 d | 20–50 d |
| flowering | 55–75 d | 90–150 d | 50–75 d |
| grain_fill | 75–100 d | 150–210 d | 75–105 d |
| maturity | 100–130 d | 210–270 d | 105–135 d |

CAFE (coffee) has much longer windows because it's a perennial crop with a 9-month fruit cycle.

### 3.3 API Strategy

**Primary: Statistical API** — Sentinel Hub computes zonal statistics server-side. Response is ~1 KB JSON vs ~400 KB raster download. Cloud masking is baked into the evalscript via dataMask.

**Fallback: Process API** — Downloads full rasters as TIFF when the Statistical API is unavailable or when raw imagery is needed for visualization.

### 3.4 Crash Safety

Each field is committed to SQLite immediately after its 6 stages complete. On restart, existing field_ids are loaded and skipped. A crash loses at most one field's work.

---

## 4. Feature Engineering

Starting from the base 90 optical columns (+ 120 SAR columns), the training script derives additional features that capture **temporal dynamics** and **cross-index relationships**:

### 4.1 Temporal Features

| Feature | Formula | Purpose |
|---|---|---|
| **Stage deltas** | value(stage_n+1) − value(stage_n) | Greenup/senescence rate between consecutive stages |
| **Peak stage** | argmax of mean values across stages | Which growth stage has the highest index |
| **Peak value / Min value** | max / min across stages | Overall range of the growth curve |
| **Amplitude** | peak − min | Total dynamic range |
| **Greenup rate** | (peak − baseline) / peak_stage_index | How fast the crop greens up |
| **Senescence rate** | (peak − maturity) / stages_after_peak | How fast the crop senesces |
| **Temporal CV** | std / mean across stages | Overall temporal variability |
| **Cumulative** | sum of means across stages | Total integrated greenness (biomass proxy) |

### 4.2 Cross-Index Features

| Feature | Formula | Purpose |
|---|---|---|
| **NDVI/EVI ratio** | NDVI_mean / EVI_mean per stage | Canopy saturation — dense vs sparse |
| **NDVI/NDWI ratio** | NDVI_mean / NDWI_mean per stage | Greenness vs water content balance |
| **NDVI−EVI difference** | NDVI − EVI at late stages | Late-stage divergence for crop separation |
| **NDVI−NDWI difference** | NDVI − NDWI at grain_fill/maturity | Separates crops with different drying patterns |
| **Std ratio** | NDVI_std / EVI_std at late stages | Variability structure differences |

### 4.3 Geographic & Calendar Features

| Feature | Source | Purpose |
|---|---|---|
| **latitude, longitude** | KML polygon centroid | Regional crop distribution (CAFE in MG vs TRIGO in RS) |
| **planting_doy** | Day-of-year from planting_date | Winter (TRIGO/AVEIA: ~May–Jun) vs summer (SOJA/MILHO: ~Oct–Dec) |
| **planting_doy_sin/cos** | Cyclical encoding of DOY | Ensures Jan 1 ≈ Dec 31 in feature space |
| **area_hectares** | Geodesic area of polygon | Field size correlates with crop type in Brazil |

### 4.4 Null Indicators

With ~9–15% null rate from cloud cover, binary `_is_null` columns are added for features with >10% missing values. This lets the model distinguish "low NDVI" from "no data available" — different meanings that raw NaN handling obscures.

---

## 5. Model Evolution

### 5.1 Architecture Progression

The model evolved through four iterations, each addressing bottlenecks identified in the previous version:

**v1 — Baseline XGBoost** (`train_xgboost.py`)
- Single XGBoost classifier with manual hyperparameters
- 91 raw features (no engineering)
- 5-fold stratified cross-validation

**v2 — Feature Engineering + Optuna** (`train_xgboost_v2.py`)
- Added 63 engineered features (deltas, peaks, ratios, etc.)
- Optuna hyperparameter search (80 trials)
- Same single XGBoost architecture
- `min_stages >= 3` filter to remove low-quality samples

**v3 — Ensemble + Feature Selection + Geographic** (`train_xgboost_v3.py`)
- **Ensemble:** XGBoost + ExtraTreesClassifier with soft voting
  - XGBoost: gradient boosting — strong on structured/tabular data
  - ExtraTrees: random splits — decorrelated from XGBoost, catches different patterns
  - Soft voting: averages predicted probabilities, reduces variance
- **Feature selection:** trains a quick XGBoost, drops bottom 40% features by gain, retrains on clean set (270 → 162 features)
- **Geographic features:** latitude, longitude, planting_doy (sin/cos encoded)
- **Null indicators:** binary flags for high-null columns
- Both models independently tuned by Optuna (80 XGB + 40 ET trials)
- Best model auto-selected by CV F1 macro

### 5.2 Why Not Random Forest?

The initial exploration used Random Forest, but XGBoost consistently outperformed it because:

1. **Boosting vs bagging:** XGBoost builds trees sequentially, each correcting the errors of the previous. Random Forest builds independent trees and averages. For tabular data with complex interactions (phenological stages × indices × geography), boosting captures more signal.
2. **Missing value handling:** XGBoost natively learns optimal split directions for NaN values (our 9–15% null rate). Random Forest requires imputation, which introduces noise.
3. **Regularization:** XGBoost's L1/L2 regularization (reg_alpha, reg_lambda) prevents overfitting on the 160+ features. Random Forest relies only on max_depth and min_samples.

ExtraTrees was added back to the ensemble (not Random Forest) because it uses **random split thresholds** rather than optimal splits, making it maximally decorrelated from XGBoost — better for ensemble diversity.

### 5.3 Final Hyperparameters (v3)

**XGBoost** (Optuna-tuned, 80 trials):
| Parameter | Value |
|---|---|
| n_estimators | 823 |
| max_depth | 9 |
| learning_rate | 0.014 |
| subsample | 0.83 |
| colsample_bytree | 0.88 |
| min_child_weight | 9 |
| gamma | 0.18 |
| reg_alpha | 0.20 |
| reg_lambda | 0.84 |

**ExtraTrees** (Optuna-tuned, 40 trials):
| Parameter | Value |
|---|---|
| n_estimators | 807 |
| max_depth | 19 |
| min_samples_split | 7 |
| min_samples_leaf | 1 |
| max_features | 0.90 |

---

## 6. Results Progression

### 6.1 Overall Metrics

| Version | Samples | Features | Accuracy | F1 macro | Key Change |
|---|---|---|---|---|---|
| v1 baseline (50 KML/crop) | ~350 | 91 | 45.4% | 0.451 | Raw features, manual params |
| v2 (50 KML + feat eng + Optuna) | ~350 | 154 | 54.1% | 0.519 | +63 engineered features |
| v1 baseline (500 KML/crop) | 3500 | 91 | 56.6% | 0.565 | 10x more data |
| v2 (500 KML + feat eng + Optuna) | 3407 | 154 | 59.9% | 0.598 | Engineering + tuning at scale |
| **v3 (ensemble + feat sel + planting_doy)** | 3407 | 162 | **80.0%** | **0.800** | Ensemble, planting date, null indicators |
| **v3 + lat/lon** | 3407 | 162 | **89.8%** | **0.898** | Geographic coordinates |

### 6.2 Per-Class F1 Score Progression

| Crop | v1 (500 KML) | v2 (500 KML) | v3 | v3 + lat/lon |
|---|---|---|---|---|
| ARROZ | 0.672 | 0.685 | 0.807 | **0.957** |
| AVEIA | 0.496 | 0.537 | 0.733 | 0.780 |
| CAFE | 0.721 | 0.748 | 0.873 | **0.979** |
| FEIJAO | 0.465 | 0.517 | 0.876 | **0.946** |
| MILHO | 0.643 | 0.635 | 0.808 | **0.917** |
| SOJA | 0.505 | 0.564 | 0.796 | **0.942** |
| TRIGO | 0.452 | 0.497 | 0.707 | 0.767 |

### 6.3 What Drove Each Jump

**v1 → v2 (+8.7pp):** Feature engineering — stage deltas and peak detection gave the model temporal shape information instead of raw snapshots.

**v2 → v3 (+20.1pp):** Three changes stacked:
- `planting_doy` (#1 feature by gain) — separated winter vs summer crops immediately
- Ensemble (XGB + ET) — reduced variance on hard pairs
- Feature selection — removing 40% noise features improved generalization

**v3 → v3+latlon (+9.8pp):** Geographic coordinates. CAFE is grown in Minas Gerais (lat ~−20), far from the southern grain belt (lat ~−28). Latitude alone nearly solved CAFE classification (97.9% F1). Also improved ARROZ (concentrated in RS) and SOJA (spread across center-south).

### 6.4 Remaining Challenge: TRIGO vs AVEIA

The confusion matrix shows the dominant remaining error:
- 24% of true TRIGO predicted as AVEIA
- 20% of true AVEIA predicted as TRIGO

These winter cereals share:
- Same planting window (May–June)
- Same geographic region (Parana/Rio Grande do Sul)
- Nearly identical NDVI/EVI curves
- Similar field sizes

This is why SAR is being added — radar backscatter can detect the structural differences (wheat: taller, denser heads; oats: shorter, more open canopy) that optical sensors cannot see.

---

## 7. Feature Importance Analysis

### 7.1 Top 10 Features by Gain (v3 + lat/lon)

| Rank | Feature | Gain | Category |
|---|---|---|---|
| 1 | planting_doy | ~40 | Calendar |
| 2 | planting_doy_cos | ~35 | Calendar |
| 3 | latitude | ~22 | Geographic |
| 4 | longitude | ~18 | Geographic |
| 5 | planting_doy_sin | ~14 | Calendar |
| 6 | NDVI_peak_stage | ~8 | Peak/amplitude |
| 7 | EVI_peak_stage | ~7 | Peak/amplitude |
| 8 | NDWI_std_grain_fill | ~6 | Base feature |
| 9 | EVI_p90_grain_fill | ~6 | Base feature |
| 10 | area_hectares | ~5 | Geographic |

### 7.2 Importance by Category (total gain)

| Category | Total Gain | Role |
|---|---|---|
| Base features (raw stats) | ~163 | Foundation — direct spectral/water measurements |
| Planting date | ~68 | Season separator — winter vs summer crops |
| Stage deltas | ~54 | Temporal dynamics — growth/decline rates |
| Other (lat/lon, area) | ~34 | Geographic context |
| Peak/amplitude | ~32 | Growth curve shape |
| Cross-index ratios | ~11 | Multi-index relationships |
| Greenup/senescence | ~8 | Growth speed characterization |
| Temporal CV | ~6 | Overall variability signature |
| Late-stage divergence | ~5 | End-of-season differentiation |
| Early/late ratio | ~4 | Full-cycle shape comparison |
| Cumulative | ~2 | Total integrated index |

### 7.3 Key Insights

1. **Calendar + geographic features contribute ~40% of total importance**, despite being only 5 of 162 features. Knowing *when* and *where* a crop is planted is as informative as *what it looks like* spectrally.

2. **Base features still dominate** at ~163 total gain. The raw spectral measurements remain the foundation — derived features refine the signal but don't replace it.

3. **Stage deltas are the most valuable engineering category** (~54 gain). The *rate of change* between phenological stages contains more discriminative power than absolute values at any single stage.

4. **grain_fill and maturity stage features appear disproportionately** in the top 30. Late-season divergence is where crops differentiate most — by flowering, most crops look "green"; it's how they ripen and dry that separates them.

---

## 8. Next Steps: SAR Integration

The pipeline v2 (`phenology_feature_pipeline_v2.py`) adds Sentinel-1 SAR data alongside optical. Expected impact:

| Expected Benefit | Mechanism |
|---|---|
| TRIGO vs AVEIA separation | Different canopy structure → different VH backscatter |
| Reduced null rate | SAR penetrates clouds — fills gaps in optical coverage |
| Complementary signal | Structure (SAR) + color (optical) = more robust classification |

**SAR backfill for existing rows:** 3500 fields × 6 stages = 21,000 API calls. Only SAR is requested — existing optical data stays untouched. The migration is crash-safe and resumable.

**Target:** With SAR features added, the model should push TRIGO/AVEIA F1 from ~0.77 toward 0.85+, bringing overall accuracy above 92%.
