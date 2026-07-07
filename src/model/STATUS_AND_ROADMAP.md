# Crop Classifier — Status & Upgrade Plan

**Single source of truth** for where the project is and what to do next. Merges the former
`MODEL_UPGRADE_PLAN.md`, `DATA_UPGRADE_PLAN.md`, `best_models/DATASETS.md`, and
`pipelines/LARGE_FIELD_FAILURE_INVESTIGATION.md`.

**Last updated:** 2026-07-07
**Current models (all on `features_v8`, the resolution-fixed re-extraction):**

| Model (`best_models/`) | Crops | Held-out test acc / macro-F1 | n |
|---|---|---|---|
| `5_culturas_no_aveia` | ARROZ, FEIJAO, MILHO, SOJA, TRIGO | **0.983 / 0.982** | 973 |
| `6_culturas_no_aveia` | + CAFE | **0.979 / 0.978** | 1,173 |
| `7_culturas_cafe` | + AVEIA (all 7) | **0.952 / 0.952** | 1,373 |

The ladder is deliberate: **AVEIA is the accuracy-dragging crop** (its confusion with TRIGO is the
whole residual error budget), so the 5- and 6-crop models drop it and land at ~0.98. Deployments
that don't need oats should use those.

---

# PART I — WHERE WE ARE

## 1. The models

All three are dense XGBoost classifiers trained by `train_xgboost_v7.py` (weighted 5-fold OOF CV,
class balance, safrinha-SOJA up-weight, global abstain gate, season-stratified diagnostic) on the
**v7 extended grid** (`features_v8`), differing only by which crops are excluded at load.

**7-crop held-out per-class (test, n=1,373):**

| Crop | P | R | F1 |
|---|---|---|---|
| ARROZ | 0.99 | 0.98 | 0.99 |
| CAFE | 0.98 | 0.97 | 0.97 |
| FEIJAO | 0.94 | 1.00 | 0.97 |
| SOJA | 0.99 | 0.96 | 0.97 |
| MILHO | 0.96 | 0.96 | 0.96 |
| AVEIA | 0.89 | 0.92 | **0.90** |
| TRIGO | 0.92 | 0.89 | **0.90** |
| **Overall** | | | **0.952** |

In the 5-/6-crop models (no AVEIA) **TRIGO reaches F1 1.00** — removing its only confuser eliminates
the pair. Abstain gate: all three keep ≥99.97 % coverage at threshold 0.30 (covered-accuracy ≈ OOF).
Each model folder holds the model + `selected_features.json` + `abstain_policy.json` + `metrics.json`
+ `test_eval/` + plots; **a model is bound to its `selected_features.json` — extract inference
features with `phenology_feature_pipeline_v7.py` and reindex to that list.**

## 2. The pipeline

`phenology_feature_pipeline_v7.py` (a thin subclass of `_v6.py`) — one Statistical-API request per
source, aggregated server-side on a **hybrid grid**: dekadal **P10D** over `planting−15 … +275 d`
(`N_DEKADS=29`, covers CAFE's ~270-day cycle) + fine **P5D** over flowering→maturity (the
AVEIA/TRIGO senescence-timing window). ~4 calls/field, field-level concurrency behind a shared rate
limiter, resume-safe. Schema ≈ 1,485 columns.

### 🔧 The resolution fix (2026-07-07) — the most recent and highest-impact change
The pipeline set `aggregation.resx/resy` in **metres** (`10`) while the request `bounds` were in
**EPSG:4326 (degrees)**, so Sentinel Hub read ~10°/pixel and **collapsed every field to ~1 pixel**.
Two consequences, both now fixed:
- **283 large fields (≥50 ha) 400'd** with *"request of N m/px exceeds the 1,500 m/px limit"* and were
  silently written as `dekads_covered=0` rows (dropped at train, **unservable** at inference).
- **All `p10/p90` features were degenerate** (`p10==p90==mean` for ~99 % of optical, ~88 % of SAR
  bins) — a third of the schema carried no within-field information.

`_adaptive_res` now returns `(resx, resy)` **in degrees** (~10 m/px per-axis, capped ≤2,500 px/side);
the Stats-API timeout was raised 60→180 s (largest fields take ~87 s). Verified live: all seed 400s
became HTTP 200 with full bins. Full set re-extracted into **`features_v8` / `features_v8_test`**
(0 dead fields, 99.5 % multi-pixel), and the 7-crop retrain gained **+1.6 pp held-out acc (0.936→0.952)
with AVEIA↔TRIGO swaps 52→38 (−27 %)** — so the long-standing "~0.857 AUC ceiling" on that pair was
**partly a 1-pixel artifact**, not a hard limit. Repro tool: `scripts/probe_single_field.py`.

> **Latent, not yet fixed:** `phenology_feature_pipeline_v4.py` and `_v5.py` carry the identical
> `resx/resy:10`-with-degree-bounds bug. They don't build the current models — fix before any v4/v5
> re-extraction.

## 3. The datasets (`features_v8`)

One row = **one field** (a KML-delineated plot) in one season, with a crop label, planting date, and
a full dense satellite time series. Train/test are split at the **field level** (curated
`src/data/dataset_split/{train,test}/`); the same field never appears in both. Byte-exact snapshots +
checksums live in `best_models/datasets/` (`features_v8.db`, `features_v8_test.db`, `MANIFEST.json`).

**Schema:** bookkeeping columns (`field_id, crop_label, planting_date, area_hectares, latitude,
longitude, dekads_covered, fine_covered, interpolated`) + time-binned satellite stats named
`<SIGNAL>_<mean|p10|p90>_<d{k}|f{k}>`. **8 optical indices** (S2: NDVI, NDWI, EVI, NDRE, CIRE, MTCI,
PSRI, NDMI) + **4 SAR channels** (S1: VV, VH, CR, RVI). The trainer appends null-indicators + cyclic
planting-date features.

**Composition (train `features_v8`, 6,174 fields; test 1,373):**

| Crop | Train | Test |
|---|--:|--:|
| AVEIA | 1,200 | 200 |
| MILHO | 1,200 | 182 |
| FEIJAO | 1,016 | 175 |
| TRIGO | 1,001 | 200 |
| ARROZ | 637 | 200 |
| SOJA | 633 | 216 |
| CAFE | 487 | 200 |

**Variability & biases (carry into any deployment decision):**
- **Regional** — fields cluster tightly in **South Brazil** (lat −25…−28°, Paraná/SC/RS); a thin tail
  reaches the North. A field in the Cerrado/Matopiba is out-of-distribution; the abstain gate is the
  only safety net. *(This is the #1 real-world-accuracy risk — see Part II.)*
- **Single crop-year** — training is essentially **2025**; inter-annual robustness is unproven.
  Planting **month** is well spread (Feb + Jun peaks), so intra-year phenology is well represented.
- **Field size** — median ~8 ha, up to 1,826 ha; small fields carry more mixed-pixel noise.
- **Class imbalance** ~2.5× (handled by `class_balance` weights). AVEIA/MILHO capped at 1,200.
- **SAR ~16 % null** (concentrated at season edges d0/d28) — *not* the "90 %" claimed in older docs;
  SAR levels already contribute ~14.5 % of 7-crop gain. What's untapped is SAR **texture** (GLCM).

## 4. Resolved / corrected (don't re-investigate)
- ✅ **Large-field extraction failure & 1-pixel degeneracy** — root-caused to the resx/resy unit bug,
  fixed, re-extracted (§2).
- ✅ **Grid too short for CAFE** — v7 extends the dekadal grid to +275 d; all crops re-extracted
  uniformly (annuals carry real post-harvest late-dekad data → no train/serve skew, no
  "late-bins ⇒ CAFE" shortcut, verified).
- ✅ **"SAR ~90 % null"** — stale; it's ~16 % (§3).
- **Engineered dense-timing features** — tested negative; raw dense bins already carry the senescence
  timing. See `AVEIA_TRIGO_DISCRIMINATION.md`, `V7_NEXT_LEVERS.md`.

## 5. The 7-crop universe

| Crop | Type | Cycle (d) | Discriminating signal / risk |
|---|---|---|---|
| SOJA | C3 legume, annual | ~130 | safrinha vs MILHO |
| MILHO | C4 grass, annual | ~140 | C4 vs C3 (easy) |
| FEIJAO | C3 legume, annual | ~90 | short cycle (easy) |
| AVEIA | C3 cereal, annual | ~130 | **vs TRIGO — senescence timing / canopy texture** |
| TRIGO | C3 cereal, annual | ~135 | **vs AVEIA** |
| ARROZ | C3 grass, flooded | ~145 | flooding (high early NDMI/NDWI + low SAR) |
| CAFE | perennial, evergreen | ~270 | no annual senescence cycle — separates cleanly |

---

# PART II — UPGRADE PLAN (future tasks)

Ranked by impact. Each says whether the lever is **data**, **signal**, or **infrastructure**.

## Obstacle 1 — Generalization: single-year, single-region *(data — highest real-world value)*
Held-out 0.95–0.98 is honest but measured **within one crop-year and one agro-climatic zone**. No
architecture change fixes this — only more diverse labels. **The good news: they already exist on
disk.** The `culturas/` KML library holds **~416 k usable fields** (only ~1.9 % extracted), and the
*unused* pool is exactly what we're short on:

- **Temporal:** the pool is **2024-dominant** (276 k in 2024 vs 139 k in 2025) — a full second
  crop-year, unextracted.
- **Geographic:** ~**37 % is outside** the S-Brazil cluster (far-South 14 %, SP/MS 12 %, Cerrado 7 %,
  North 4 %); CAFE's pool is largely the SP/MS + Cerrado coffee belt.

**Per-crop unused supply:** SOJA 158 k, MILHO 125 k, TRIGO 66 k, CAFE 40 k, FEIJAO 10 k — abundant;
**ARROZ 5 k and AVEIA 2.8 k are the only constrained crops** (and more AVEIA won't fix AVEIA↔TRIGO —
that's signal, not volume).

**Plan:**
1. **Data-quality gates:** drop `plantio_nan` (≈12 k CAFE files), reject junk planting years
   (2000/2099/…), de-dupe by `field_id` (crop + numeric id; note filenames are accented `CAFÉ_`/`FEIJÃO_`
   but the DB stores ASCII — the pipeline normalizes, new tooling must too).
2. **2024 pilot batch** (~5 k, stratified by `crop × region`) → extract with the fixed pipeline into a
   new DB, retrain, run **leave-one-year-out** (train 2024 / test 2025 and vice-versa).
3. **Full stratified expansion** (~2.5–3 k/crop, bounded by ARROZ/AVEIA; ~19 k total) if the pilot holds.
4. **Standing held-out-year and held-out-region metrics** in the trainer report — success = accuracy
   holds across years/regions, not a higher single-split number.

## Obstacle 2 — AVEIA↔TRIGO residual *(signal)*
Now 0.90/0.90 (swaps 38) after the resolution fix, still the dominant error budget in the 7-crop
model. Prior work proved optical features are information-limited here, so the lever is a **physically
orthogonal signal**, not a better model on the same features:
- **SAR texture (GLCM).** Oats' open panicle vs wheat's compact spike differ in canopy *structure* →
  radar backscatter texture. The old blocker (SAR null) is gone; what's missing is a **raster/GLCM
  path** (Process API) computing contrast/homogeneity/entropy over flowering→maturity. De-risk on the
  matched AVEIA/TRIGO pilot before any full re-extraction.
- **Per-pair abstain calibration** — route low-confidence AVEIA↔TRIGO to `NAO_CLASSIFICAVEL` rather
  than one global threshold, so easy crops keep coverage. (See `V7_NEXT_LEVERS.md` for the full trail.)

## Obstacle 3 — Safrinha (second-season) SOJA *(data)*
`features_v8` matched the same field set as the old v6_ext, which had only 633/777 SOJA — the ~144
missing are disproportionately second-season, so held-out safrinha recall sits at ~0.63 (OOF, n=32).
**SOJA→MILHO confusion is 0** in all three models, so the economic risk is contained. Recover the
missing SOJA KMLs from `culturas/` (v6 sourced them outside `dataset_split/train`), extract over the
v7 grid, retrain. Low effort, high confidence. Also grow the safrinha *test* set beyond n≈16.

## Obstacle 4 — Serving hardening *(infrastructure)*
- **No silent-null rows:** a 400/empty API response must be a logged failure, never a stored
  `dekads_covered=0` row (the resolution fix makes this rare, but the guard prevents recurrence).
- **Serving guard:** inference on an all-null / uncovered feature row must return `NAO_CLASSIFICAVEL`,
  never a confident guess.
- **Coverage report** after each extraction: N fields, N with `dekads_covered=0`, failure rate by
  area bucket — so a regression like the resolution bug is caught immediately.
- **Inference service** applying `selected_features.json` + model + `abstain_policy.json`; monitor
  MILHO precision and the AVEIA/TRIGO / ARROZ confusions in the field.

## Cross-cutting (independent of crop count)
- **Phenology-normalized resampling** — resample every field onto a fixed number of *phenological-time*
  steps (0–100 % of cycle) so FEIJAO (90 d), TRIGO (135 d) and CAFE (270 d) share one comparable
  schema and timing features become directly cross-crop. The principled long-term grid design (more
  work than the current fixed `N_DEKADS=29`).
- **Rotate the hardcoded Sentinel Hub credentials** in `src/data_ingestion/request_sentinel_v1.py`
  (committed `client_id`/`client_secret`) into an env var.

## Anti-levers (don't spend time here)
- Engineered dense-timing / senescence-shape features — tested negative.
- Extra cross-index red-edge ratios (~2 %), extra date features (~1 %).
- Optuna / hyperparameter tuning — lost to defaults on every sibling model.
- Treating SAR as null/useless — it contributes ~14.5 % of gain.

## Recommended sequence
1. ~~Resolution fix + re-extract (`features_v8`) + retrain 5/6/7~~ ✅ **DONE 2026-07-07** — promoted.
2. **Recover safrinha SOJA KMLs** (Obstacle 3) — smallest, high-value.
3. **Serving guards** (Obstacle 4) before any large new extraction.
4. **2024 + out-of-region data upgrade** (Obstacle 1) with held-out-year/region eval — the biggest
   real-world-accuracy win.
5. **SAR texture + per-pair abstain** (Obstacle 2) if AVEIA/TRIGO is still short of target.

---

*Deep-dive appendices (unchanged): `AVEIA_TRIGO_DISCRIMINATION.md`, `V7_NEXT_LEVERS.md`,
`SECOND_SEASON_SOJA_RESULTS.md`. Reproducibility: `best_models/README.md` +
`best_models/datasets/MANIFEST.json`.*
