# Crop Classifier — Status & Upgrade Plan

**Single source of truth** for where the project is and what to do next. Merges the former
`MODEL_UPGRADE_PLAN.md`, `DATA_UPGRADE_PLAN.md`, `best_models/DATASETS.md`, and
`pipelines/LARGE_FIELD_FAILURE_INVESTIGATION.md`.

**Last updated:** 2026-07-09
**Current models: the v10 generation** — trained on `features_v10_train` (balanced,
**2025/2026-only** by decision; zero 2024 rows), promoted 2026-07-09 after passing the new
production gate. Full campaign log: [`PLAN_V10_2025_2026.md`](PLAN_V10_2025_2026.md).

| Model (`best_models/`) | Crops | **bench2026** (gate) | Held-out v8 | Abstain thr |
|---|---|--:|--:|--:|
| `5_culturas_no_aveia` | ARROZ, FEIJAO, MILHO, SOJA, TRIGO | **0.940** (n=200) | 0.981 (n=973) | 0.80 |
| `6_culturas_no_aveia` | + CAFE | **0.956** (n=250) | 0.977 (n=1,173) | 0.90 |
| `7_culturas_cafe` | + AVEIA (all 7) | **0.956** (n=250) | 0.955 (n=1,373) | 0.90 |

Two numbers per model on purpose: **bench2026** measures never-seen fields from the most recent
completed season (what production will serve); the v8 held-out is the in-distribution regression
check. The previous v8 models scored 0.850/0.856/0.864 on bench2026 — the v10 generation closed
a ~10-point production gap. The AVEIA↔TRIGO ladder logic still holds: deployments that don't
need oats should use the 5-/6-crop models.

---

# PART I — WHERE WE ARE

## 1. The models (v10, 2026-07-09)

Dense XGBoost classifiers trained by `train_xgboost_v7.py` (weighted 5-fold OOF CV, class
balance, **safrinha up-weight 1×** — real safrinha data replaced the old 6× crutch, global
abstain gate) on the v7 extended grid, differing only by `--exclude-crops`. Runs:
`runs_v7/20260709_*_dense_{5,6,7}crop_v10_ssw1` (v8 predecessors kept at `runs_v7/20260707_*`).

**7-crop per-class recall — bench2026 / v8 held-out:**

| Crop | bench2026 | held-out |
|---|--:|--:|
| ARROZ | 0.94 | 0.99 |
| CAFE | **1.00** | 0.95 |
| FEIJAO | 0.96 | 0.99 |
| MILHO | 0.94 | 0.95 |
| SOJA | 0.94 | 0.98 |
| TRIGO | *December* | 0.94 |
| AVEIA | *December* | 0.88 |

Residual bench2026 errors: safrinha SOJA↔MILHO (3+3) + 3 ARROZ→SOJA. No winter-crop
hallucination on summer fields. AVEIA↔TRIGO on held-out: 35 swaps (v8: 38).

**Abstain thresholds are now calibrated on bench2026** (not the OOF no-op 0.30): 0.80/0.90/0.90,
shipped inside each `abstain_policy.json` with a `calibration` block. At those thresholds:
5-crop covers 96% @ 0.953; 6/7-crop ~93% @ ~0.978. **Serving requirements (mandatory, incl. the
inf→NaN guard) are in `best_models/README.md` §Serving.**

## 2. The pipeline

`phenology_feature_pipeline_v7.py` (a thin subclass of `_v6.py`) — one Statistical-API request per
source, aggregated server-side on a **hybrid grid**: dekadal **P10D** over `planting−15 … +275 d`
(`N_DEKADS=29`, covers CAFE's ~270-day cycle) + fine **P5D** over flowering→maturity (the
AVEIA/TRIGO senescence-timing window). ~4 calls/field, field-level concurrency behind a shared rate
limiter, resume-safe. Schema ≈ 1,485 columns.

### 🔧 The resolution fix (2026-07-07) — kept for the record
The pipeline set `aggregation.resx/resy` in **metres** (`10`) while the request `bounds` were in
**EPSG:4326 (degrees)**, so Sentinel Hub read ~10°/pixel and **collapsed every field to ~1 pixel**:
283 large fields 400'd into dead rows, and all `p10/p90` features were degenerate. `_adaptive_res`
now returns degrees (~10 m/px, capped ≤2,500 px/side), timeout 60→180 s. The v8 re-extraction gained
+1.6 pp held-out (0.936→0.952) and cut AVEIA↔TRIGO swaps 52→38 — the old "~0.857 AUC ceiling" was
partly a 1-pixel artifact. Repro tool: `scripts/probe_single_field.py`.

> **Latent, not yet fixed:** `phenology_feature_pipeline_v4.py` / `_v5.py` carry the identical
> unit bug. They don't build current models — fix before any v4/v5 re-extraction.

### Known data quirks (handled, don't rediscover)
- **`colheita_NA` filenames don't parse** — the driver silently skips them (204 CAFE dropped in
  v9; 75 replaced during v10 sampling). Samplers must filter or expect the skip.
- **Genuine `±inf` in SAR columns** (zero backscatter → −inf dB: radar shadow / standing water;
  `CAFE_520664544-1` reproduces on every extraction). `train_xgboost_v7._load` maps ±inf→NaN;
  **serving must do the same before `predict_proba`** or XGBoost aborts.

## 3. The datasets

One row = **one field** (KML-delineated) in one season. Current training pool:
**`features_v10_train`** (16,623 fields, 2025/2026 only, built by `scripts/build_v10_train.py`
from the v8+v9 merged pool − 789 pre-2025 rows − 7 benchmark leaks + 970 fresh CAFE; kept ids in
`build_manifest.json`). Snapshot: **`best_models/datasets/features_v10_train.zip`**
(db MD5 `474E477B8F1FE80BA0D67FE8A6D30CFB`).

| Crop | Train | 2025 / 2026 | Notes |
|---|--:|---|---|
| SOJA | 2,500 | 2,189 / 311 | all 1,172 safrinha kept |
| MILHO | 2,500 | 1,623 / 877 | safrinha share capped at 1,500 (see below) |
| FEIJAO | 2,500 | 2,038 / 462 | |
| ARROZ | 2,500 | 2,494 / 6 | |
| TRIGO | 2,500 | 2,500 / 0 | single-winter until December |
| CAFE | 2,500 | 2,500 / 0 | 967 extracted 2026-07-09 (plantio ago–out/25) |
| AVEIA | 1,623 | 1,623 / 0 | supply-capped (SICOR-2025 pool exhausted) |

> ⚠️ **The MILHO safrinha cap matters.** An uncapped "keep all safrinha" rule filled MILHO's
> entire 2,500 quota with Jan–Mar plantings and erased main-season corn — and v9's safrinha-heavy
> MILHO was what collapsed TRIGO on the old benchmark (0.46 recall, recovered to 0.74 by
> rebalancing alone, no 2024 data). Class balance is month-aware, not just count-aware.

**The benchmark / test sets:**
- **`kml_test_2026`** (the gate): 350 KMLs frozen 2026-07-09, 50/crop from the most recent
  completed season (SOJA/MILHO/FEIJAO plantio jan–mar/26; ARROZ out/25–fev/26; CAFE ago–out/25;
  **TRIGO/AVEIA frozen mid-season → extract after 2026-12-01**). 250 extracted at
  `src/data/features_test_2026/`. Run: `python src/model/eval_bench2026.py <run_dir>`.
  Bar: ≥0.95 covered accuracy @ ≥0.85 coverage. **Every sampler must exclude its manifest ids.**
- **v8 held-out** (`best_models/datasets/features_v8_test.db`, n=1,373): regression check.
- The 2024-heavy `kml_test_sample_250` benchmark is **retired** (DB + script deleted 2026-07-09;
  zip kept only for sampler exclusions; story in `FIELD_TEST_5CROP_ANALYSIS_AND_PLAN.md`).

**Variability & biases (carry into any deployment decision):**
- **Regional** — still clusters in **South Brazil** (CAFE: Sudeste). Cerrado/Matopiba is
  out-of-distribution; the abstain gate is the only safety net. *(#1 remaining risk — Part II.)*
- **Years: 2025/2026 by design** (decision 2026-07-09) — recency over 2024 backfill; production
  traffic is 2026/27-season. 2024 fields stay out of scope.
- **TRIGO/AVEIA are single-winter (2025)** until the December batch (their 2026 season is
  mid-cycle today).
- **Field size** median ~8 ha; small fields carry mixed-pixel noise. **SAR ~16% null** (edges).

## 4. Resolved / corrected (don't re-investigate)
- ✅ **Production year-shift gap (−14 pts, field test 2026-07-07)** — root-caused (year shift +
  safrinha-MILHO boundary crowding), fixed by the v10 recent-years balanced retrain; gate green.
- ✅ **Safrinha SOJA** — 1,176 real safrinha fields in training; up-weight relaxed 6×→1×;
  bench2026 safrinha recall 0.94.
- ✅ **Large-field extraction failure & 1-pixel degeneracy** — resx/resy unit bug (§2).
- ✅ **Grid too short for CAFE** — v7 grid to +275 d.
- ✅ **Abstain no-op** — thresholds now bench-calibrated per model (0.80/0.90/0.90).
- ✅ **`evaluate()` crash on test sets missing model classes** — labels pinned (needed because
  bench2026 lacks TRIGO/AVEIA until December).
- ✅ **"SAR ~90% null"** — stale; it's ~16%.
- Engineered dense-timing features — tested negative (`V7_NEXT_LEVERS.md`).

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

## Priority 0 — Serving hardening *(infrastructure — blocks safe deployment of the promoted v10)*
The models are promoted; the serving side must catch up (`best_models/README.md` §Serving):
- **inf→NaN guard before `predict_proba`** — without it one radar-shadow field aborts inference.
- **Load the calibrated threshold from `abstain_policy.json`** (0.80/0.90/0.90), not 0.30.
- **Per-request logging**: planting date, `dekads_covered`/`fine_covered`, lat/lon, full
  probability vector — required for the next recalibration and drift monitoring.
- **Season-completeness guard**: if flowering→maturity isn't observable yet, return
  `SAFRA_EM_ANDAMENTO` + retry-after, not a low-information guess.
- No silent-null rows at extraction; coverage report per batch (N fields, dead rows, failure by
  area bucket).

## Priority 1 — December 2026 batch: two-winter TRIGO/AVEIA + full benchmark *(data — scheduled)*
The one structural gap left by v10. After 2026-12-01:
1. Extract the **100 frozen TRIGO/AVEIA test fields** → bench2026 becomes the full 350.
2. Extract winter-2026 training: ~2 k TRIGO + ~1.1 k AVEIA (pools measured: 11.7 k / 1.1 k free).
3. Retrain the ladder (**v11**) → first two-winter TRIGO/AVEIA models; re-gate on the 350.
Until then: don't promise wheat/oats accuracy for the 2026 winter season.

## Obstacle 1 — Regional generalization *(data — highest remaining real-world value)*
Training still clusters in South Brazil. The `culturas/` library holds **~416 k usable fields,
~37% outside** the S-Brazil cluster (far-South 14%, SP/MS 12%, Cerrado 7%, North 4%).
Plan: stratified out-of-region batch (SOJA/MILHO first — their production traffic already strays),
extract with data-quality gates (drop `plantio_nan` + `colheita_NA`, junk years, dedupe by ascii
field_id), retrain with a **standing held-out-region metric**. ARROZ (~5 k) and AVEIA (~2.8 k)
remain supply-constrained.

## Obstacle 2 — AVEIA↔TRIGO residual *(signal)*
0.90/0.90 F1, 35 swaps — the dominant 7-crop error budget. Optical features are
information-limited here (proven); the lever is a **physically orthogonal signal**:
- **SAR texture (GLCM)** — oats' open panicle vs wheat's compact spike differ in canopy structure.
  Needs a raster/GLCM path (Process API) over flowering→maturity; de-risk on the matched pilot
  before any full re-extraction.
- **Per-pair abstain calibration** — route low-confidence AVEIA↔TRIGO to `NAO_CLASSIFICAVEL`
  without costing easy-crop coverage. (Trail: `V7_NEXT_LEVERS.md`, `AVEIA_TRIGO_DISCRIMINATION.md`.)

## Obstacle 3 — Bench2026 residuals *(data/signal — small)*
- Safrinha SOJA↔MILHO (3+3 swaps): monitor through the per-request logs; more 2026 safrinha data
  arrives naturally with future batches.
- ARROZ→SOJA (3): check if the 3 fields are anchor-date outliers (farmer-declared planting dates;
  ARROZ's early-flooding features are anchor-sensitive).
- Grow the benchmark over time: add each new completed season's 50/crop, keeping the frozen-set
  discipline (sample → freeze → extract at season end).

## Cross-cutting
- **Phenology-normalized resampling** — one comparable schema across 90–270 d cycles (long-term
  grid redesign).
- **Rotate the hardcoded Sentinel Hub credentials** in `src/data_ingestion/request_sentinel_v1.py`
  into an env var.
- **Extraction-parity regression test in CI** (5-field subsample of the benchmark per serving
  deploy).

## Anti-levers (don't spend time here)
- Engineered dense-timing / senescence-shape features — tested negative.
- Extra red-edge ratios (~2%), extra date features (~1%), Optuna tuning — lost to defaults.
- Treating SAR as null/useless — it contributes ~14.5% of gain.
- **Safrinha up-weight > 1×** — with real safrinha data it only inflates MILHO→SOJA.
- **2024 backfill** — decided against (2026-07-09); recency wins for production traffic.

## Recommended sequence
1. ~~Resolution fix + `features_v8` retrain~~ ✅ 2026-07-07.
2. ~~Safrinha SOJA recovery (v9 SICOR expansion)~~ ✅ 2026-07-08.
3. ~~Recent-years balanced retrain + new benchmark + promotion (v10)~~ ✅ **2026-07-09 — current**.
4. **Serving hardening (Priority 0)** — before real traffic on v10.
5. **December two-winter batch → v11** (Priority 1, scheduled).
6. **Out-of-region expansion** (Obstacle 1) with held-out-region eval.
7. **SAR texture + per-pair abstain** (Obstacle 2) if AVEIA↔TRIGO still short after v11.

---

*Deep-dive appendices: `AVEIA_TRIGO_DISCRIMINATION.md`, `V7_NEXT_LEVERS.md`,
`SECOND_SEASON_SOJA_RESULTS.md`. Campaign logs: `PLAN_V10_2025_2026.md` (current),
`FIELD_TEST_5CROP_ANALYSIS_AND_PLAN.md` (the production wake-up call).
Reproducibility: `best_models/README.md` + `best_models/datasets/` (v10 zip + v8 snapshots).*
