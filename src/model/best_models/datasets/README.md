# datasets/ — what's in here, what we're doing, and what comes next

**Last updated:** 2026-07-08

## What this folder is

The data side of the crop-classifier project: byte-exact snapshots of what the promoted models
trained on, plus the raw KML supply and analysis for the next model generation.

| Item | What it is |
|---|---|
| `features_v8.db` / `features_v8_test.db` | Frozen train/test snapshots of the **currently promoted** models (checksums in `MANIFEST.json`) |
| `MANIFEST.json` | Provenance: checksums, crop counts, bbox/area/date stats for the v8 snapshots |
| `DATASET_ANALYSIS.md` | Per-crop deep dive: counts, regions, areas, dates — v8 snapshot **and** the new merged pool |
| `kml_sicor_2025/` (242k) / `kml_sicor_2026/` (66k) | Raw KML supply exported from the SICOR rural-credit registry by `extract_kml_sicor.py` |
| `../kml_test_sample_250.zip` | The 250-field production benchmark — **never train on these** |

Related working data outside this folder: `src/data/features_v9/` (the new extraction),
`src/data/features_v8v9_train/` (merged training DB), `src/data/kml_train_v9/` (staged KMLs +
sampling manifest).

## What we are doing right now (the v9 retrain)

The story so far, in one paragraph: the promoted 5-crop model scores 0.983 on its in-year held-out
test but dropped to **84.3%** in the first production field test. Analysis of the actual 250 test
KMLs showed the cause is a **planting-year shift** (test fields 65% 2024; training 99.9% 2025) —
not region, not mid-season truncation, not the serving path (parity verified). Full trail:
`../../FIELD_TEST_5CROP_ANALYSIS_AND_PLAN.md`.

The response (in flight):

1. ✅ **Generated new KML supply** from SICOR (308k fields across safras 2025 + 2026).
2. ✅ **Sampled 16,757 fields** (`scripts/sample_kml_v9.py`, seed 42) into targeted cohorts:
   safrinha SOJA/MILHO 2025, all scarce ARROZ/FEIJAO, the 2025/26 spring season, 2026 summer
   crops (first second-year data), 2024-planted CAFE. Hard-excluded: everything in the v8
   snapshots and the 250-KML benchmark (74 benchmark ids were lurking in the SICOR pool).
3. ✅ **Extracted `features_v9`**: 16,553 clean rows, 0 dead rows (one Sentinel Hub outage
   mid-run; damaged rows deleted and re-extracted).
4. ✅ **Merged v8+v9** → 22,727-field training pool (safrinha SOJA 32 → 1,176).
5. 🔄 **Training sweep running**: 5-crop XGBoost on the merged pool with the safrinha up-weight
   swept **1× / 2× / 6×** — the hypothesis is that real safrinha data replaces the 6× weight that
   made SOJA the model's default guess under distribution shift (production SOJA precision 0.67).
6. 🔄 **Benchmark extraction running**: the 250 field-test KMLs are being extracted locally so
   every candidate model is scored on the exact fields production failed on.

**Promotion gate for the new model:** beats 84.3% on the 250-KML benchmark (target ≥0.95 covered
accuracy), no regression on `features_v8_test`, safrinha recall ≥0.85, SOJA precision recovered.

## Planned future stages

| Stage | What | When / trigger |
|---|---|---|
| **1. Evaluate + promote** | Score the three sweep runs on the benchmark; pick the best weight; recalibrate the abstain gate (current 0.30 threshold is a no-op — sweep to covered-acc ≥0.95, plus a stricter SOJA-specific threshold); promote to `best_models/` with new MANIFEST | as soon as the sweep finishes |
| **2. The 2024 crop-year batch** | The measured failure year is still barely in training (~790 rows, mostly CAFE). Extract ~5k stratified 2024 fields from `culturas/` (SICOR doesn't have them), retrain, run **leave-one-year-out** both directions | after stage 1 |
| **3. December 2026 batch** | 2026 winter TRIGO (11.7k) + AVEIA (1.1k) are mid-season now; extract once harvested → first AVEIA/TRIGO year diversity. Also re-extract the ~2,700 truncated-tail summer-2026 fields to complete their grids | ~Dec 2026 |
| **4. Out-of-region expansion** | Geography is still Sul-dominated everywhere (SICOR didn't change it). Pull the Cerrado/Matopiba/SP-MS tail from `culturas/` (~37% of its pool), add **leave-one-region-out** eval | after stage 2 |
| **5. Serving hardening** | Per-request logging (coverage, probabilities), season-completeness guard (`SAFRA_EM_ANDAMENTO` instead of a low-information guess), truncation-augmentation training validated on the real mid-season 2026 fields, extraction-parity regression test in CI | interleaved; details in `FIELD_TEST_5CROP_ANALYSIS_AND_PLAN.md` Phases 1/4 |
| **6. Standing eval discipline** | Every promotion reports: in-year test + 250-KML benchmark + 2026 next-year benchmark + safrinha/season diagnostics. Single-split accuracy alone is never again the promotion criterion | permanent |

## Companion docs

- `../../FIELD_TEST_5CROP_ANALYSIS_AND_PLAN.md` — production-gap root cause + phased recovery plan
- `../../SICOR_DATA_PLAN.md` — what each SICOR dataset is good for (and the contamination trap)
- `../../STATUS_AND_ROADMAP.md` — overall project status and obstacle list
- `../../PROJECT_SUMMARY.md` — one-page project overview
- `DATASET_ANALYSIS.md` — the data itself, per crop
