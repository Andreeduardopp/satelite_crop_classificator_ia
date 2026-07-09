# 5-crop model — first production field test: analysis & upgrade plan

**Date:** 2026-07-07 (rev. 3 — after analyzing the actual 250 test KMLs in
`best_models/kml_test_sample_250.zip`; serving-path parity verified clean)
**Model:** `best_models/5_culturas_no_aveia` (features_v8, held-out 0.983 acc)
**Field test:** 250 requests (50/crop) against the remote API → **84.3% accuracy** (249 classified,
1 gated for insufficient satellite coverage; latency median 33s, p95 34s, max 98s).

## 1. The headline: a 14-point gap, and it's not noise

| | Held-out test (n=973) | Production (n=249) |
|---|--:|--:|
| Accuracy | 0.983 | 0.843 |
| ARROZ recall | 0.98 | 0.76 |
| FEIJAO recall | 0.99 | 0.82 |
| MILHO recall | 0.98 | 0.84 |
| SOJA recall | 0.96 | 0.92 |
| TRIGO recall | **1.00** | 0.88 |
| SOJA precision | 0.97 | **0.67** |

The production errors are **patterns the held-out test never showed**: SOJA over-predicted
(pulls 7 from ARROZ, 9 from FEIJAO, 6 from MILHO — 22 of the 39 errors), TRIGO→MILHO 6 swaps
(TRIGO was perfect held-out), ARROZ collapsing 0.98→0.76. New confusion pairs appearing only in
production = distribution shift or train/serve skew, not model capacity.

## 2. What the 250 test KMLs actually are (measured, not hypothesized)

Parsed all 250 filenames (`{CROP}_{id}_plantio_{date}_colheita_{date}.kml`) + polygon centroids,
and cross-checked against the training/test snapshots:

| Crop | n | Planted 2024 | Planted 2025 | Harvest complete | Region | In train snapshot |
|---|--:|--:|--:|--:|---|--:|
| ARROZ | 50 | 35 | 15 | 50/50 | Sul 50 | 6 |
| FEIJAO | 50 | 30 | 20 | 50/50 | Sul 50 | 3 |
| MILHO | 50 | 30 | 20 | 50/50 | Sul 39, NE 7, CO 2, SE 2 | 1 |
| SOJA | 50 | 36 | 14 | 50/50 | Sul 45, CO 3, other 2 | 0 |
| TRIGO | 50 | 32 | 18 | 50/50 | Sul 50 | 0 |
| **Total** | **250** | **163 (65%)** | **87 (35%)** | **250/250** | **Sul 234 (94%)** | **10** |

Three verdicts fall straight out of this:

- ❌ **Mid-season inference is NOT the cause.** Every field's harvest is complete (median ~600
  days elapsed since planting); the full feature grid was observable for ~93% of fields. My rev. 1
  prime suspect is refuted for this test. *(It remains a real risk for live traffic on
  current-season fields — kept in the plan, demoted.)*
- ❌ **Regional shift is NOT the main cause.** 94% of the fields are in the Sul cluster, exactly
  where the training data lives. Only ~16 fields (mostly MILHO-Nordeste) are out-of-region — too
  few to explain 39 errors, though they likely contribute a handful.
- ✅ **Year shift IS the dominant distribution difference.** The training DB is **99.9%
  planting-year 2025** (4,483 of 4,487 rows); the test sample is **65% planting-year 2024**. These
  are almost entirely novel fields (239/250 in neither snapshot) from a crop-year the model has
  essentially never seen. This is precisely the inter-annual generalization gap
  STATUS_AND_ROADMAP flagged as Obstacle 1 — now measured at ~14 points.

**Why a year hurts this much** (mechanisms to verify in Phase 0):
1. **Different weather = different phenology curves.** Features are planting-anchored, so
   calendar shift is absorbed, but drought/rain timing, cloud-gap patterns and growth rates are
   year-specific — and the model's 1,155 features were selected on one year's texture.
2. **Sentinel-1 availability differed.** S1 flew solo (S1A) through 2024; S1C only became
   operational in 2025. 2024 fields will have sparser SAR (~14.5% of model gain) and null-indicator
   patterns the model never trained on. Directly measurable after extracting the 250 locally.
3. **The 6× safrinha-SOJA up-weight + class balance** make SOJA the default guess wherever the
   evidence is ambiguous — which is exactly what off-year features produce. Hence SOJA precision
   0.67 while its recall stays high.

**Serving-path parity: verified clean.** The remote extractor is confirmed not to be degrading
features (checked 2026-07-07), which eliminates train/serve skew as an explanation. **Year shift
is therefore the driver of the gap**, with minor contributions from the ~16 out-of-region fields
and the 10 train-snapshot fields.

**The abstain gate did not do its job either way**: threshold 0.30 ≈ 100% coverage; zero
low-confidence abstentions among 39 errors. The OOF sweep shows threshold 0.95 keeps 93.4%
coverage at 0.997 accuracy in-distribution — huge unused headroom.

## 3. Upgrade plan

### Phase 0 — Quantify the year effect on the existing results (no new extraction needed)

*(Serving parity already verified clean — the local-reproduction experiment from rev. 2 is no
longer needed as a hypothesis splitter.)*

0.1 **Slice the 249 production results by planting year** (65/35 split gives decent n): expect
    2025 fields ≈ held-out accuracy, 2024 fields carrying nearly all the errors. This turns
    "year shift is the driver" from strong inference into a measured per-year accuracy number —
    and gives the baseline the Phase 2 retrain must beat.
0.2 **Slice the ~16 out-of-region fields** (MILHO-Nordeste etc.) separately — if most of them
    failed, they explain a fixed chunk of the 39 errors and sharpen the year-effect estimate.
0.3 **Measure SAR null-rate by year** on the served feature rows (2024 vs 2025) to confirm/kill
    the S1-availability mechanism (S1A flew solo in 2024; S1C arrived 2025).
0.4 Remove the 10 train-snapshot fields from the benchmark (mild leakage; they inflate the score).
0.5 **Make the serving side log per request**: planting date, request date, `dekads_covered`,
    `fine_covered`, lat/lon, full probability vector. Required for abstain recalibration (1.1)
    and all future monitoring.

### Phase 1 — Quick wins, no retrain (days)

1.1 **Recalibrate the abstain gate.** Sweep the threshold on the 249 production confidences (or
    the Phase 0 local reproduction); pick max coverage at covered-accuracy ≥0.95. Today's 0.30 is
    a no-op; even in-distribution, 0.90–0.95 costs <7% coverage.
1.2 **Per-class threshold for SOJA predictions** (the over-predicted class): a stricter gate on
    SOJA caps the dominant production error mode immediately, before any data fix lands.
1.3 **Season-completeness guard** for live traffic: if the flowering→maturity window isn't
    observable yet, return a dedicated status (`SAFRA_EM_ANDAMENTO` + retry-after) instead of a
    silent low-information guess. Not the cause of *this* test's errors, but a standing risk for
    real current-season requests.

### Phase 2 — Close the year gap (the main fix; ~2–3 weeks)

This is Roadmap Obstacle 1, promoted from "highest long-term value" to **proven production
blocker**. The data already exists: the `culturas/` pool is 2024-dominant (276k fields unextracted).

2.1 **2024 pilot batch** (~5k fields, stratified crop × region), extract with the fixed v7
    pipeline into a new DB.
2.2 **Retrain 5-crop on 2024+2025 combined**, with **leave-one-year-out eval** both directions
    (train-2025/test-2024 is exactly this production test, reproduced offline).
2.3 Re-run the 250-KML benchmark after retrain — it becomes the standing acceptance gate for
    every model promotion. Success = ≥0.95 covered accuracy at ≥0.85 coverage on it.
2.4 If the pilot holds, full stratified expansion (~2.5–3k/crop) per the roadmap, adding
    out-of-region MILHO/SOJA (the crops whose production samples already stray from Sul).

### Phase 3 — Fix SOJA over-prediction at the root (with the Phase 2 retrain)

3.1 **Sweep the safrinha up-weight 1×–6×** against the safrinha diagnostic *and* the 250-KML
    benchmark jointly. The 6× weight bought in-distribution safrinha recall at the price of
    making SOJA the model's fallback class under shift.
3.2 **Prefer data over weight:** recover the ~144 missing safrinha SOJA KMLs (Roadmap Obstacle 3
    — small, high-confidence) so the up-weight can be relaxed without losing safrinha recall.

### Phase 4 — Robustness for live serving (after the above)

4.1 **Truncation augmentation** (train on rows with late bins masked at day 60/90/120/150
    cutoffs) so current-season requests degrade gracefully into abstention instead of confident
    errors — the model currently trains only on complete seasons (`dekads_covered` median 29/29).
4.2 **Accuracy-vs-cutoff curve** per crop in the trainer report → sets the Phase 1.3 guard
    thresholds from measurement.
4.3 **Planting-date sensitivity check**: production anchors are farmer-declared; re-extract a
    sample with ±15/±30-day shifts and measure decay (ARROZ's early-flooding features are the
    most anchor-sensitive; its 0.76 recall may be partly bad anchors).
4.4 Extraction-parity regression test in CI for every serving deploy (a 5-field subsample of the
    250-KML set) — parity is clean today; this keeps it that way.

## 4. What this does *not* change

- The held-out 0.983 is real — same-year, in-region, complete-season performance is excellent.
  The production test measured a different (harder, more honest) condition: **next year's fields**.
- AVEIA↔TRIGO (Roadmap Obstacle 2) stays behind all of this — the 5-crop model dodges it, and the
  production test surfaced a bigger, now-quantified problem.

---
*Companion docs: `STATUS_AND_ROADMAP.md` (overall roadmap), `best_models/README.md` (model cards),
`best_models/datasets/DATASET_ANALYSIS.md` (training-data composition). Test KMLs:
`best_models/kml_test_sample_250.zip`.*
