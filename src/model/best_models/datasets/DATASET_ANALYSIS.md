# Dataset analysis — `features_v8` / `features_v8_test` / `features_v9`

Analysis of the exact data used to train and test the crop-classifier models in `../`
(5/6/7-culturas). §1–§7 analyze the **v8 snapshot** the currently-promoted models trained on
(generated 2026-07-07 from `features_v8.db` / `features_v8_test.db`; see
[`MANIFEST.json`](MANIFEST.json) for checksums). §0 covers the **v9 expansion + merged pool**
(2026-07-08) feeding the retrain in progress. One row = one **KML-delineated field** in one season.

## 0. Update 2026-07-08 — `features_v9` (extraction COMPLETE)

After the production field test exposed a ~14-point out-of-year accuracy gap (see
`../../FIELD_TEST_5CROP_ANALYSIS_AND_PLAN.md`), a 16,757-field expansion was sampled from the
SICOR registry KMLs (`kml_sicor_2025/` + `kml_sicor_2026/`) by `scripts/sample_kml_v9.py`
(seed 42, manifest at `src/data/kml_train_v9/sampling_manifest.json`) and extracted into
`src/data/features_v9/features.db` on 2026-07-07/08 (~21 h wall clock, interrupted once by a
Sentinel Hub outage; 204 CAFE files dropped at filename-parse). It is **additive to features_v8**
(v8 field_ids excluded from sampling; training merges both), and hard-excludes the held-out test
set and the 250-KML production benchmark (74 benchmark ids were present in the SICOR pool).

**Final state: 16,553 rows, 0 dead rows (`dekads_covered=0` or `fine_covered=0`), 0 rows below
the min-coverage gate** — the 15 rows damaged by transient API failures mid-run were deleted and
re-extracted cleanly.

| Crop | Rows | Median dekads/fine | Planted 2024 | 2025 | 2026 |
|---|--:|---|--:|--:|--:|
| SOJA | 4,333 | 26 / 15 | 23 | 3,999 | 311 |
| MILHO | 3,300 | 25 / 17 | 0 | 2,423 | 877 |
| ARROZ | 2,604 | 28 / 17 | 8 | 2,590 | 6 |
| FEIJAO | 2,597 | 26 / 13 | 4 | 2,131 | 462 |
| CAFE | 1,796 | 28 / 17 | **750** | 1,046 | 0 |
| TRIGO | 1,500 | 29 / 18 | 0 | 1,500 | 0 |
| AVEIA | 423 | 29 / 18 | 0 | 423 | 0 |
| **Total** | **16,553** | | 785 | 14,112 | 1,656 |

*(2026-summer cohorts planted Dec 2025 count under "2025" here; the truncated-tail cohorts total
~2,700 fields across the Dec-2025..Mar-2026 window.)*

Cohort design (why these fields): safrinha SOJA/MILHO 2025 (833 + 1,500 vs the 32-sample safrinha
gap in v8), all available ARROZ/FEIJAO (doubles the scarce classes), the 2025/26 spring main
season, 2026 summer crops (first second-year data; truncated post-harvest tails by design), and
750 2024-planted CAFE (first CAFE year diversity). 2026 winter TRIGO/AVEIA are deferred to a
December 2026 batch (mid-season now).

**Merged training pool = features_v8 (6,174) + features_v9 (16,553) = 22,727 fields** — 3.7× the
v8 training set, materialized at `src/data/features_v8v9_train/features.db` (zero field_id
collisions verified at merge). The v9 sweep models (`runs_v7/*dense_5crop_v9_ssw*`) train on it.

### Merged pool (v8+v9) — per-crop composition

Queried directly from the merged DB, 2026-07-08:

| Crop | n | Area med/mean/max (ha) | Regions | Planting years |
|---|--:|---|---|---|
| SOJA | 4,966 | 8.5 / 24.8 / 2,514 | Sul 89%, SE 5%, CO 4% | 2024: 27 · 2025: 4,628 · 2026: 311 |
| MILHO | 4,500 | 7.5 / 17.0 / 1,089 | Sul 82%, NE 8%, CO 5%, SE 4% | 2025: 3,623 · 2026: 877 |
| FEIJAO | 3,613 | 7.2 / 11.6 / 644 | Sul 98%, SE 2% | 2025: 3,147 · 2026: 462 |
| ARROZ | 3,241 | 10.0 / 25.5 / 1,357 | Sul 97%, NE 2% | 2025: 3,227 · 2026: 6 |
| TRIGO | 2,501 | 8.4 / 13.3 / 333 | Sul 100% | 2025: 2,501 |
| CAFE | 2,283 | 2.2 / 4.6 / 242 | Sudeste 95% | **2024: 750** · 2025: 1,533 |
| AVEIA | 1,623 | 9.0 / 17.5 / 496 | Sul 100% | 2025: 1,623 |
| **Total** | **22,727** | | | 2024: 789 · 2025: 20,282 · 2026: 1,656 |

Key shifts vs the v8-only pool:
- **Safrinha-window SOJA (Jan–Mar planting): 1,176 fields** — up from 32 (37×). The 6× up-weight
  can now be swept down (see the v9 training sweep).
- **Class ranking inverted**: SOJA is now the *largest* class (was 2nd-smallest); AVEIA the
  smallest. Imbalance is ~3.1× (SOJA:AVEIA), still handled by class-balance weights.
- **First multi-year signal**: 2026-planted 1,656 (summer cohorts, truncated tails) + 750
  2024-planted CAFE. Still ~89% single-year (2025) — the 2024 annual-crop year remains the gap.
- **Geography unchanged** (Sul-dominated) — SICOR didn't move it; out-of-region expansion still
  pending from `culturas/`.

## 1. Size at a glance

| | Train (`features_v8.db`) | Test (`features_v8_test.db`) | Total |
|---|--:|--:|--:|
| Fields (KMLs) | 6,174 | 1,373 | **7,547** |
| Columns | 1,485 | 1,485 | — |
| File size | 79,409,152 bytes (~76 MB) | 17,612,800 bytes (~17 MB) | ~93 MB |

Split is done at the **field level** (`src/data/dataset_split/{train,test}/`) — no field ever appears
in both sets.

## 2. Fields per crop (the "number of KMLs")

| Crop | Train | Test | Total | % of dataset |
|---|--:|--:|--:|--:|
| AVEIA | 1,200 | 200 | 1,400 | 18.5% |
| MILHO | 1,200 | 182 | 1,382 | 18.3% |
| TRIGO | 1,001 | 200 | 1,201 | 15.9% |
| FEIJAO | 1,016 | 175 | 1,191 | 15.8% |
| SOJA | 633 | 216 | 849 | 11.2% |
| ARROZ | 637 | 200 | 837 | 11.1% |
| CAFE | 487 | 200 | 687 | 9.1% |
| **Total** | **6,174** | **1,373** | **7,547** | 100% |

AVEIA/MILHO are capped at 1,200 in training (class-balance cap). CAFE is the smallest class overall
(687 fields) — consistent with it having the lowest per-class F1 alongside AVEIA/TRIGO.

## 3. Geographic distribution (region of Brazil, per crop)

Regions below are a **coarse rectangular classification** of each field's `(latitude, longitude)` into
Brazil's five macro-regions — good enough to see where a crop's data lives, not a substitute for real
state-level geocoding. Computed over train+test combined.

| Crop | Sul (PR/SC/RS) | Sudeste (SP/MG/RJ/ES) | Centro-Oeste | Nordeste | Norte | Lat range | Lon range |
|---|--:|--:|--:|--:|--:|---|---|
| AVEIA | 100% | — | — | — | — | −32.4 .. −23.4 | −56.0 .. −48.8 |
| TRIGO | 100% | — | — | — | — | −31.9 .. −23.1 | −55.7 .. −48.0 |
| FEIJAO | 98% | 1–2% | <1% | <1% | <1% | −29.3 .. 0.5 | −54.1 .. −37.9 |
| ARROZ | 97–98% | <1–1% | <1% | <1% | <1% | −33.7 .. 3.2 | −59.9 .. −36.6 |
| MILHO | 76–82% | 3–5% | 3–9% | 3–18% | <1% | −30.9 .. −3.0 | −59.3 .. −36.8 |
| SOJA | 82–83% | 7–11% | 7–8% | <1% | <1% | −32.4 .. 3.3 | −63.1 .. −44.6 |
| CAFE | 1–2% | **90–94%** | <1–1% | 2–3% | <1% | −24.4 .. −3.9 | −72.7 .. −39.0 |

**Reading this:**
- **AVEIA and TRIGO are exclusively South Brazil** (Paraná/Santa Catarina/Rio Grande do Sul) — makes
  sense agronomically (temperate winter cereals), but it also means the AVEIA↔TRIGO confusion pair is
  never tested outside that climate zone.
- **FEIJAO, ARROZ** are also almost entirely South Brazil (~97–98%), with a thin scatter elsewhere.
- **SOJA and MILHO** have the widest spread: still South-Brazil-majority (~76–83%), but with a real
  Center-West/Cerrado and Southeast tail (soy/corn belt), and MILHO reaches into the Nordeste.
- **CAFE is the outlier** — it lives almost entirely in the **Southeast** (São Paulo/Minas Gerais coffee
  belt), the only crop where South Brazil is a minority (~1–2%).
- A few fields land in "Outra/indefinida" (near-coastline or edge-of-bbox points the coarse rectangles
  don't cleanly cover) — negligible (≤3% for any crop, mostly CAFE).

This matches `STATUS_AND_ROADMAP.md`'s framing: the dataset **clusters tightly in South Brazil**
(lat ≈ −25…−28°), with CAFE as the deliberate Southeast exception and thin tails elsewhere. A field
from the Cerrado/Matopiba interior (safrinha SOJA/MILHO heartland) or the North is largely **out of
distribution** — the abstain gate is the safety net for those, not learned generalization.

## 4. Field size (area_hectares)

| Crop | Min (ha) | Median (ha) | Mean (ha) | Max (ha) | Total area (ha, train+test) |
|---|--:|--:|--:|--:|--:|
| CAFE | 0.01 | 2.5 | 4.9 | 97.5 | 3,374 |
| FEIJAO | 0.08 | 7.4 | 10.9 | 140.7 | 12,802 |
| TRIGO | 0.11 | 7.9 | 13.5 | 332.8 | 16,180 |
| MILHO | 0.09 | 8.0 | 17.6 | 923.1 | 23,712 |
| AVEIA | 0.12 | 8.8 | 18.0 | 495.6 | 25,334 |
| ARROZ | 0.09 | 10.0 | 24.0 | 460.4 | 20,090 |
| SOJA | 0.30 | 9.9 | 35.6 | 1,826.4 | 30,304 |

Overall (all crops, all 7,547 fields): min 0.01 ha, median ~7.9 ha, mean ~17.5 ha, max **1,826 ha**
(a single very large SOJA field). CAFE fields are consistently the smallest (median 2.5 ha — small
family coffee plots); SOJA/ARROZ carry the biggest outliers (large commercial grain farms), which is
also why they have the heaviest mean/median gap.

> Field size matters for feature quality: smaller fields carry more mixed-pixel noise, and (until the
> 2026-07-07 resolution fix) very large fields used to fail extraction outright — see
> `../../STATUS_AND_ROADMAP.md` §2.

## 5. Temporal coverage (planting date)

| Crop | Train planting range | Test planting range |
|---|---|---|
| AVEIA | 2025-04-01 .. 2025-06-21 | 2025-04-01 .. 2025-06-01 |
| TRIGO | 2025-04-01 .. 2025-07-26 | 2025-04-01 .. 2025-06-02 |
| FEIJAO | 2025-01-01 .. 2025-12-31 | 2025-01-01 .. 2025-02-20 |
| ARROZ | 2025-01-01 .. 2025-10-10 | 2025-03-22 .. 2025-10-10 |
| MILHO | 2025-01-01 .. 2025-10-15 | 2025-01-10 .. 2025-04-01 |
| SOJA | 2024-02-01 .. 2025-11-01 | 2024-01-05 .. 2025-11-01 |
| CAFE | 2025-01-01 .. 2025-12-10 | **2021-09-14** .. 2025-12-05 |

Training is **overwhelmingly a single crop-year (2025)**; SOJA is the one crop with meaningful 2024
volume (captures safrinha/second-season plantings), and one CAFE test field dates back to 2021 (a
perennial crop, so an old planting date is expected — CAFE doesn't get replanted yearly).
AVEIA/TRIGO's tight 3-month window reflects their real winter-cereal planting season in South Brazil;
FEIJAO/ARROZ/MILHO span nearly the full year (multiple annual planting windows, including safrinha).

## 6. Where this data came from — the unused supply

These 7,547 extracted fields are a small sample of much larger labeled KML libraries:

**`culturas/`** (referenced in `../../STATUS_AND_ROADMAP.md` Part II, Obstacle 1):
- **~416,000 usable KML fields exist on disk**; only **~1.9% have been extracted** into any training set.
- The unused pool skews **2024** (276k) over **2025** (139k) — the crop-year the production field
  test failed on, still largely unextracted.
- Geographically, ~37% of the unused pool is **outside** the South-Brazil cluster this dataset lives in:
  far-South 14%, SP/MS 12%, Cerrado 7%, North 4%.
- Per-crop unused supply is abundant for SOJA (158k), MILHO (125k), TRIGO (66k), CAFE (40k), FEIJAO
  (10k) — but **ARROZ (~5k) and AVEIA (~2.8k) are supply-constrained**, capping how much more
  region/year diversity those two crops can gain without new data collection.

**SICOR registry exports** (added 2026-07-07, `kml_sicor_2025/` 242,280 + `kml_sicor_2026/`
66,296 files — same registry/ID space as `culturas/`, so dedupe by field_id is mandatory):
- safra-2025 is ~97% planting-year 2025 (same year as v8 — volume, not year diversity);
  safra-2026 is a fully disjoint new crop-year (zero overlap with train/test/benchmark).
- Includes 833 available safrinha-window SOJA and 8.6k safrinha MILHO (2025).
- Feeding `features_v9` (§0); full breakdown and usage strategy in `../../SICOR_DATA_PLAN.md`.

## 7. Known biases (carry into any deployment decision)

- **Regional bias** — South Brazil (Paraná/SC/RS) dominates every crop except CAFE; a field from the
  Cerrado/Matopiba interior is largely out-of-distribution for AVEIA/TRIGO/FEIJAO/ARROZ.
  *(features_v9 does not fix this — SICOR is equally Sul-dominated; the fix is culturas/' 
  out-of-region tail.)*
- **Single-year bias** — effectively a 2025 snapshot (SOJA partially excepted); **confirmed as the
  driver of the 84.3% production field-test result on mostly-2024 fields**
  (`../../FIELD_TEST_5CROP_ANALYSIS_AND_PLAN.md`). features_v9 adds 2025/26-season and 2026-summer
  fields; the 2024 crop-year still needs a culturas/ batch.
- **Class imbalance** — ~2.5× between largest (AVEIA/MILHO, 1,200 train) and smallest (CAFE, 487
  train), handled via class-balance weighting at train time.
- **Safrinha (2nd-season) SOJA under-represented** — ~144 mostly-safrinha SOJA fields were never
  matched into this set; held-out safrinha SOJA recall is only ~0.63 (vs ~0.96 overall). SOJA→MILHO
  confusion is 0 regardless, so the economic risk is contained. *(features_v9 targets this with 833
  safrinha-2025 SOJA — pending extraction.)*

## Sources

- `MANIFEST.json` (checksums, aggregate bbox/area stats, crop counts)
- `../README.md` (model summary, per-class recall)
- `../../STATUS_AND_ROADMAP.md` (dataset composition, biases, unused-KML-pool figures)
- Direct query of `features_v8.db` / `features_v8_test.db` (`phenology_features` table) for the
  per-crop area/geo/date breakdowns in §3–§5 above.
