# Training datasets for the best_models

This documents **exactly which data trained each flagship model**, how variable that data is,
and the biases you should know before trusting a prediction. The byte-exact databases are
snapshotted in [`datasets/`](datasets/) with checksums in [`datasets/MANIFEST.json`](datasets/MANIFEST.json).

> The source DBs (`src/data/features_*/features.db`) are **git-ignored** (`*.db`). The copies under
> `datasets/` are the preserved record of what produced these models. They are *also* covered by the
> `*.db` ignore rule — they will not be committed unless force-added. Treat this folder as the
> archival snapshot regardless of git state.

---

## 1. What a "sample" is

One row = **one field** (a farmer's plot, delineated by a KML polygon) in **one growing season**,
with a known crop label and planting date. Each row carries a full dense satellite time series for
that field, aggregated into time bins by the phenology pipeline. There are no raw pixels here — the
Sentinel-Hub Statistical API returns per-bin statistics server-side; the DB stores those statistics.

**Label provenance:** crop labels come from the curated KML field library
(`src/data/dataset_split/{train,test}/arquivos_kml_<CROP>/`). Train and test are split at the
**field level** — the same field never appears in both. Feature rows are matched to the v5 lineage
by `field_id`, so a field keeps its identity across pipeline versions.

---

## 2. The four databases and which model each one trained

| DB (in `datasets/`) | Role | Rows (raw) | Rows (effective¹) | Cols | Grid | Consumed by |
|---|---|---|---|---|---|---|
| `features_v6.db` | train | 5,887 | 5,586 | 1,053 | 17 dekads (−15…+155 d) | `5_culturas_no_aveia`, `6_culturas_arroz` |
| `features_test_v6.db` | test | 1,216 | 1,149 | 1,053 | 17 dekads | `5_culturas_no_aveia`, `6_culturas_arroz` |
| `features_v6_ext.db` | train | 6,174 | 5,891 | 1,485 | 29 dekads (−15…+275 d) | `7_culturas_cafe` |
| `features_test_v6_ext.db` | test | 1,373 | 1,310 | 1,485 | 29 dekads | `7_culturas_cafe` |

¹ **Effective rows** = what the trainer actually kept after dropping rows with an unparseable
`planting_date` and rows with `dekads_covered < 3` (the `--min-cov 3` cloud gate). These match the
`n_samples` in each model's `metrics.json`.

**The 5-culture model** is trained from `features_v6.db` with `--exclude-crops AVEIA`, so it drops
AVEIA at load time (5,586 → 4,438 effective). That is the *only* difference between the 5- and
6-culture data — same DB, same schema; the 6-culture model keeps AVEIA.

> ⚠️ **v6 and v6_ext are not interchangeable.** They differ in dekadal grid length (17 vs 29 bins →
> 1,053 vs 1,485 columns). Feeding a v6 row to the 7-crop model (or vice-versa) silently
> misclassifies. Always extract inference features with the pipeline named for the model.

---

## 3. Feature schema — what each row measures

Bookkeeping columns (never features): `field_id`, `crop_label`, `planting_date`, `area_hectares`,
`latitude`, `longitude`, `dekads_covered`, `fine_covered`, `interpolated`.

Everything else is a **time-binned satellite statistic**, named `<SIGNAL>_<STAT>_<bin>`:

- **STAT** ∈ {`mean`, `p10`, `p90`} — the within-field distribution per bin (p10/p90 capture
  sub-field heterogeneity, not just the average).
- **bin** — either a dekadal bin `d0…d{N−1}` (P10D, whole season) or a fine bin (P5D, over
  flowering→maturity). The dense grid is the whole point of the v6/v7 pipeline: it keeps the
  senescence-timing signal that the older 6-stage model discarded.

**8 optical indices** (Sentinel-2): `NDVI, NDWI, EVI, NDRE, CIRE, MTCI, PSRI, NDMI` — greenness,
water/moisture, red-edge/chlorophyll, senescence.
**4 SAR channels** (Sentinel-1): `VV, VH, CR, RVI` — structure/backscatter.

Each optical index contributes 141 columns and each SAR channel 87 (SAR spans fewer bins) in the
29-dekad schema. At train time the trainer also appends **null-indicator columns** and two
**cyclic planting-date** features (`planting_doy_sin/cos`, which replace the raw day-of-year so the
model can't learn a calendar shortcut). The *selected* subset each model keeps is in its
`selected_features.json` (931–1,205 features).

> **SAR is ~90 % null** in the dekadal bins — Sentinel-1's revisit is too sparse to fill a 10-day
> bin reliably. SAR texture is real but largely untapped; optical carries the load today.

---

## 4. Class distribution & variability

Raw field counts per crop (the label balance the model saw before class-balancing weights):

| Crop | `features_v6` (train) | `features_v6_ext` (train) | test_v6 | test_v6_ext |
|---|--:|--:|--:|--:|
| AVEIA | 1,200 | 1,200 | 200 | 200 |
| MILHO | 1,200 | 1,200 | 200 | 182 |
| TRIGO | 1,057 | 1,001 | 200 | 200 |
| FEIJAO | 1,016 | 1,016 | 200 | 175 |
| SOJA | 777 | **633** | 216 | 216 |
| ARROZ | 637 | 637 | 200 | 200 |
| CAFE | — | 487 | — | 200 |

Notes on variability & imbalance:

- **Moderate imbalance** (~2.5×, AVEIA/MILHO vs CAFE). Handled at train time by `class_balance=True`
  (inverse-frequency sample weights) — the raw counts above are *not* what the loss sees.
- **AVEIA and MILHO are capped at 1,200**, so those classes are effectively down-sampled relative to
  what's available; other crops use all matched fields.
- **SOJA carries a within-class split that matters economically:** main-season vs *safrinha*
  (second-season, planted Jan–Mar). The trainer up-weights safrinha SOJA **6×** (`soja_ss_weight=6.0`)
  because SOJA→MILHO confusion is the costly error. See §6.
- The `season_diagnostic` in each `metrics.json` tracks SOJA main vs safrinha recall separately —
  never read the global macro-F1 alone for SOJA.

---

## 5. Temporal & geographic variability

**Where (all four DBs):** fields are overwhelmingly in **Southern/South-Central Brazil**. Latitude
clusters tightly around **−25° to −28°** (Paraná / Santa Catarina / Rio Grande do Sul), longitude
around **−51° to −53°**. There is a long thin tail up to **+3.4° lat / −72.7° lon** (a handful of
fields reaching the far north and west), but the bulk is one agro-climatic region.

> **Geographic bias — read before deploying elsewhere.** This is a *regional* model. A field in the
> Cerrado, the Matopiba frontier, or another climate zone is outside the training distribution;
> the abstain gate is the safety net, not a guarantee. Broaden the KML library before claiming
> national coverage.

**Field size:** median **~8 ha**, mean **~18 ha** (right-skewed), spanning **0.02 ha up to 1,826 ha**
— smallholder plots through large commercial fields. Small fields carry more mixed-pixel noise.

**When (planting dates):**

- `features_v6` train: **2024-02 → 2026-02**, but **~99 % is 2025** (81 rows in 2024, 2 in 2026).
- `features_v6_ext` train: effectively **all 2025** (4 rows in 2024).
- `features_test_v6_ext`: mostly 2025 with a thin historical tail back to **2021**.

> **Single-season bias.** Training is essentially **one crop year (2025)**. Year-to-year weather
> variation (drought, planting delays) is *not* represented, so inter-annual robustness is unproven.
> Planting **month** is well spread (peaks in Feb and Jun — the two Brazilian planting windows), so
> intra-year phenological diversity is good; inter-**year** diversity is the gap.

**Coverage/quality:** `dekads_covered` reaches the grid max (17 or 29) for the median field, i.e.
most fields have a complete series. The `--min-cov 3` gate drops the small tail of heavily-clouded
fields (the raw→effective row drop in §2).

---

## 6. Known biases & caveats (carry these into any decision)

1. **Safrinha-SOJA regression in the 7-crop DB.** The v7 re-extraction matched only **633/777** SOJA
   (`features_v6_ext`); the ~144 lost were disproportionately second-season, collapsing safrinha
   training samples 158 → 32 and dropping held-out safrinha recall **0.867 → 0.60** (n=15). The
   critical **SOJA→MILHO confusion is still 0** in all three models, so the economic risk is
   contained. The **5- and 6-crop DBs (`features_v6`) keep the full 777 SOJA and the better safrinha
   recall.** Recovery plan: `../STATUS_AND_ROADMAP.md` §5.4.
2. **Regional & single-year** (see §5) — the two biggest distribution-shift risks.
3. **AVEIA↔TRIGO** — two C3 cereals, the dominant residual error (~0.86 F1 each). The 5-crop model
   looks near-perfect on TRIGO (0.99) *only because AVEIA is excluded from its data.*
4. **SAR ~90 % null** — structural/texture signal is under-exploited.
5. **Label trust** — labels are only as good as the KML library; no field-verified ground truth
   beyond the curated polygons.

---

## 7. Reproducing / regenerating a dataset

The DBs are built by the phenology pipelines (each field → one Statistical-API request, aggregated
server-side). Exact commands are in `../STATUS_AND_ROADMAP.md` §4. Sketch:

```bash
# 5/6-crop grid (17 dekads)
python src/pipelines/phenology_feature_pipeline_v6.py \
  --kml-root src/data/dataset_split/train --output-dir src/data/features_v6
# 7-crop extended grid (29 dekads, covers CAFE's ~270-day cycle)
python src/pipelines/phenology_feature_pipeline_v7.py \
  --kml-root src/data/dataset_split/train --match-db src/data/features_v6/features.db \
  --output-dir src/data/features_v6_ext
```

To verify a snapshot is the exact one that trained a model, compare its md5 against
`datasets/MANIFEST.json`.

---

## 8. Files in `datasets/`

| File | What it is |
|---|---|
| `features_v6.db` | train DB for the 5- & 6-culture models (17 dekads) |
| `features_test_v6.db` | held-out test DB for the 5- & 6-culture models |
| `features_v6_ext.db` | train DB for the 7-culture model (29 dekads, incl. CAFE) |
| `features_test_v6_ext.db` | held-out test DB for the 7-culture model |
| `MANIFEST.json` | md5, size, row/col counts, effective-N, crop/date/geo distributions per DB |
