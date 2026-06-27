# v7 Dense — 5-Crop Classifier (`20260626_190236_dense_5crop`)

XGBoost trained on the **v6 dense temporal series** (Path B): dekadal (P10D) bins over the
full season + fine (P5D) bins over flowering→maturity, instead of v6's 6 coarse stages.
5 crops — **SOJA, MILHO, TRIGO, AVEIA, FEIJAO**. Produced by
`train_xgboost_v7.py --trials 0 --tag dense_5crop`.

This is the payoff of the AVEIA↔TRIGO investigation (see `../../AVEIA_TRIGO_DISCRIMINATION.md`):
denser temporal resolution captures the oats/wheat senescence-timing signal the stage
aggregates blur away.

---

## Headline — beats the 5-crop benchmark

Held-out test (`features_test_v6`, same fields as the v6 benchmark test set).

| Metric | v6 benchmark (stages) | **v7 dense** | Δ |
|---|---|---|---|
| **Accuracy** | 0.886 | **0.926** | **+4.0 pp** |
| **Macro-F1** | 0.884 | **0.926** | **+4.2 pp** |
| AVEIA F1 | 0.78 | **0.87** | +0.09 |
| TRIGO F1 | 0.76 | **0.87** | +0.11 |
| **AVEIA↔TRIGO swaps** | 90 | **52** | **−42%** |

Per-class held-out (v7):

| Crop | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| FEIJAO | 0.96 | 0.99 | 0.98 | 198 |
| SOJA | 0.95 | 0.97 | 0.96 | 196 |
| MILHO | 0.99 | 0.93 | 0.96 | 184 |
| AVEIA | 0.86 | 0.87 | 0.87 | 192 |
| TRIGO | 0.87 | 0.86 | 0.87 | 198 |

**No regression elsewhere** — FEIJAO/SOJA/MILHO all ≥0.96 F1 (FEIJAO up from 0.95). The
entire +4 pp comes from AVEIA/TRIGO becoming separable.

### Safrinha SOJA fix carried over (even improved)
- Train OOF second-season SOJA recall **0.911** (v6 was 0.895); SOJA→MILHO 4 (all 2nd-season).
- Held-out test second-season SOJA recall 0.867 (13/15) with **0 SOJA→MILHO**.

---

## Why it works (one line)
AVEIA and TRIGO are both C3 cereals with no spectral-level contrast, but they **head and
senesce at different times**. The 6 coarse stages average that timing offset away; the
dense P5D flowering→maturity grid resolves it. The matched pilot quantified the lift at
+0.10 AUC on the pair (`AVEIA_TRIGO_DISCRIMINATION.md §7`).

---

## Config & data
- **Features:** 1,044 raw dense bins (`{idx}_{stat}_d{k}` dekadal, `_f{k}` fine; mean/p10/p90)
  + cyclic planting date + area/lat/lon + null indicators → **1,541 engineered, 924 selected**
  (gain, top 60%). Preset `fix`: class-balance, 6× safrinha-SOJA up-weight, abstain gate.
- **Train:** `src/data/features_v6/` — 5,250 fields (incl. 144 safrinha-SOJA backfilled from
  `aug_ss_soja*/`), matched to `features_v5` field_ids.
- **Test:** `src/data/features_test_v6/` — 1,016 fields, the same held-out split as the v6
  benchmark; 968 scored after `dekads_covered >= 3`.
- Train OOF: acc 0.933, macro-F1 0.934. Abstain t=0.30 → 100% coverage / 0.933 covered acc
  (model is confident enough that the floor threshold already clears target).

## Pipeline efficiency (bonus)
The v6 pipeline that produced this data runs at **~1.3–1.8 s/field vs v5's ~9.6 s/field
(~6–7×)** — single full-season Statistical-API requests with server-side time aggregation
+ field-level concurrency. See `src/pipelines/phenology_feature_pipeline_v6.py`.

---

## Remaining levers (AVEIA/TRIGO still 0.87, 52 swaps)
1. **Engineered dense-timing features** — explicit senescence-onset/curvature/slope on the
   d/f grids (raw bins only, so far). Cheap, no new API.
2. **SAR texture** — SAR was ~90% null in dekadal bins; panicle-vs-spike canopy texture is
   untapped.
3. **Abstain** — route the residual AVEIA/TRIGO overlap to `NAO_CLASSIFICAVEL`.

## Reproduce
```powershell
.venv\Scripts\python.exe src\model\train_xgboost_v7.py `
  --train-db src\data\features_v6\features.db `
  --test-db  src\data\features_test_v6\features.db `
  --tag dense_5crop --trials 0
```
