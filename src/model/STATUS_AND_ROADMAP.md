# Crop Classifier — Status & Roadmap (7 cultures)

**Last updated:** 2026-06-30
**Current model:** `runs_v7/20260630_021450_dense_7crop_cafe/` — 7 crops, held-out **0.936 acc / 0.937 macro-F1**.

---

## 1. Where we are

### The model (v7 dense, 7 crops)
SOJA, MILHO, TRIGO, AVEIA, FEIJAO, ARROZ, **CAFE**. Held-out test (1,310 fields):

| Culture | Precision | Recall | F1 | note |
|---|---|---|---|---|
| ARROZ | 0.99 | 0.97 | **0.98** | ≈0 |
| CAFE | 0.99 | 0.95 | **0.97** | 🆕 |
| FEIJAO | 0.94 | 1.00 | **0.97** | ≈0 |
| SOJA | 0.95 | 0.96 | **0.96** | ≈0 |
| MILHO | 0.95 | 0.94 | **0.95** | ≈0 |
| TRIGO | 0.87 | 0.86 | **0.87** | ≈0 |
| AVEIA | 0.86 | 0.86 | **0.86** | ≈0 |
| **Overall** | | | **0.936** | **beats old 7-crop 0.910 (+0.026)** |

**CAFE near-solved on arrival** — precision 0.99 (1 false positive on test); recall 0.95, the 9
misses leaking to annual look-alikes (MILHO 5, SOJA 3, ARROZ 1). Once *covered* by the extended
grid it separates cleanly from the six annuals, as expected for a perennial with no annual
green-up/senescence cycle. Overall sits just below the 6-crop 0.941 because the hard AVEIA↔TRIGO
pair (52 swaps) is the remaining error budget — CAFE itself added almost no new confusion.

**Serve-safety verified:** all crops carry real data in the new late dekads d17–d28 (non-null
0.70–0.94); CAFE (0.86) is *not* an outlier, so there is no "late-bins-present ⇒ CAFE" shortcut
— the uniform re-extraction over the CAFE-length grid did its job.

> ⚠️ **Regression — safrinha SOJA.** Second-season SOJA training samples collapsed **158 → 32**
> because matching v6's field_ids found only 633 / 777 SOJA in `dataset_split/train` (the ~144
> missing were disproportionately safrinha — v6 had sourced them elsewhere). Held-out safrinha
> recall fell **0.867 → 0.60** (n=15). The critical **SOJA→MILHO confusion stays at 0**, so the
> economic risk is contained, but the missing safrinha KMLs should be recovered and re-extracted
> over the v7 grid (see §5.4) before the safrinha claim is restored.

### The pipeline (v6 dense)
`phenology_feature_pipeline_v6.py` — one full-season Statistical-API request per source with
server-side time aggregation on a **hybrid grid**: dekadal P10D over the full season + fine
P5D over flowering→maturity. **~1.4 s/field vs v5's ~9.6 (≈7×)**, field-level concurrency +
shared rate limiter + adaptive resolution, resume-safe.

### How we got here (one paragraph)
v5 used 6 coarse phenological stages; AVEIA↔TRIGO (two C3 cereals) was the entire error
budget (F1 0.76–0.78, 90 swaps). Path A — reshaping the stage features — hit a hard ceiling
(~0.857 pair AUC). Path B — a **denser temporal pipeline** — added +0.10 AUC on a matched
pilot and lifted the 5-crop benchmark 0.886 → 0.926. A follow-up (engineered dense-timing
features) tested **negative** — the raw dense bins already carry the signal. Full trail:
[`AVEIA_TRIGO_DISCRIMINATION.md`](AVEIA_TRIGO_DISCRIMINATION.md),
[`V7_NEXT_LEVERS.md`](V7_NEXT_LEVERS.md),
[`SECOND_SEASON_SOJA_RESULTS.md`](SECOND_SEASON_SOJA_RESULTS.md).

### Components
| Area | File |
|---|---|
| Dense pipeline | `src/pipelines/phenology_feature_pipeline_v6.py` (annual grid, +155d) |
| Extended-grid pipeline | `src/pipelines/phenology_feature_pipeline_v7.py` (v6 subclass, `N_DEKADS=29`, +275d for CAFE) |
| v7 trainer | `src/model/train_xgboost_v7.py` (reuses v6 CV/abstain/diagnostic) |
| Probes | `probe_v6_dense.py` (matched dense vs stage), `probe_aveia_trigo.py` |
| Data (git-ignored) | `features_v6/` (annual grid), `features_v6_ext/` (train 6,174, extended grid, 7 crops incl. CAFE 487), `features_test_v6_ext/` (test 1,373 incl. CAFE 200) |
| Guardrail | abstain gate (`abstain_policy.json`) + season-stratified diagnostic |

### Known limitations
- **AVEIA/TRIGO 0.86–0.87** — improved, not solved (~52 residual swaps; ~0.9 pair-AUC ceiling). Now the dominant error budget.
- **SAR ~90 % null** in dekadal bins (S1 revisit too sparse per 10-day bin) → texture untapped.
- ~~**Dense grid too short for CAFE**~~ ✅ **resolved** — v7 pipeline extends the dekadal grid to +275d (`N_DEKADS=29`); all 7 crops re-extracted uniformly over it.
- **Safrinha SOJA regressed** (recall 0.60, n=15) after the v7 re-extraction lost ~144 SOJA KMLs (mostly safrinha) — see §1 warning and §5.4.
- `dekads_covered ≥ 3` gate drops a few heavily-clouded fields at eval.

---

## 2. The 7-crop universe

| Crop | Type | Cycle (d) | In v7? | Discriminating signal / risk |
|---|---|---|---|---|
| SOJA | C3 legume, annual | ~130 | ✅ | safrinha vs MILHO (solved) |
| MILHO | C4 grass, annual | ~140 | ✅ | C4 vs C3 (easy) |
| FEIJAO | C3 legume, annual | ~90 | ✅ | short cycle (easy) |
| AVEIA | C3 cereal, annual | ~130 | ✅ | vs TRIGO — senescence timing |
| TRIGO | C3 cereal, annual | ~135 | ✅ | vs AVEIA — senescence timing |
| ARROZ | C3 grass, flooded | ~145 | ✅ | flooding (high early NDMI/NDWI + low SAR) — added 2026-06-28, F1 **0.99** |
| **CAFE** | perennial, evergreen | ~270 | ✅ | no annual senescence cycle — separates cleanly; added 2026-06-30 over the extended grid, F1 **0.97** |

For reference, the old **stage-based 7-crop** model scored 0.910 acc with AVEIA/TRIGO as the
bottleneck. The dense 7-crop model now scores **0.936** — the new work was *coverage* (extending
the grid for CAFE), and CAFE added almost no new confusion. The remaining bottleneck is the old
AVEIA/TRIGO pair.

---

## 3. The grid-length blocker — ✅ resolved (v7 pipeline)

The v6 dense grid spanned only **−15 → +155 days** (17 dekads), which fit every annual but left
CAFE's grain_fill/maturity (≈150–270 d) uncovered. `phenology_feature_pipeline_v7.py` (a thin
v6 subclass) extends the dekadal grid to **−15 → +275 days** (`N_DEKADS=29`, ~1485 cols, under
SQLite's ceiling); the P5D fine grid is unchanged (per-crop, flowering-anchored). Because the
classifier can't know the crop at inference, **all 7 crops were re-extracted uniformly over this
grid** (`features_v6_ext` / `features_test_v6_ext`) so annuals carry real post-harvest data in
the late dekads — no train/serve skew, no "late-bins ⇒ CAFE" shortcut (verified, §1).

---

## 4. Roadmap

### Phase 1 — 6 crops: add ARROZ  ✅ **DONE 2026-06-28**
Ran exactly as planned, no pipeline change. Reproduce:
```
# train (637 fields, matched to v5) — appends into features_v6
python src/pipelines/phenology_feature_pipeline_v6.py \
  --kml-root src/data/dataset_split/train/arquivos_kml_ARROZ \
  --match-db src/data/features_v5/features.db --output-dir src/data/features_v6
# test (200 fields) — appends into features_test_v6
python src/pipelines/phenology_feature_pipeline_v6.py \
  --kml-root src/data/dataset_split/test/arquivos_kml_ARROZ \
  --match-db src/data/features_test_v5/features.db --output-dir src/data/features_test_v6
# retrain 6 classes (v6 has no CAFE, so no exclude needed)
python src/model/train_xgboost_v7.py --tag dense_6crop_arroz
```
**Result:** held-out **0.941 acc / 0.941 macro-F1** (n=1,149); ARROZ F1 **0.99** (P 1.00 / R 0.97).
The expected ARROZ↔MILHO/grass risk was minimal (≤1.8 % OOF); the SAR-null worry was a
non-issue (NDMI + `VV_p10_d9` both rank top). Skipped the optional probe — ARROZ's signal was
clearly strong enough to go straight to the full train. Run: `runs_v7/20260628_075718_dense_6crop_arroz/`.
*(Used dedicated ARROZ kml-root dirs rather than `--exclude-crops`; trainer needed no `--exclude-crops` since v6 carries no CAFE.)*

### Phase 2 — 7 crops: add CAFE  ✅ **DONE 2026-06-30**
Extended the dekadal grid to +275d via the new `phenology_feature_pipeline_v7.py`, then
re-extracted all 7 crops uniformly. Reproduce:
```
# train: 6 crops matched to v6 (incl. AVEIA) + CAFE matched to v5 (487) — into features_v6_ext
python src/pipelines/phenology_feature_pipeline_v7.py --kml-root src/data/dataset_split/train \
  --match-db src/data/features_v6/features.db --output-dir src/data/features_v6_ext
python src/pipelines/phenology_feature_pipeline_v7.py --kml-root src/data/dataset_split/train/arquivos_kml_CAFE \
  --match-db src/data/features_v5/features.db --output-dir src/data/features_v6_ext
# test: mirror into features_test_v6_ext (match features_test_v6, then features_test_v5 for CAFE)
# train 7 classes on the extended DBs
python src/model/train_xgboost_v7.py --train-db src/data/features_v6_ext/features.db \
  --test-db src/data/features_test_v6_ext/features.db --tag dense_7crop_cafe
```
**Result:** held-out **0.936 acc / 0.937 macro-F1** (n=1,310); CAFE F1 **0.97** (P 0.99 / R 0.95),
beating the old stage-based 7-crop **0.910**. CAFE leaks only 9/200 to annual look-alikes (MILHO
5, SOJA 3, ARROZ 1), 1 false positive. Serve-safety verified. Caveat: safrinha SOJA regressed
(see §1 warning, §5.4). Run: `runs_v7/20260630_021450_dense_7crop_cafe/`.

*Design note (confirmed):* CAFE is a perennial with no annual green-up/senescence cycle, so once
*covered* it separates easily from the six annuals. The risk was coverage, not confusion.

---

## 5. Cross-cutting upgrades (independent of crop count)

### 5.1 Temporal grid — extended ✅, phenology-normalization still open
The CAFE fix shipped via **fixed-max `N_DEKADS=29`** in `phenology_feature_pipeline_v7.py`. Note we
chose **full uniform re-extraction** of all crops over the long grid (annuals carry real late-dekad
data) rather than null-padding shorter crops — null-padding would have created a train/serve skew
since the inference grid is the same length for every crop. Still open as a cleaner long-term design:
- **Phenology-normalized resampling** — resample every field's cycle onto a fixed number of
  **phenological-time** steps (0–100 % of cycle). Unifies FEIJAO (90 d), TRIGO (135 d) and CAFE
  (270 d) into one comparable schema and makes timing features directly cross-crop. More work; the
  principled design.

### 5.2 SAR texture (Option 2 — the remaining AVEIA/TRIGO lever)
Optical-only so far. Oats (open panicle) vs wheat (compact spike) differ in **canopy
structure** → radar backscatter **texture** (GLCM), a signal *orthogonal* to spectral indices
(so not subject to the redundancy that sank engineered optical timing). Needs (1) a coarser
SAR grid (P20–30D) to kill the 90 % null, and (2) a raster/GLCM path. De-risk on the matched
pilot before re-extracting. Details in [`V7_NEXT_LEVERS.md`](V7_NEXT_LEVERS.md).

### 5.3 Abstain calibration
Per-class / per-confusion-pair thresholds rather than one global gate, so the residual
AVEIA↔TRIGO (and any new hard pair) is routed to `NAO_CLASSIFICAVEL` without sacrificing
coverage on easy crops.

### 5.4 Data / evaluation
- **Recover the missing safrinha SOJA** — the v7 re-extraction matched only 633/777 SOJA from
  `dataset_split/train` (second-season samples 158 → 32), regressing safrinha recall to 0.60.
  Find the ~144 missing SOJA KMLs (v6 sourced them outside `dataset_split/train`), extract them
  over the v7 grid into `features_v6_ext`, and retrain to restore the safrinha claim.
- Grow the **safrinha SOJA** test set beyond n≈15 (needs new KMLs) to tighten that recall CI.
- As crops are added, **track new confusion pairs** with the same stratified diagnostic that
  caught AVEIA/TRIGO and safrinha — never trust global macro-F1 alone.

### 5.5 Production
Inference service that applies `selected_features.json` + the model + `abstain_policy.json`;
monitor MILHO precision and the AVEIA/TRIGO / ARROZ confusions in the field.

---

## 6. Anti-levers (don't spend time here)
- **Engineered dense-timing features** — tested, negative (raw bins already capture it).
- **Cross-index red-edge ratios** (~2 % gain), **extra date features** (~1 %).
- **Optuna tuning** — lost to default params on every sibling model.

---

## 7. Recommended sequence
1. ~~**ARROZ → 6 crops**~~ ✅ **DONE** (0.941, ARROZ F1 0.99).
2. ~~**Extended grid (§5.1)**~~ ✅ **DONE** (`N_DEKADS=29`, v7 pipeline).
3. ~~**CAFE → 7 crops**~~ ✅ **DONE** (0.936, CAFE F1 0.97).
4. **Recover safrinha SOJA KMLs (§5.4)** and retrain — restore the regressed second-season recall. ← **next**
5. If AVEIA/TRIGO (the now-dominant error pair) is short of target: **SAR texture (§5.2)** and
   **abstain calibration (§5.3)**.
