# Crop Classifier — Status & Roadmap (toward 6 → 7 cultures)

**Last updated:** 2026-06-27
**Current model:** `runs_v7/20260626_190236_dense_5crop/` — 5 crops, held-out **0.926 acc / 0.926 macro-F1**.

---

## 1. Where we are

### The model (v7 dense, 5 crops)
SOJA, MILHO, TRIGO, AVEIA, FEIJAO. Held-out test (968 fields):

| Culture | Precision | Recall | F1 | vs v6 stage |
|---|---|---|---|---|
| FEIJAO | 0.96 | 0.99 | **0.98** | +0.03 |
| MILHO | 0.99 | 0.93 | **0.96** | ≈0 |
| SOJA | 0.95 | 0.97 | **0.96** | −0.01 |
| AVEIA | 0.86 | 0.87 | **0.87** | **+0.09** |
| TRIGO | 0.87 | 0.86 | **0.87** | **+0.11** |
| **Overall** | | | **0.926** | **+0.04** |

Safrinha (second-season) SOJA recall 0.867, **0 SOJA→MILHO**. AVEIA↔TRIGO swaps 90 → 52.

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
| Dense pipeline | `src/pipelines/phenology_feature_pipeline_v6.py` |
| v7 trainer | `src/model/train_xgboost_v7.py` (reuses v6 CV/abstain/diagnostic) |
| Probes | `probe_v6_dense.py` (matched dense vs stage), `probe_aveia_trigo.py` |
| Data (git-ignored) | `features_v6/` (train 5,250), `features_test_v6/` (test 1,016), `features_v6_matched/` (de-risk pilot) |
| Guardrail | abstain gate (`abstain_policy.json`) + season-stratified diagnostic |

### Known limitations
- **AVEIA/TRIGO 0.87** — improved, not solved (~52 residual swaps; ~0.9 pair-AUC ceiling).
- **SAR ~90 % null** in dekadal bins (S1 revisit too sparse per 10-day bin) → texture untapped.
- **Dense grid covers only ~170 days** (planting−15 → +155) — fine for annuals, **too short for CAFE** (see §3).
- Safrinha SOJA test sample is small (n≈16).
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
| **ARROZ** | C3 grass, **flooded** | ~145 | ❌ | flooding (low SAR + high NDWI early) — **distinctive, fits the grid** |
| **CAFE** | **perennial, evergreen** | **~270** | ❌ | no annual senescence cycle — easy to separate, **but exceeds the grid** |

For reference, the old **stage-based 7-crop** model scored 0.910 acc with AVEIA/TRIGO as the
bottleneck. With that pair now fixed by the dense pipeline, a dense 7-crop model should clear
0.91 comfortably — the new work is mostly *coverage* (grid length) and *new confusions*, not
the old AVEIA/TRIGO problem.

---

## 3. The key blocker for scaling: dense-grid length

The dense grid is anchored at planting and spans **−15 → +155 days** (17 dekads) + a P5D
fine window over flowering→maturity. Cross-check against crop cycle length:

- **ARROZ (~145 d)** → fits within +155. ✅ No pipeline change needed.
- **CAFE (~270 d)** → grain_fill/maturity (≈150–270 d) fall **outside** the grid. The whole
  back half of coffee's cycle is uncovered. ❌ The grid must be extended before CAFE can be
  modelled.

This is why the recommended order is **ARROZ first (6 crops, no pipeline change), CAFE second
(7 crops, after a grid upgrade).**

---

## 4. Roadmap

### Phase 1 — 6 crops: add ARROZ  *(quick win, no pipeline change)*
1. Dense-extract ARROZ train + test, matched to `features_v5` / `features_test_v5`:
   `phenology_feature_pipeline_v6.py --kml-root dataset_split/{train,test} --match-db ... --exclude-crops <all but ARROZ>` into the existing `features_v6` / `features_test_v6`.
2. Retrain `train_xgboost_v7.py --exclude-crops CAFE` (6 classes), evaluate.
3. **Expected:** high — rice's flooding signature (low SAR backscatter + high early NDWI/NDMI)
   is unlike any other crop here. Main thing to watch: ARROZ↔MILHO/other grasses, but the
   water signal should keep it clean.
4. De-risk first (optional): a 2-class ARROZ-vs-rest probe on the matched pilot, same pattern
   as `probe_v6_dense.py`.

### Phase 2 — 7 crops: add CAFE  *(needs the grid upgrade in §5.1)*
1. Ship the crop-adaptive grid (§5.1) so the pipeline covers CAFE's full ~270-day cycle.
2. Re-extract CAFE (and ideally all crops, for one uniform schema) → train + test.
3. Retrain 7-class, evaluate; compare to the old stage-based 0.910.
4. **Expected:** CAFE is a perennial with no annual green-up/senescence cycle, so once it's
   *covered* it should separate easily from the six annuals. The risk is coverage, not
   confusion.

---

## 5. Cross-cutting upgrades (independent of crop count)

### 5.1 Crop-adaptive / phenology-normalized temporal grid  *(also the CAFE fix)*
Two options:
- **(a) Crop-adaptive timeRange** — extend the dekadal request to each crop's own maturity
  (CAFE → ~28 dekads, annuals → ~16), fixed max `N_DEKADS`, shorter crops null-padded.
  Simple; schema grows to the CAFE length.
- **(b) Phenology-normalized resampling** *(cleaner, recommended long-term)* — resample every
  field's cycle onto a fixed number of **phenological-time** steps (0–100 % of cycle). Unifies
  FEIJAO (90 d), TRIGO (135 d) and CAFE (270 d) into one comparable schema and makes timing
  features directly cross-crop. More work; the principled design.

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
- Grow the **safrinha SOJA** test set beyond n≈16 (needs new KMLs) to tighten that recall CI.
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
1. **ARROZ → 6 crops** (no pipeline change; fast).
2. **Crop-adaptive grid (§5.1)** — unblocks CAFE and improves everything.
3. **CAFE → 7 crops.**
4. If AVEIA/TRIGO (or a new pair) is still short of target: **SAR texture (§5.2)** and
   **abstain calibration (§5.3)**.
