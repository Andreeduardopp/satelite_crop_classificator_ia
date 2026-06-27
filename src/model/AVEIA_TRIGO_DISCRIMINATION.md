# AVEIA-vs-TRIGO Discrimination — Findings

**Status:** 🟥 Path A exhausted · 🟩 **Path B SHIPPED** — v7 dense model beats benchmark.
**Last updated:** 2026-06-27
**Bottom line:** Reshaping the 6-stage aggregates can't beat ~0.857 AUC (Path A). The
**denser temporal pipeline** (v6: dekadal + P5D fine grid) does: the v7 dense model lifts
the 5-crop held-out benchmark **0.886 → 0.926 acc** and cuts AVEIA↔TRIGO swaps **90 → 52
(−42%)**, AVEIA/TRIGO F1 0.77 → 0.87, with no regression elsewhere. Model:
`runs_v7/20260626_190236_dense_5crop/`. (§7–§8)

---

## 1. The problem

In the v7 5-crop benchmark (`runs_v7/20260626_141339_5culturas_benchmark/`,
held-out acc 0.886 / macro-F1 0.884), **AVEIA and TRIGO are essentially the entire
error budget**:

```
test confusion (rows=true, cols=pred)
          AVEIA  FEIJAO   MILHO    SOJA   TRIGO
  AVEIA     164       0       0       0      36
 FEIJAO       0     191       2       7       0
  MILHO       0       8     189       3       0
   SOJA       0       3       2     211       0
  TRIGO      54       1       0       0     145
```

- **116 total errors → 90 (78%) are AVEIA↔TRIGO** (TRIGO→AVEIA 54, AVEIA→TRIGO 36).
- The other three crops are near-clean (SOJA/MILHO/FEIJAO F1 0.95–0.97).
- Both are C3 winter cereals with near-identical phenology — and it is **not** a
  data-scarcity problem (training: AVEIA 1200, TRIGO 1057).

---

## 2. Separability probe — the ceiling

`probe_aveia_trigo.py` trains a *dedicated* 2-class XGBoost (AVEIA vs TRIGO) on the
full 808-feature v6 set (2,257 train fields; 400 held-out test).

| | CV (5-fold) | Held-out test (n=400) |
|---|---|---|
| Accuracy | 0.772 | 0.763 |
| AUC | **0.857** | **0.857** |

CV and held-out agree to the third decimal → this is a real information ceiling, not
overfitting or a small-sample wobble. A model that does *nothing but* separate these
two still misclassifies ~1 in 4.

---

## 3. Where the (limited) signal is

Gain breakdown of the 2-class model:

| Index family | gain | | Feature type | gain |
|---|---|---|---|---|
| **RED_EDGE** (NDRE/CIRE/MTCI/PSRI/NDMI) | **51%** | | level (per-stage stat) | ~62% |
| BROADBAND (NDVI/NDWI/EVI) | 35% | | timing (rates/deltas/peaks) | ~36% |
| SAR (VV/VH/CR/RVI) | 11% | | **cross-index (over/minus)** | **2%** |
| DATE (planting_doy) | **1%** | | | |

Top features: `EVI green-up delta veg→flowering`, then red-edge **levels at
grain-fill / maturity** (`CIRE_p10_grain_fill`, `MTCI/PSRI/NDRE _p10_maturity`).

**Reading:**
- **Red-edge matters most** (51%) — concentrated at senescence (grain-fill→maturity)
  and the green-up rate. So oats and wheat *do* differ in red-edge chlorophyll dynamics.
- **The C3/C4 cross-index ratios are worthless here (2%)** — the discriminator-C
  construction that fixed SOJA↔MILHO does not transfer: two cereals have no C3/C4 contrast.
- **Date is irrelevant (1%)** — same planting window, so no date shortcut to exploit.

---

## 4. Path A result — engineered timing/shape features add nothing

Hypothesis: the residual signal is the *shape and timing* of the senescence/green-up
curve, which per-stage levels miss. `features_v7.add_v7_timing_features` adds 72 new
label-agnostic curve-geometry descriptors (late-season curvature, drydown acceleration,
grain-fill/maturity retention vs peak, normalized senescence depth, senescence-onset &
green-up half-max **stages**, rise-vs-fall span asymmetry, red-edge consensus).

| Feature set | #feat | CV AUC | TEST AUC |
|---|---|---|---|
| baseline (v6) | 808 | 0.8570 | 0.8570 |
| **+v7 timing** | 880 | 0.8556 | **0.8572** |
| Δ | +72 | −0.0014 | **+0.0002** |

**No effect.** The 58 new features that get used absorb 8.5% of gain, but the best one
ranks only **#23** — they merely re-express information already present in correlated
existing features. **Conclusion: the 6-stage aggregates are information-limited; no
reshaping of them breaks 0.857 AUC.**

---

## 7. Path B pilot — dense temporal pipeline (v6) — POSITIVE

`phenology_feature_pipeline_v6.py` replaces v5's per-stage requests with the Statistical
API's own time aggregation: a **hybrid grid** of dekadal (P10D) bins over the full season
plus **fine P5D bins over flowering→maturity** (where oats/wheat diverge), in ~3 optical +
1 SAR calls/field (was ~12). Columns `{idx}_{stat}_d{k}` (dekadal) and `_f{k}` (fine).

**Efficiency:** **1.4–1.8 s/field vs v5's ~9.6 s/field (~6–7×)** — fewer round-trips +
field-level concurrency behind a shared rate limiter. (PU cost ~flat; the win is wall-clock.)

**Signal (the de-risk):** matched comparison — identical fields (all present in
`features_v5`), identical CV folds, averaged over 5 seeds. Run at two sizes:

| Matched set | v5 stage AUC | v6 dense AUC | Δ AUC | v6 dense acc |
|---|---|---|---|---|
| n=300 (150+150) | 0.786 ± 0.006 | 0.836 ± 0.006 | **+0.049** | 0.749 |
| **n=800 (400+400)** | 0.789 ± 0.004 | **0.894 ± 0.005** | **+0.105** | **0.808** |

Both deltas are >10× the seed std → unambiguous. Critically the **gap widens with data**:
v5 stage is flat (~0.79, truly plateaued), while v6 dense climbs from 0.836→0.894 as n
grows (the 1,044-feature dense representation was mildly data-starved at n=300). At n=800
dense **already exceeds the full-set v5 ceiling of 0.857**, on a third of the available
AVEIA/TRIGO data — so full-scale dense projects to ~0.90–0.92+ AUC. This is **optical-only**
(SAR was 90% null in dekadal bins → SAR texture remains untapped upside).

---

## 8. Path B SHIPPED — v7 dense model (full 5-crop)

Full dense re-extraction (`features_v6/` train 5,250 incl. safrinha-SOJA backfill;
`features_test_v6/` test 1,016) → `train_xgboost_v7.py` (raw dense bins + cyclic date,
preset `fix`). Held-out test:

| Metric | v6 benchmark | **v7 dense** | Δ |
|---|---|---|---|
| Accuracy | 0.886 | **0.926** | +4.0 pp |
| Macro-F1 | 0.884 | **0.926** | +4.2 pp |
| AVEIA F1 | 0.78 | **0.87** | +0.09 |
| TRIGO F1 | 0.76 | **0.87** | +0.11 |
| AVEIA↔TRIGO swaps | 90 | **52** | −42% |

No regression on the other crops (FEIJAO 0.95→0.98, SOJA/MILHO ≥0.96); safrinha SOJA recall
held/improved (0 SOJA→MILHO on test). Model + full writeup:
`runs_v7/20260626_190236_dense_5crop/README.md`.

**Follow-up — engineered dense-timing features (Option 1): tested, negative.** Adding 163
senescence-onset/slope/curvature descriptors on the dense grid gave −0.0034 AUC on the
matched pair and −0.002 acc on the full model — the raw bins already capture the timing.
See `V7_NEXT_LEVERS.md`. **SAR texture (Option 2)** is the remaining (orthogonal) lever.

---

## 5. Recommendation (Path A)

**Stop adding features to the current aggregates.** The cheap path is exhausted. Options,
by cost:

1. **Abstain fallback (cheapest, ship-now).** Route low-confidence AVEIA/TRIGO to
   `NAO_CLASSIFICAVEL` via the existing gate. Honest, and the residual overlap is
   genuinely ambiguous at this resolution.
2. **Path B — denser temporal signal (highest ceiling).** The 6 phenology stages are
   likely too coarse to resolve the heading/senescence *timing offset* between oats and
   wheat. Extend the v5 pipeline to sample finer time steps around flowering→maturity
   (e.g. dekadal composites) instead of one aggregate per stage. This is a pipeline +
   re-extraction job, not a model change.
3. **Path B' — SAR texture.** Current SAR is only basic VV/VH stats (11% gain). Panicle
   (oats) vs spike (wheat) canopy structure may show in SAR texture / GLCM features that
   the current pipeline doesn't compute.

What **not** to do: more cross-index red-edge ratios (2% gain), date features (1%), or
hyperparameter tuning (lost on the sibling models).

---

## 6. Artifacts & reproduce

| File | Role |
|---|---|
| `runs_v7/20260626_141339_5culturas_benchmark/` | 5-crop benchmark (the gap to close) |
| `probe_aveia_trigo.py` | 2-class separability probe + gain breakdown |
| `features_v7.py` | candidate v7 timing/shape features (kept; Path A negative result) |

```powershell
# Reproduce the probe (baseline vs +v7 timing, ~2 min)
.venv\Scripts\python.exe src\model\probe_aveia_trigo.py
```

`features_v7.py` is retained intentionally: it's a clean, reusable home for v7 features,
and the negative result is the evidence that Path B (not more engineered features) is the
next move.
