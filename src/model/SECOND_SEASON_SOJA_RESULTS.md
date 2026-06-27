# Second-Season (Safrinha) SOJA Fix — Experiment Results

**Status:** ✅ Resolved & validated on held-out data
**Last updated:** 2026-06-22
**Model to ship:** `src/model/runs_v6/20260620_055235_aug_febmar/` (`train_xgboost_v6.py --preset fix --trials 0`)

---

## 1. The problem

Second-season ("safrinha") SOJA — sown Jan–Mar — was systematically misclassified as
**MILHO**, often at high confidence. Main-season (Sep–Nov) SOJA classified fine.

Random k-fold CV **hid** this: global SOJA recall looked like ~0.94 because ~95% of SOJA
is main-season. The failure only showed up under a **season-stratified** diagnostic.

**Root cause: data scarcity, not the date feature.** The original training set had only
~28 second-season SOJA (all January, **zero in Feb–Mar**) against ~563 MILHO in the same
window — so "early-year planting ⇒ MILHO" was a near-perfect shortcut. Critically,
dropping the date features (ablation) did **not** help (recall stayed ~0.18 and everything
else got worse): those few fields look like MILHO *spectrally*, not just by date.

---

## 2. What we did

1. **Collected the missing data.** Extracted 103 Feb–Mar SOJA fields with the v5
   pipeline into `features.db` (`src/data/aug_ss_soja_febmar/`). Second-season SOJA went
   from 28 → 176 (Feb–Mar: 0 → 103). Optical signal on the new fields is healthy
   (1–6% null per stage after interpolation).

2. **Retrained** with `train_xgboost_v6.py --preset fix` =
   date_mode=cyclic + class-balance + **6× up-weight on second-season SOJA** +
   C3-legume-vs-C4-grass red-edge/senescence features (NDRE/CIRE/PSRI/NDMI) + abstain gate.

3. **Stress-tested the result** against two traps (see §4).

4. **Validated on a truly held-out test set** (§5).

---

## 3. Headline result

Held-out test = **1,414 fields** (~200/crop, SOJA 216 incl. all 16 safrinha),
extracted with the v5 pipeline into `src/data/features_test_v5/`.

| Metric | Before fix | After fix (train OOF) | **Held-out test (n=1,414)** |
|---|---|---|---|
| 2nd-season SOJA recall | 0.321 | 0.895 | **0.938** (15/16) |
| SOJA → MILHO confusions | many | 7 | **2** (0 second-season) |
| main-season SOJA recall | 0.972 | 0.975 | 0.985 |
| overall accuracy | 0.883 | 0.882 | **0.910** |
| macro-F1 | 0.898 | 0.895 | **0.910** |

The original failure mode is **gone on unseen data** (zero safrinha SOJA→MILHO), and the
model is **not overfit** — held-out F1 actually exceeds train OOF, with positive per-class
test−train deltas on 6 of 7 crops (the balanced 200/crop test is a cleaner draw than the
imbalanced, safrinha-up-weighted training OOF).

Per-class held-out F1 (test vs train OOF):

| Crop | Test F1 | Test recall | Δ vs train |
|---|---|---|---|
| ARROZ | 0.99 | 0.98 | +0.023 |
| CAFE | 0.98 | 0.97 | +0.014 |
| SOJA | 0.96 | 0.98 | +0.040 |
| FEIJAO | 0.95 | 0.96 | +0.017 |
| MILHO | 0.94 | 0.93 | +0.012 |
| AVEIA | 0.79 | 0.82 | −0.011 |
| TRIGO | 0.76 | 0.72 | +0.003 |

---

## 4. Traps we checked (and ruled out)

### a) SAR-missingness shortcut — ruled out
Sentinel-1 SAR is ~64% missing for the new Feb–Mar SOJA, and "SAR absent" correlates with
SOJA within the Jan–Mar window (P(SOJA | no-SAR, Jan–Mar)=61% vs base 24%). Risk: the model
learns a Sentinel-1 *coverage artifact* instead of crop biology.

**Test:** added `--no-sar-features` (drops VV/VH/CR/RVI + their null-indicators) and retrained.
Second-season recall was **0.901 without SAR vs 0.895 with** — essentially identical. So the
gain is the real augmentation, not the missingness artifact. **Decision: keep SAR** (it adds a
small broad lift elsewhere: ARROZ/AVEIA/TRIGO ~0.3–1pp), but it is *not* load-bearing for the
safrinha split.

### b) Hyperparameter tuning — did not help
`--trials 40` Optuna run (`runs_v6/20260622_071428_aug_febmar_tuned/`) **lost on every metric**:
acc 0.876 vs 0.882, macro-F1 0.890 vs 0.895, 2nd-season recall 0.866 vs 0.895, SOJA→MILHO 16 vs 11.
Reason: the Optuna objective maximizes *global weighted macro-F1*, not second-season recall, so it
optimizes the wrong target — and 40 trials didn't even match the hand-set defaults
(n_est=500, depth=6, lr=0.05). **Decision: ship the `--trials 0` default-params model.**

---

## 5. Held-out validation details

- `evaluate_test.py` was rewired from the obsolete v4 modules to the **v5 pipeline + v6
  functions**, and now reports the season diagnostic and applies the abstain policy to test.
- Test features re-extracted with the v5 pipeline into `src/data/features_test_v5/`
  (1,416 fields: ~200/crop, SOJA 216 incl. all 16 held-out safrinha SOJA). The old
  `features_test/features.db` was v4-era (no red-edge columns) and unusable for the v6 model.
  Initial validation was a 50/crop sample (366 fields, acc 0.876, macro-F1 0.875); the
  ~200/crop run above tightened the per-class CIs and raised the numbers.
- One leaked field (`SOJA_517392809-1`, present in both augmentation and test) was confirmed
  **absent** from the final test set — no train/test contamination.
- **Abstain gate (t=0.80) on test:** 75% coverage at 96% covered accuracy. Of 16 safrinha
  fields, 6 fell below threshold (→ `NAO_CLASSIFICAVEL`) and the 10 it committed to were all correct.

---

## 6. Known limitations

- **Small safrinha test sample (n=16) — cannot be grown from current data.** The test split
  contains only 17 second-season SOJA total (one excluded as a train leak), so even the
  1,416-field extraction added zero new safrinha. 0.938 = 15/16; CI is wide and one field
  flipping moves it ~6pp. Directionally unambiguous, but don't over-read the third decimal.
- **MILHO held up** on the larger set: F1 0.94, recall 0.93, with only 3 MILHO → SOJA in 200.
  The earlier 50/crop dip (F1 0.87) was small-sample noise; the precision/recall trade from the
  6× safrinha up-weight is mild. Still worth monitoring in production.
- **AVEIA ↔ TRIGO** remain the weakest pair (AVEIA F1 0.79, TRIGO 0.76; TRIGO recall 0.72) —
  a separate, pre-existing problem and the next-biggest accuracy lever.

---

## 7. Future tasks (optional, by priority)

1. **Grow the safrinha test set** beyond 16 fields to tighten the recall estimate
   (the test split only contains ~17 second-season SOJA total — would need new KMLs).
2. **If a tuned model is ever wanted:** change the Optuna objective in `tune_xgb` to weight
   second-season SOJA recall (e.g. blend macro-F1 with the season-diagnostic recall), otherwise
   tuning will keep chasing the wrong target. Plain `--trials N` is not worth re-running.
3. **Watch MILHO precision** in production; if it degrades, lower `--soja-ss-weight` from 6×
   and re-check the safrinha recall trade-off.
4. **Backlog (unrelated):** the AVEIA/TRIGO confusion pair is the next-biggest accuracy lever.
5. **Housekeeping:** the experimental runs (`*_nosar`, `*_tuned`, `*_smoke`) can be deleted;
   only `20260620_055235_aug_febmar/` is the shippable artifact.

---

## 8. Reproduce

```powershell
# Train the shippable model (default params, ~4 min)
.venv\Scripts\python.exe src\model\train_xgboost_v6.py --preset fix --trials 0 --tag aug_febmar

# Evaluate on the held-out test set (no API calls; reuses features_test_v5)
.venv\Scripts\python.exe src\model\evaluate_test.py `
  --run-dir src\model\runs_v6\20260620_055235_aug_febmar `
  --skip-extraction --test-features-dir src\data\features_test_v5 `
  --exclude-field-ids SOJA_517392809-1
```
