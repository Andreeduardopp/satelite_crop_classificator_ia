# 6-Crop Classifier (no AVEIA) — `20260625_050029_noaveia`

XGBoost crop classifier trained on **6 cultures** — **ARROZ, CAFE, FEIJAO, MILHO,
SOJA, TRIGO** — produced by `train_xgboost_v6.py --preset fix --trials 0
--exclude-crops AVEIA`. AVEIA (oats) is deliberately excluded because it is
spectrally near-identical to TRIGO (wheat) and was the dominant source of error in
the 7-crop model; removing it isolates the rest of the problem and yields a
near-perfect classifier (see *Why no AVEIA* below).

> ⚠️ This model **cannot identify AVEIA**. It will confidently assign AVEIA fields to
> another class (most likely TRIGO). Only use it where AVEIA is out of scope.

---

## Results

### Held-out test (the number that matters)
Evaluated on `src/data/features_test_v5` — **1,214 fields**, ~200/crop, never seen in
training (AVEIA test rows dropped automatically). Confidence intervals are tight except
for second-season SOJA (see caveats).

| Metric | Value |
|---|---|
| **Accuracy** | **0.972** |
| **Macro-F1** | **0.972** |
| Weighted-F1 | 0.972 |
| Overfit gap (train − test) | **−0.024** (test *beats* train OOF → not overfit) |

| Crop | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ARROZ | 1.00 | 0.98 | 0.99 | 200 |
| CAFE | 0.99 | 0.98 | 0.99 | 198 |
| FEIJAO | 0.95 | 0.96 | 0.95 | 200 |
| MILHO | 0.96 | 0.94 | 0.95 | 200 |
| SOJA | 0.94 | 0.98 | 0.96 | 216 |
| **TRIGO** | **1.00** | **1.00** | **1.00** | 200 |

Second-season ("safrinha") SOJA recall: **0.875 (14/16)**; SOJA→MILHO = 3 (1 safrinha).
The safrinha fix from the 7-crop work carries over (see `../../SECOND_SEASON_SOJA_RESULTS.md`).

### Training (5-fold out-of-fold CV)
5,169 samples (full `features_v5` DB minus AVEIA, `stages_covered >= 3`).

| Metric | Value |
|---|---|
| Accuracy | 0.948 |
| Macro-F1 | 0.950 |
| TRIGO F1 | 0.996 |
| Second-season SOJA recall | 0.901 |

---

## Why no AVEIA (the comparison)

AVEIA↔TRIGO (both C3 winter cereals, near-identical phenology) was essentially the
entire remaining error budget of the 7-crop model. Dropping AVEIA:

| | 7-crop (with AVEIA) | **6-crop (this model)** |
|---|---|---|
| Test accuracy | 0.910 | **0.972** |
| Test macro-F1 | 0.910 | **0.972** |
| TRIGO F1 | 0.76 | **1.00** |

Every other crop also nudged up. This confirms AVEIA/TRIGO is the bottleneck; the
"real" fix for a true 7-crop model is AVEIA-vs-TRIGO discriminator features, not
dropping the class.

---

## Timing

| Step | Wall time | Notes |
|---|---|---|
| Training (`--trials 0`) | **~4 min** | 05:00:30 → 05:04:46, local CPU. Default params, no Optuna tuning. |
| └ feature selection | ~30 s | gain-ranked, kept 484 / 808 |
| └ 5-fold OOF CV + final fit | ~3.5 min | |
| Held-out evaluation | **~2 s** | reuses cached `features_test_v5` (no API calls) |
| (context) test-set feature extraction | ~3h 47m | Sentinel Hub API, ~200/crop; shared across all v6 evals, one-time |

Hyperparameter tuning was **not** used: a separate 40-trial Optuna run on the
sibling 7-crop model lost to default params on every metric (the objective optimizes
global macro-F1, not the targeted recalls), so `--trials 0` defaults are the choice.

---

## Configuration (`--preset fix`)

| Setting | Value |
|---|---|
| date_mode | `cyclic` (sin/cos of planting day-of-year; drops the raw monotonic DOY) |
| class_balance | on (inverse-frequency sample weights) |
| second-season SOJA up-weight | 6× (safrinha SOJA fix) |
| abstain gate | on — `NAO_CLASSIFICAVEL` below max-softmax threshold |
| date features present | `planting_doy_sin`, `planting_doy_cos` |
| features | 808 engineered → 484 selected (gain, top 60%) |

### Abstain policy
Chosen threshold **0.30** (target covered-accuracy 0.93). Because the 6-crop model is
so confident, coverage is ~100% at the lowest threshold while already exceeding the
target. To trade coverage for higher SOJA/MILHO precision, raise the threshold — e.g.
`t=0.80` → 92% coverage / 0.977 covered accuracy / 0.942 SOJA-MILHO accuracy
(full sweep in `metrics.json`).

---

## Artifacts

| File | Description |
|---|---|
| `xgboost_crop_classifier.json` | the trained 6-class XGBoost model |
| `selected_features.json` | ordered list of the 484 features the model expects |
| `best_params.json` | XGBoost hyperparameters used |
| `metrics.json` | full training metrics, season diagnostic, abstain sweep, per-class |
| `abstain_policy.json` | inference gate (emit `NAO_CLASSIFICAVEL` below threshold) |
| `test_eval/test_metrics.json` | held-out test metrics (1,214 fields) |
| `*.png` | confusion matrix / feature importance / abstain curve (git-ignored) |

---

## Reproduce

```powershell
# Train (reads src/data/features_v5/features.db; ~4 min)
.venv\Scripts\python.exe src\model\train_xgboost_v6.py `
  --preset fix --trials 0 --exclude-crops AVEIA --tag noaveia

# Evaluate on the held-out test set (reuses src/data/features_test_v5; ~2 s)
.venv\Scripts\python.exe src\model\evaluate_test.py `
  --run-dir src\model\runs_v6\20260625_050029_noaveia `
  --skip-extraction --test-features-dir src\data\features_test_v5 `
  --exclude-field-ids SOJA_517392809-1
```

## Caveats
- **Safrinha SOJA n=16** in the test set (the split holds only 17 second-season SOJA,
  one excluded as a train leak). 0.875 = 14/16 — wide CI, one field moves it ~6pp.
- **No AVEIA capability** — see the warning at the top.
- Inference must apply `abstain_policy.json` to emit `NAO_CLASSIFICAVEL` on low confidence.
