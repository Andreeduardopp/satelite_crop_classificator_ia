# v7 Benchmark — 5-Crop Classifier (`20260626_141339_5culturas_benchmark`)

**Purpose:** honest pre-discriminator **baseline** for the v7 work. 5 cultures —
**SOJA, MILHO, TRIGO, AVEIA, FEIJAO** (ARROZ and CAFE excluded). This deliberately
keeps the hard **AVEIA↔TRIGO** pair *in scope* so we can measure how much the planned
v7 AVEIA-vs-TRIGO discriminator features move it. Nothing here is "the v7 model" yet —
it is the number v7 must beat.

Built with the existing v6 trainer (architecture unchanged), so any gain later is
attributable to the new features, not a model-class change.

```powershell
.venv\Scripts\python.exe src\model\train_xgboost_v6.py `
  --preset fix --trials 0 --exclude-crops ARROZ CAFE `
  --runs-dir src\model\runs_v7 --tag 5culturas_benchmark
```

---

## Held-out test result (the benchmark number)

Evaluated on `src/data/features_test_v5` — **1,016 fields** never seen in training
(ARROZ/CAFE test rows auto-dropped; leaked `SOJA_517392809-1` excluded).

| Metric | Value |
|---|---|
| **Accuracy** | **0.886** |
| **Macro-F1** | **0.884** |
| Weighted-F1 | 0.886 |
| Overfit gap (test − train OOF) | **+0.022** (test beats train → not overfit) |

| Crop | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SOJA | 0.95 | 0.98 | **0.97** | 216 |
| MILHO | 0.98 | 0.94 | **0.96** | 200 |
| FEIJAO | 0.94 | 0.95 | **0.95** | 200 |
| AVEIA | 0.75 | 0.82 | **0.78** | 200 |
| TRIGO | 0.80 | 0.72 | **0.76** | 200 |

Second-season ("safrinha") SOJA recall **0.875 (14/16)**; SOJA→MILHO = 2 (0 safrinha).
The safrinha fix carries over cleanly — SOJA is *not* a problem here.

---

## The entire error budget is AVEIA↔TRIGO

Test confusion matrix (rows = true, cols = predicted):

```
          AVEIA  FEIJAO   MILHO    SOJA   TRIGO
  AVEIA     164       0       0       0      36
 FEIJAO       0     191       2       7       0
  MILHO       0       8     189       3       0
   SOJA       0       3       2     211       0
  TRIGO      54       1       0       0     145
```

- **116 total errors; 90 of them (78%) are AVEIA↔TRIGO** (TRIGO→AVEIA 54, AVEIA→TRIGO 36).
- The other 3 crops are near-clean (SOJA/MILHO/FEIJAO F1 0.95–0.97); the remaining 26
  errors are minor scatter among them.

**Implication for v7:** AVEIA and TRIGO are both C3 winter cereals with near-identical
phenology, so the C3-legume-vs-C4-grass red-edge logic that fixed SOJA↔MILHO does **not**
transfer — there is no C3/C4 contrast to exploit between two cereals. This pair is also
*not* a data-scarcity problem (AVEIA 1200, TRIGO 1057 in training). The lever has to be
features that separate two cereals: phenology *timing* (heading/senescence offset) and
canopy *structure* (SAR texture), not another chlorophyll ratio. Removing AVEIA entirely
took the 6-crop model to 0.972 (see `../../runs_v6/20260625_050029_noaveia/`); the gap
between that and this 0.886 is the AVEIA↔TRIGO problem v7 has to close.

> **Follow-up:** the AVEIA↔TRIGO separability probe and the (negative) Path-A
> feature-engineering result are written up in `../../AVEIA_TRIGO_DISCRIMINATION.md`.
> Short version: these two cereals cap out at ~0.857 AUC on the current features, so the
> next lever is a denser pipeline (Path B), not more engineered features.

---

## Config

Preset `fix`: date_mode=cyclic, class-balance, 6× safrinha-SOJA up-weight, abstain gate.
808 engineered features → 484 selected (gain, top 60%). `--trials 0` (default params;
Optuna tuning lost on the sibling 7-crop model). Abstain @ t=0.80 → coverage 79.5%,
covered acc 0.954.

## Reproduce eval
```powershell
.venv\Scripts\python.exe src\model\evaluate_test.py `
  --run-dir src\model\runs_v7\20260626_141339_5culturas_benchmark `
  --skip-extraction --test-features-dir src\data\features_test_v5 `
  --exclude-field-ids SOJA_517392809-1
```
