---
name: arroz-6crop-added
description: ARROZ added to the dense v7 model — 6 crops, held-out 0.941 macro-F1, ARROZ F1 0.99
metadata:
  type: project
---

ARROZ was added to the dense v7 classifier on 2026-06-28 (Phase 1 of the roadmap), giving a
**6-crop** model (SOJA, MILHO, TRIGO, AVEIA, FEIJAO, ARROZ). Held-out test (n=1,149):
**0.941 acc / 0.941 macro-F1** — up from the 5-crop 0.926 and past the old stage-based 7-crop 0.910.
ARROZ landed near-perfect: **F1 0.99 (P 1.00 / R 0.97)**, 0 false positives on test. Run dir:
`src/model/runs_v7/20260628_075718_dense_6crop_arroz/`.

**How it was done:** no pipeline change needed (ARROZ ~145d fits the dense grid). Dense-extracted
the exact v5 ARROZ subset (637 train / 200 test) via `--match-db features_v5`/`features_test_v5`,
appended into the existing `features_v6`/`features_test_v6` DBs (pipeline is append/resume-safe:
skips field_ids already present). Then `train_xgboost_v7.py --tag dense_6crop_arroz` (no
`--exclude-crops` — v6 carries no CAFE). Full reproduce commands are in `STATUS_AND_ROADMAP.md` §4 Phase 1.

**Two pre-flight risks, both non-issues:** the ARROZ↔MILHO/grass confusion I flagged stayed ≤1.8 %
(OOF); and the [[sar-coverage-gap-safrinha]] ~90 %-null SAR worry didn't bite — ARROZ is carried by
optical moisture (NDMI top features) *plus* surviving SAR (`VV_p10_d9` is #5 overall). No regressions:
safrinha SOJA still 0 SOJA→MILHO ([[second-season-soja-fix]]); AVEIA↔TRIGO swaps 52→47
([[aveia-trigo-bottleneck]]).

**Gotcha:** the repo's runnable Python is `.venv/Scripts/python.exe`; the pyenv-shim `python` on PATH
lacks numpy/sentinelhub. Use the venv for all pipeline/training runs.

Next per roadmap: crop-adaptive grid (§5.1) to unblock CAFE → 7 crops.
