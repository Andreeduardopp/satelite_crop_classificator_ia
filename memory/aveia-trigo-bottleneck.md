---
name: aveia-trigo-bottleneck
description: AVEIA<->TRIGO confusion is the dominant error in the crop classifier; explains v4 vs v5 score gaps
metadata:
  type: project
---

The model's headline accuracy is dominated by one confusion pair: **AVEIA (oats) ↔ TRIGO (wheat)**, both C3 winter cereals with near-identical phenology/spectral signatures. In the 5-crop v5 run (`runs_v5/20260616_190750_5crops`), TRIGO loses 28% of its samples to AVEIA and AVEIA loses 17% to TRIGO; every other class (FEIJAO/MILHO/SOJA) is 91–94% correct. This pair accounts for almost all error.

**Why:** AVEIA was added to the dataset *after* the early v4 run `runs_v4/20260612_095013` (which scored 94.6% / macro-F1 0.9458 on 6 classes, balanced 1000/crop, no AVEIA, TRIGO F1=0.999). Once AVEIA was introduced (7-class and 5-class v5 runs ~86–88%), TRIGO collapsed to ~0.75 and AVEIA only reaches ~0.80. So the "better" old result was an easier problem, not a better model.

**How to apply:** Don't treat the old 94.6% as a regression baseline — it's apples-to-oranges. To raise real performance, add features that separate AVEIA from TRIGO specifically (finer winter-cereal drydown/senescence timing), not class-set changes or ensembling (ensemble only added +0.002 over XGB-alone). Removing easy classes (ARROZ/CAFE) does NOT help — see [[5crops-experiment]].

**QUANTIFIED on held-out test (2026-06-25):** trained v6 with `--exclude-crops AVEIA` (`runs_v6/20260625_050029_noaveia`, 6 classes) and evaluated on `features_test_v5` (1,214 fields after dropping AVEIA). Result: **overall test acc 0.910 → 0.972, macro-F1 0.910 → 0.972, and TRIGO F1 0.76 → 1.00** vs the 7-class model. Every other crop also nudged up (ARROZ .99, CAFE .99, FEIJAO .95, MILHO .95, SOJA .96). Confirms AVEIA↔TRIGO is essentially the ENTIRE remaining error budget — without it the classifier is near-perfect. Safrinha SOJA recall held (0.875 = 14/16, within n=16 noise vs 0.938). **Caveat: this is a diagnostic, not a free win — the 6-class model can no longer identify AVEIA at all; only ship it if AVEIA is genuinely out of scope.** The real fix remains AVEIA-vs-TRIGO discriminator features (see [[second-season-soja-fix]] for the red-edge/senescence feature pattern that worked for SOJA/MILHO).

**SUPERSEDED (2026-07-07):** all numbers above are pre-resolution-fix (`features_v5`/`v6`-era). Current `features_v8` figures (see `src/model/STATUS_AND_ROADMAP.md`): 7-crop AVEIA/TRIGO F1 0.90/0.90 (38 swaps, down from 52 pre-fix); dropping AVEIA still gives TRIGO F1 1.00 in the 5-/6-crop models. The core insight (AVEIA↔TRIGO is the whole residual error budget) still holds — only the magnitudes changed.
