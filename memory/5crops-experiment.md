---
name: 5crops-experiment
description: Result of training the v5 classifier on 5 crops (excluding ARROZ + CAFE)
metadata:
  type: project
---

Ran `train_xgboost_v5.py --exclude-crops ARROZ CAFE --tag 5crops` (120 trials) → `runs_v5/20260616_190750_5crops`. Result: Acc 86.16%, macro-F1 0.8665 vs the 7-crop v5 (Acc 87.82%, macro-F1 0.8929).

**Why:** Removing ARROZ (F1 0.95) and CAFE (F1 0.97) only dropped the headline numbers because those were the two easiest classes — a denominator effect, not degradation. The 5 retained crops each moved <0.5 F1 points. AVEIA (~0.80) and TRIGO (~0.75) stayed exactly as bad.

**How to apply:** Pruning easy classes is not a path to better real performance. The bottleneck is intrinsic AVEIA↔TRIGO confusion — see [[aveia-trigo-bottleneck]]. `train_xgboost_v5.py` now supports `--exclude-crops` and `--tag` (run folder = `{timestamp}_{tag}`); `metrics.json` records an `excluded_crops` field.
