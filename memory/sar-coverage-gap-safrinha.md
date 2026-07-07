---
name: sar-coverage-gap-safrinha
description: Sentinel-1 SAR is mostly missing for second-season SOJA, and its missingness is a spurious season/crop proxy that can fake a recall gain
metadata:
  type: project
---

Sentinel-1 SAR coverage is severely gappy for exactly the failing population in [[second-season-soja-fix]]. Measured on `src/data/features_v5/features.db` (6,374 fields) on 2026-06-20:

- **Feb–Mar SOJA (the new augmentation, n=103): SAR 67–79% fully-null per stage; 64% of fields have NO SAR at all, anywhere.** Optical for the same fields is fine (1–6% null/stage). So the augmentation added optical signal but almost no SAR.
- Contrast: main-season SOJA 17% no-SAR, MILHO 8% no-SAR. SAR availability tracks season/region (S1A-only-era acquisition gaps over safrinha-soy regions), not crop.

**Why:** SAR-missingness is a spurious discriminator. Within the Jan–Mar window (the v6 decision boundary): SOJA 52% no-SAR vs MILHO 10% no-SAR → P(SOJA | no-SAR, Jan–Mar)=61% vs base rate 24%. `add_null_indicators` (≥10% threshold) creates `VV_*_is_null` columns and XGBoost routes raw NaNs natively, so the model gets a ready-made rule "Jan–Mar + SAR absent → SOJA" — amplified by the 6× second-season up-weight. This is the SAME failure class as the original "early-year → MILHO" shortcut v6 killed, just inverted. It will inflate OOF 2nd-season recall and then degrade silently as Sentinel-1C coverage fills in through 2025–26.

**How to apply:** When 2nd-season SOJA recall jumps above 0.321 after retraining on the augmented data, do NOT take it at face value. Train twice and compare: (1) as-is, (2) with SAR features AND their `_is_null` indicators dropped, forcing the split onto the well-populated optical red-edge/PSRI/NDMI discriminators ([[red-edge-indices-added]]). Durable gain = optical-only recovers most of the recall. Added a `--no-sar-features` flag to `train_xgboost_v6.py` (drops VV/VH/CR/RVI cols + their `_is_null` indicators in `load_data`) for exactly this A/B. Do NOT bother with more aggressive `backfill-sar` — 64% have zero passes ever acquired with already-24-day windows; it's a data-availability gap, not a window-tuning problem.

**RESULT (2026-06-20, `--trials 0` fast read): the shortcut is NOT load-bearing — concern retired.** 2nd-season SOJA recall = **0.895 with SAR vs 0.901 without** (identical; differs by 1 field of 172). If the gain were a SAR-missingness artifact, dropping SAR would have collapsed it; it didn't. So the 0.321→~0.90 jump is the AUGMENTATION (172 real 2nd-season examples + optical discriminators), durable. SAR still gives a small broad lift elsewhere (overall acc 0.882 vs 0.874, macro-F1 0.895 vs 0.888; ARROZ/AVEIA/TRIGO all ~0.3–1pp better with SAR) — so **keep SAR**. Still TODO: confirm the 0.90 on the real held-out 2nd-season test batch (random-CV OOF can flatter), and rerun `--trials 120` for the tuned final model.

**Note on terminology (2026-07-07):** the "SAR mostly missing" framing here is about a narrow Feb–Mar SOJA subset (64–79% null), NOT a global figure — don't confuse it with the old "SAR ~90% null" claim that `src/model/STATUS_AND_ROADMAP.md` explicitly retired as stale. Current global SAR nullness (all crops, `features_v8`) is ~16%, concentrated at season edges. This file's subset-specific numbers are still accurate for the Feb–Mar SOJA population; they just don't generalize.
