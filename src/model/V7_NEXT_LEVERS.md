# v7 — Next Levers for AVEIA↔TRIGO

The v7 dense model (`runs_v7/20260626_190236_dense_5crop/`) lifted the 5-crop held-out
benchmark **0.886 → 0.926** and cut AVEIA↔TRIGO swaps **90 → 52**, using **raw dense bins
only**. AVEIA/TRIGO sit at F1 0.87 with **52 residual swaps** — improved, not solved.

Two levers remain to push the pair further. **Option 1 is the recommended next step**
(free, fast, low-risk); Option 2 is a larger, orthogonal bet.

---

## Option 1 — Engineered dense-timing features  *(TESTED → negative, do not ship)*

> **Result (2026-06-27):** built `features_v7_dense.py` (+163 timing features) and tested
> two ways. Matched 800-field probe: raw bins 0.8937 AUC vs raw+timing 0.8903 (**−0.0034**).
> Full 5-class retrain: raw 0.9256 acc / 52 swaps vs +timing 0.9236 acc / 53 swaps. **No
> gain — slightly worse.** The raw dense bins already let the trees extract the
> senescence-timing offset directly; explicit descriptors are redundant and dilute gain
> selection. Code kept (toggle: `train_xgboost_v7.py --dense-timing`) as a documented
> negative result. Lesson: on the dense grid, *raw bins > hand-engineered shape features*.

### Idea (the hypothesis — which did not hold)
v7 feeds XGBoost the **raw** dekadal (`*_d{k}`) and fine (`*_f{k}`) bins. Trees can
threshold individual time points, but they don't get the *shape* of the curve handed to
them explicitly. Oats and wheat differ mainly in **when** they head and senesce — a timing
offset. Explicit descriptors of that timing, computed on the dense grid, give the model the
signal directly instead of making it reconstruct it from ~30 separate bin-features.

### Why it should work now (when Path A didn't)
Path A (`features_v7.py`) tried the *same kind* of features — senescence onset, curvature,
slopes — but on the **6 coarse stages**, and got nothing (+0.0002 AUC): the stages were too
coarse to locate the transition. On the **dense P5D flowering→maturity grid** those same
descriptors finally have the temporal resolution to be meaningful. This is the natural
follow-through of the pilot result.

### What to build
A `features_v7_dense.py` that, per optical index, computes over the `_d` and `_f` grids:
- **senescence-onset dekad** — first bin after peak below (peak − ½·amplitude)
- **green-up half-max dekad** — first bin above (trough + ½·amplitude)
- **peak dekad**, **amplitude**, **AUC** (season-integrated greenness)
- **senescence slope** / **green-up slope** (rate, not just level)
- **late-season curvature** (2nd difference over flowering→maturity — concave vs convex)
- **rise-vs-fall span asymmetry** (heading-timing proxy)
- red-edge consensus across NDRE/CIRE/MTCI

~60–90 features, **added to** the raw bins (not replacing them), then the usual gain
selection. Reuse the logic in `features_v7.py`, retargeted from stage columns to `_d{k}/_f{k}`.

### Cost / risk / validation
- **Cost:** $0, no API. ~1–2 h dev + a 4-min retrain.
- **Risk:** partial redundancy with raw bins → marginal gain. Cheap to find out.
- **Validate:** `probe_v6_dense.py` with the engineered set (matched AUC delta), then retrain
  v7 and compare AVEIA/TRIGO swaps on the held-out test. Ship only if swaps drop.
- **Expected:** a few points of AVEIA/TRIGO F1 and better behaviour on low-coverage
  (cloudy) fields, where explicit shape features degrade more gracefully than sparse bins.

---

## Option 2 — SAR texture (canopy structure)

### Idea
Everything so far is **optical** — it measures greenness/chlorophyll. Oats (open, drooping
**panicle**) and wheat (compact, erect **spike**) differ in **canopy structure**, which
shows up in **radar backscatter texture**, a signal physically *orthogonal* to the optical
indices. That orthogonality is why it could add beyond the 0.87 optical ceiling.

### Two problems with SAR today
1. **~90 % null in dekadal bins.** Sentinel-1 revisits ~6–12 days, so most P10D bins have no
   pass. Fix: aggregate SAR on a **coarser grid (P20–P30D)** so every bin has ≥1 acquisition.
2. **Only basic stats.** The pipeline computes VV/VH/CR/RVI means — no texture. Structure
   lives in **spatial texture** (GLCM contrast / homogeneity / entropy) and polarimetric
   ratios, which need the **raster** (Process API), not the Statistical API.

### What to build
- Coarsen the SAR aggregation grid (cheap config change) to kill the null rate.
- Add a **raster SAR path**: download VV/VH GAMMA0 rasters per coarse window and compute
  **GLCM texture** (e.g. `skimage.feature.graycomatrix`) + within-field backscatter
  heterogeneity (p90−p10 spread, already partly present).
- Optionally dual-pol decomposition (entropy/alpha) if it pays off.

### Cost / risk / validation
- **Cost:** medium — pipeline work (raster path) + re-extraction of the SAR portion. Slower
  than Stats-only because rasters are heavier downloads.
- **Risk:** SAR speckle is noisy at field scale; texture payoff is **uncertain**. Validate on
  the matched AVEIA/TRIGO set *before* a full re-extraction, same as Path B was de-risked.
- **Expected:** potentially a larger, independent gain than Option 1 — but lower confidence.

---

## Recommendation (updated 2026-06-27)

1. ~~Option 1~~ — **tested, negative** (raw bins already capture it). Closed.
2. **Option 2 (SAR texture) is now the primary remaining lever** — an *orthogonal*
   (structural, not spectral) signal, so it's not subject to the "raw bins already have it"
   redundancy that killed Option 1. De-risk on the matched pilot (`features_v6_matched/`)
   before any SAR re-extraction.
3. Keep the **abstain gate** as the honest fallback for the genuinely ambiguous oats/wheat
   overlap (the ~0.9 pair-AUC ceiling suggests a residual that no feature will fully remove).

Anti-levers (don't bother): more cross-index red-edge ratios (2 % gain), date features
(1 %), Optuna tuning (lost on every sibling model).
