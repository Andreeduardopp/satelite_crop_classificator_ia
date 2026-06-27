"""v7 phenology-timing / curve-shape features — AVEIA-vs-TRIGO focused.

The AVEIA<->TRIGO separability probe showed the residual signal lives in the *shape
and timing* of the red-edge / EVI phenology curve at green-up and senescence, not in
per-stage levels (already captured) nor in C3/C4 cross-index ratios (worthless here:
two cereals, no C3/C4 contrast). v6's engineer_features already has levels, deltas,
peak stage/value, green-up rate, senescence rate and cumulative — so these add only
genuinely NEW descriptors of curve *geometry*:

  * curvature / 2nd-difference of the late-season limb (concave vs convex senescence)
  * drydown acceleration (is the decline speeding up or slowing down)
  * retention of signal at grain-fill / maturity relative to peak (heading-timing proxy)
  * normalized senescence depth (fraction of green-up lost by maturity)
  * senescence-onset stage and green-up half-max stage (WHEN the transitions happen)
  * rise-vs-fall span asymmetry (is the curve rise-heavy or fall-heavy)

All are label-agnostic (computed identically for every field, no crop_label used) and
guarded by column presence, so they no-op cleanly if an index is absent.

Designed to run AFTER train_xgboost_v6.engineer_features (it reuses no v6 columns; it
recomputes peak from the stage means so the two modules stay decoupled).
"""
import numpy as np
import pandas as pd

STAGES = ["baseline", "emergence", "vegetative", "flowering", "grain_fill", "maturity"]
# Red-edge + the two broadband indices that carried the gain in the probe.
V7_INDICES = ["NDRE", "CIRE", "MTCI", "PSRI", "NDMI", "EVI", "NDVI"]
RED_EDGE = ["NDRE", "CIRE", "MTCI"]


def _stage_matrix(df: pd.DataFrame, idx: str):
    """(n, 6) float matrix of {idx}_mean_{stage}; None if not all stages present."""
    cols = [f"{idx}_mean_{s}" for s in STAGES]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].to_numpy(dtype="float64")


def add_v7_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new: dict[str, np.ndarray] = {}
    n = len(df)

    for idx in V7_INDICES:
        M = _stage_matrix(df, idx)
        if M is None:
            continue
        valid = ~np.all(np.isnan(M), axis=1)
        with np.errstate(all="ignore"):
            pk = np.nanmax(M, axis=1)
            mn = np.nanmin(M, axis=1)
            amp = pk - mn
            flow, gf, mat = M[:, 3], M[:, 4], M[:, 5]

            peak_idx = np.full(n, np.nan)
            if valid.any():
                peak_idx[valid] = np.nanargmax(M[valid], axis=1)

            # --- curve geometry (vectorized) ---
            new[f"{idx}_late_curvature"] = flow - 2.0 * gf + mat
            new[f"{idx}_drydown_accel"] = (gf - mat) - (flow - gf)
            safe_pk = np.where(pk == 0, np.nan, pk)
            safe_amp = np.where(amp == 0, np.nan, amp)
            new[f"{idx}_grainfill_retention"] = gf / safe_pk
            new[f"{idx}_maturity_retention"] = mat / safe_pk
            new[f"{idx}_senescence_depth_norm"] = (pk - mat) / safe_amp

            # --- transition timing (row-wise; cheap at this scale) ---
            thr_sen = pk - 0.5 * amp          # half-down from peak
            thr_gu = mn + 0.5 * amp           # half-up from trough
            sen_onset = np.full(n, np.nan)
            gu_half = np.full(n, np.nan)
            for r in np.where(valid)[0]:
                row = M[r]
                pidx = int(peak_idx[r])
                for j in range(pidx + 1, len(STAGES)):       # first descent below half
                    if not np.isnan(row[j]) and row[j] <= thr_sen[r]:
                        sen_onset[r] = j
                        break
                for j in range(len(STAGES)):                 # first rise above half
                    if not np.isnan(row[j]) and row[j] >= thr_gu[r]:
                        gu_half[r] = j
                        break
            new[f"{idx}_senescence_onset_stage"] = sen_onset
            new[f"{idx}_greenup_halfmax_stage"] = gu_half
            # rise span vs fall span: >0 => rise-heavy (slow green-up), <0 => fall-heavy
            new[f"{idx}_rise_fall_span_diff"] = (peak_idx - gu_half) - (sen_onset - peak_idx)

    # --- red-edge senescence consensus (mean of the three RE depth/curvature signals) ---
    if all(f"{i}_senescence_depth_norm" in new for i in RED_EDGE):
        new["REDEDGE_senescence_depth_consensus"] = np.nanmean(
            np.vstack([new[f"{i}_senescence_depth_norm"] for i in RED_EDGE]), axis=0)
    if all(f"{i}_late_curvature" in new for i in RED_EDGE):
        new["REDEDGE_late_curvature_consensus"] = np.nanmean(
            np.vstack([new[f"{i}_late_curvature"] for i in RED_EDGE]), axis=0)

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index).astype(np.float32)], axis=1)
    return df


V7_FEATURE_COUNT_HINT = len(V7_INDICES) * 9 + 2  # for sanity-checking
