"""Engineered dense-timing features for the v6 dense series (v7 Option 1).

The v7 dense model uses RAW bins (`{idx}_mean_d{k}` dekadal, `_f{k}` fine). Trees can
threshold individual bins, but they don't get the *shape/timing* of the phenology curve
handed to them explicitly. Oats vs wheat differ mainly in WHEN they head and senesce —
exactly the descriptors below, now computed at the dense grid's resolution.

Why this can work when Path A (`features_v7.py`) didn't: Path A built the same descriptors
on the 6 coarse stages and got nothing (+0.0002 AUC) — the stages were too coarse to locate
the transition. On the dense P5D flowering->maturity grid they finally have the resolution.

All features are label-agnostic and computed on the `mean` series. The full set is emitted
on both grids; gain selection prunes the redundant ones (e.g. green-up on the fine grid,
which starts near peak).
"""
import warnings
import numpy as np
import pandas as pd

# Keep in sync with phenology_feature_pipeline_v6 (N_DEKADS/N_FINE, indices).
GRIDS = [("d", 29), ("f", 18)]
OPTICAL = ["NDVI", "NDWI", "EVI", "NDRE", "CIRE", "MTCI", "PSRI", "NDMI"]
RED_EDGE = ["NDRE", "CIRE", "MTCI"]


def _matrix(df, idx, suffix, n):
    cols = [f"{idx}_mean_{suffix}{k}" for k in range(n)]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")


def _first_last(valid):
    """First/last True column index per row (0/n-1 fallback handled by caller via any_valid)."""
    n = valid.shape[1]
    first = valid.argmax(axis=1)
    last = n - 1 - valid[:, ::-1].argmax(axis=1)
    return first, last


def add_dense_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    nrows = len(df)
    new: dict[str, np.ndarray] = {}

    for idx in OPTICAL:
        for suffix, n in GRIDS:
            M = _matrix(df, idx, suffix, n)
            if M is None:
                continue
            valid = ~np.isnan(M)
            any_valid = valid.any(axis=1)
            rows = np.arange(nrows)

            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN
                Mhi = np.where(valid, M, -np.inf)
                Mlo = np.where(valid, M, np.inf)
                peak_idx = Mhi.argmax(axis=1)
                trough_idx = Mlo.argmin(axis=1)
                peak = np.where(any_valid, M[rows, peak_idx], np.nan)
                trough = np.where(any_valid, M[rows, trough_idx], np.nan)
                amp = peak - trough
                auc = np.nanmean(np.where(valid, M, np.nan), axis=1)

                first_i, last_i = _first_last(valid)
                first_v = M[rows, first_i]
                last_v = M[rows, last_i]
                gu_span = np.maximum(peak_idx - first_i, 1)
                sen_span = np.maximum(last_i - peak_idx, 1)
                greenup_slope = (peak - first_v) / gu_span
                senescence_slope = (peak - last_v) / sen_span
                maturity_retention = last_v / np.where(peak == 0, np.nan, peak)

                # second-difference curvature over the series (concave vs convex)
                d2 = M[:, :-2] - 2.0 * M[:, 1:-1] + M[:, 2:]
                curvature = np.nanmean(d2, axis=1)

            # transition-timing stages (row loop; cheap at this scale)
            thr_sen = peak - 0.5 * amp
            thr_gu = trough + 0.5 * amp
            sen_onset = np.full(nrows, np.nan)
            gu_half = np.full(nrows, np.nan)
            for r in np.where(any_valid)[0]:
                row = M[r]
                pk = int(peak_idx[r])
                for k in range(pk + 1, n):
                    if not np.isnan(row[k]) and row[k] <= thr_sen[r]:
                        sen_onset[r] = k
                        break
                for k in range(n):
                    if not np.isnan(row[k]) and row[k] >= thr_gu[r]:
                        gu_half[r] = k
                        break
            rise_fall = (peak_idx - gu_half) - (sen_onset - peak_idx)

            p = f"{idx}_{suffix}"
            new[f"{p}_peakpos"] = peak_idx.astype(float)
            new[f"{p}_amp"] = amp
            new[f"{p}_auc"] = auc
            new[f"{p}_guslope"] = greenup_slope
            new[f"{p}_senslope"] = senescence_slope
            new[f"{p}_matret"] = maturity_retention
            new[f"{p}_curv"] = curvature
            new[f"{p}_senonset"] = sen_onset
            new[f"{p}_guhalf"] = gu_half
            new[f"{p}_risefall"] = rise_fall

    # red-edge consensus on the fine grid (the senescence window)
    for feat in ("senonset", "senslope", "curv"):
        keys = [f"{i}_f_{feat}" for i in RED_EDGE if f"{i}_f_{feat}" in new]
        if len(keys) == len(RED_EDGE):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                new[f"REDEDGE_f_{feat}_consensus"] = np.nanmean(
                    np.vstack([new[k] for k in keys]), axis=0)

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index).astype(np.float32)], axis=1)
    return df


if __name__ == "__main__":
    # quick self-check on the matched pilot DB
    import sqlite3
    d = pd.read_sql("SELECT * FROM phenology_features",
                    sqlite3.connect("src/data/features_v6_matched/features.db"))
    before = d.shape[1]
    d2 = add_dense_timing_features(d)
    print(f"added {d2.shape[1] - before} dense-timing features; sample:",
          [c for c in d2.columns if c.endswith("_f_senonset")][:3])
