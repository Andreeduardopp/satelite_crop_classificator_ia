"""XGBoost crop classifier v6.

Built on top of v5, but specifically engineered around the failure mode found in
field testing: **second-season ("safrinha") SOJA sown in Jan-Feb is systematically
misclassified as MILHO, often at high confidence**, while main-season (Nov-Dec)
SOJA classifies fine.

Root cause (confirmed in the training data):
    - In Jan-Mar the training set has ~563 MILHO vs only 28 SOJA, and ZERO SOJA in
      Feb-Mar. So "early-year planting => MILHO" is a near-perfect shortcut, and the
      planting-date features (planting_doy / sin / cos) let the model lean on it.
    - Random k-fold CV is BLIND to this: SOJA recall looks like ~0.94 because 95% of
      SOJA is main-season. The failure only shows up on second-season fields.

This script attacks it on four fronts (the A/B/C/E plan):
    A. Date-feature ablation  -> --date-mode {full,cyclic,none} (diagnostic).
    B. Rebalance/augment       -> class-balanced + heavy up-weight of second-season
       second-season SOJA          SOJA via sample_weight (--soja-ss-weight).
    C. Senescence discriminator-> red-edge / PSRI / NDMI features + explicit
                                   C3-legume vs C4-grass cross-index features.
    E. Operational abstain     -> a max-probability gate derived from OOF
       guardrail                   predictions, saved as abstain_policy.json.

It also reports a **season-stratified diagnostic** (second-season SOJA recall and
SOJA<->MILHO confusion) so we stop being fooled by the global CV number, and uses a
manual weighted CV so sample weights are applied to training folds only.

Presets (override individual flags as needed):
    --preset fix          (default) date-mode=cyclic, class-balance, ss-weight=6, abstain
    --preset ablate-date  date-mode=none  (everything else off) -> the A diagnostic
    --preset baseline     reproduces v5 behaviour (no weighting, full dates)
"""

import os
import sqlite3
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

TEXT_COLS = {"field_id", "crop_label", "planting_date"}
# Bookkeeping / data-quality columns that must never be used as features.
# NOTE: v5 forgot to exclude optical_backfill_done (a constant 1 after backfill) and
# interpolated_stages (a data-collection artefact); v6 excludes both.
NON_FEATURE_COLS = TEXT_COLS | {
    "stages_covered", "sar_backfill_done", "optical_backfill_done", "interpolated_stages",
}

STAGES = ["baseline", "emergence", "vegetative", "flowering", "grain_fill", "maturity"]
# Red-edge (NDRE/CIRE/MTCI), senescence (PSRI) and moisture (NDMI) indices are now
# populated in the DB. Engineered features (deltas, peaks, greenup/senescence rates,
# drydown ratios) are generated for every index here, which is most of discriminator C.
INDICES = ["NDVI", "NDWI", "EVI", "NDRE", "CIRE", "MTCI", "PSRI", "NDMI"]
SAR_INDICES = ["VV", "VH", "CR", "RVI"]
STATS_BASE = ["mean", "median", "std", "p10", "p90"]

DATE_FEATURES = ["planting_doy", "planting_doy_sin", "planting_doy_cos"]
NULL_INDICATOR_THRESHOLD = 0.10

# The label produced by the inference layer when confidence is below the gate.
ABSTAIN_LABEL = "NAO_CLASSIFICAVEL"
# SOJA sown in these months is "second-season" (safrinha), the failing sub-population.
DEFAULT_SECOND_SEASON_MONTHS = [1, 2, 3]

PRESETS = {
    # reproduce v5: full date features, no weighting, no abstain
    "baseline":    dict(date_mode="full",   class_balance=False, soja_ss_weight=1.0, abstain=False),
    # diagnostic A: drop ALL date features, otherwise unweighted
    "ablate-date": dict(date_mode="none",   class_balance=False, soja_ss_weight=1.0, abstain=False),
    # the B+C+E fix
    "fix":         dict(date_mode="cyclic", class_balance=True,  soja_ss_weight=6.0, abstain=True),
}


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, date_mode: str = "full") -> pd.DataFrame:
    """date_mode controls how the planting date enters the model:
        full   -> planting_doy + sin + cos (v5 behaviour)
        cyclic -> only sin + cos (drops the raw monotonic doy that is easiest to
                  turn into a hard "early-year => MILHO" threshold)
        none   -> no date features at all (ablation A)
    """
    df = df.copy()

    # 1. Planting day-of-year (key for winter/summer crop separation, but also the
    #    source of the second-season-SOJA => MILHO shortcut).
    if date_mode != "none" and "planting_date" in df.columns:
        doy = pd.to_datetime(df["planting_date"], errors="coerce").dt.dayofyear
        if date_mode == "full":
            df["planting_doy"] = doy
        df["planting_doy_sin"] = np.sin(2 * np.pi * doy / 365)
        df["planting_doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # 2. Stage-to-stage deltas
    for idx in INDICES:
        for stat in ["mean", "median"]:
            for i in range(len(STAGES) - 1):
                s_cur, s_nxt = STAGES[i], STAGES[i + 1]
                col_cur, col_nxt = f"{idx}_{stat}_{s_cur}", f"{idx}_{stat}_{s_nxt}"
                if col_cur in df.columns and col_nxt in df.columns:
                    df[f"{idx}_{stat}_delta_{s_cur}_to_{s_nxt}"] = df[col_nxt] - df[col_cur]

    # 3. Peak stage indicator
    for idx in INDICES:
        stage_cols = [f"{idx}_mean_{s}" for s in STAGES if f"{idx}_mean_{s}" in df.columns]
        if stage_cols:
            subset = df[stage_cols]
            has_data = subset.notna().any(axis=1)
            stage_map = {f"{idx}_mean_{s}": float(i) for i, s in enumerate(STAGES)}
            df[f"{idx}_peak_stage"] = np.nan
            if has_data.any():
                df.loc[has_data, f"{idx}_peak_stage"] = subset.loc[has_data].idxmax(axis=1).map(stage_map)
            df[f"{idx}_peak_value"] = subset.max(axis=1)
            df[f"{idx}_min_value"] = subset.min(axis=1)
            df[f"{idx}_amplitude"] = df[f"{idx}_peak_value"] - df[f"{idx}_min_value"]

    # 4. Greenup rate
    for idx in INDICES:
        baseline_col = f"{idx}_mean_baseline"
        if baseline_col in df.columns and f"{idx}_peak_value" in df.columns:
            peak_stage = df[f"{idx}_peak_stage"].replace(0, np.nan)
            df[f"{idx}_greenup_rate"] = (df[f"{idx}_peak_value"] - df[baseline_col]) / peak_stage

    # 5. Senescence rate
    for idx in INDICES:
        maturity_col = f"{idx}_mean_maturity"
        if maturity_col in df.columns and f"{idx}_peak_value" in df.columns:
            stages_after = (len(STAGES) - 1 - df[f"{idx}_peak_stage"]).replace(0, np.nan)
            df[f"{idx}_senescence_rate"] = (df[f"{idx}_peak_value"] - df[maturity_col]) / stages_after

    # 6. Cross-index ratios at key stages
    for stage in ["vegetative", "flowering", "grain_fill"]:
        ndvi_col, ndwi_col, evi_col = f"NDVI_mean_{stage}", f"NDWI_mean_{stage}", f"EVI_mean_{stage}"
        if ndvi_col in df.columns and evi_col in df.columns:
            df[f"NDVI_EVI_ratio_{stage}"] = df[ndvi_col] / df[evi_col].replace(0, np.nan)
        if ndvi_col in df.columns and ndwi_col in df.columns:
            df[f"NDVI_NDWI_ratio_{stage}"] = df[ndvi_col] / df[ndwi_col].replace(0, np.nan)

    # 7. Coefficient of variation across stages
    for idx in INDICES:
        mean_cols = [f"{idx}_mean_{s}" for s in STAGES if f"{idx}_mean_{s}" in df.columns]
        if len(mean_cols) >= 3:
            row_mean = df[mean_cols].mean(axis=1)
            row_std = df[mean_cols].std(axis=1)
            df[f"{idx}_temporal_cv"] = row_std / row_mean.replace(0, np.nan)

    # 8. Within-field heterogeneity
    for idx in INDICES:
        spreads = []
        for stage in STAGES:
            p10, p90 = f"{idx}_p10_{stage}", f"{idx}_p90_{stage}"
            if p10 in df.columns and p90 in df.columns:
                spreads.append(df[p90] - df[p10])
        if spreads:
            df[f"{idx}_mean_spread"] = pd.concat(spreads, axis=1).mean(axis=1)

    # 9. Cumulative index
    for idx in INDICES:
        mean_cols = [f"{idx}_mean_{s}" for s in STAGES if f"{idx}_mean_{s}" in df.columns]
        if mean_cols:
            df[f"{idx}_cumulative"] = df[mean_cols].sum(axis=1)

    # 10. Late-stage divergence features (TRIGO/AVEIA and FEIJAO/SOJA separation)
    for stage in ["grain_fill", "maturity"]:
        ndvi_col, evi_col, ndwi_col = f"NDVI_mean_{stage}", f"EVI_mean_{stage}", f"NDWI_mean_{stage}"
        ndvi_std, evi_std = f"NDVI_std_{stage}", f"EVI_std_{stage}"
        if ndvi_col in df.columns and evi_col in df.columns:
            df[f"NDVI_minus_EVI_{stage}"] = df[ndvi_col] - df[evi_col]
        if ndvi_col in df.columns and ndwi_col in df.columns:
            df[f"NDVI_minus_NDWI_{stage}"] = df[ndvi_col] - df[ndwi_col]
        if ndvi_std in df.columns and evi_std in df.columns:
            df[f"std_ratio_NDVI_EVI_{stage}"] = df[ndvi_std] / df[evi_std].replace(0, np.nan)

    # 11. Full-cycle shape: ratio of early vs late means
    for idx in INDICES:
        early_cols = [f"{idx}_mean_{s}" for s in ["baseline", "emergence"] if f"{idx}_mean_{s}" in df.columns]
        late_cols = [f"{idx}_mean_{s}" for s in ["grain_fill", "maturity"] if f"{idx}_mean_{s}" in df.columns]
        if early_cols and late_cols:
            early_mean = df[early_cols].mean(axis=1)
            late_mean = df[late_cols].mean(axis=1)
            df[f"{idx}_early_late_ratio"] = early_mean / late_mean.replace(0, np.nan)

    # 12. C4-milestone features (MILHO is C4; faster canopy closure than C3 cereals)
    stages_ordered = ["emergence", "vegetative", "flowering", "grain_fill", "maturity"]
    idx = "NDVI"
    stage_cols = [f"{idx}_mean_{s}" for s in stages_ordered if f"{idx}_mean_{s}" in df.columns]
    if stage_cols:
        subset = df[stage_cols]
        valid = subset.notna().any(axis=1)
        stage_to_i = {s: i for i, s in enumerate(stages_ordered)}
        df["NDVI_c4_milestone_idx"] = np.nan
        df["NDVI_c4_cumulative_half_stage"] = np.nan
        if valid.any():
            vs = subset[valid]
            peak_labels = vs.idxmax(axis=1)
            df.loc[valid, "NDVI_c4_milestone_idx"] = peak_labels.map(
                lambda x: stage_to_i.get(x.replace(f"{idx}_mean_", ""), np.nan)).values
            cumsum = vs.cumsum(axis=1)
            half_max = cumsum.max(axis=1) / 2.0
            reached = (cumsum >= half_max.values.reshape(-1, 1)).idxmax(axis=1).map(
                lambda x: stage_to_i.get(x.replace(f"{idx}_mean_", ""), np.nan))
            df.loc[valid, "NDVI_c4_cumulative_half_stage"] = reached.values
    for idx in ["NDVI", "EVI", "NDWI"]:
        for stage in ["vegetative", "flowering"]:
            bl, sc = f"{idx}_mean_baseline", f"{idx}_mean_{stage}"
            if bl in df.columns and sc in df.columns:
                denom = df[sc].replace(0, np.nan)
                df[f"{idx}_rise_{stage}_vs_baseline"] = (denom - df[bl]) / denom

    # 13. Late-season drydown ratios
    for idx in INDICES:
        gf_col, mat_col = f"{idx}_mean_grain_fill", f"{idx}_mean_maturity"
        if gf_col in df.columns and mat_col in df.columns:
            df[f"{idx}_maturity_vs_grainfill_ratio"] = df[mat_col] / df[gf_col].replace(0, np.nan)

    # 14. Vegetative-to-maturity total drop
    for idx in INDICES:
        veg_col, mat_col = f"{idx}_mean_vegetative", f"{idx}_mean_maturity"
        if veg_col in df.columns and mat_col in df.columns:
            df[f"{idx}_veg_to_maturity_drop"] = df[veg_col] - df[mat_col]

    # 15. C3-legume (SOJA) vs C4-grass (MILHO) discriminators — DATE-INDEPENDENT.
    #     A N-fixing broadleaf legume and a C4 cereal differ in red-edge chlorophyll
    #     and senescence/moisture dynamics even when sown in the same window. These
    #     give the model a way to tell second-season SOJA from MILHO without the
    #     planting-date shortcut. All label-agnostic (no crop_label used).
    new_cols = {}
    for stage in ["flowering", "grain_fill", "maturity"]:
        ndre, ndvi = f"NDRE_mean_{stage}", f"NDVI_mean_{stage}"
        if ndre in df.columns and ndvi in df.columns:
            new_cols[f"NDRE_over_NDVI_{stage}"] = df[ndre] / df[ndvi].replace(0, np.nan)
        psri, ndmi = f"PSRI_mean_{stage}", f"NDMI_mean_{stage}"
        if psri in df.columns and ndmi in df.columns:
            new_cols[f"PSRI_minus_NDMI_{stage}"] = df[psri] - df[ndmi]
        cire = f"CIRE_mean_{stage}"
        if cire in df.columns and ndvi in df.columns:
            new_cols[f"CIRE_over_NDVI_{stage}"] = df[cire] / df[ndvi].replace(0, np.nan)
    if {"PSRI_mean_maturity", "PSRI_mean_flowering"}.issubset(df.columns):
        new_cols["PSRI_rise_flow_to_mat"] = df["PSRI_mean_maturity"] - df["PSRI_mean_flowering"]
    if {"NDRE_mean_flowering", "NDRE_mean_maturity"}.issubset(df.columns):
        new_cols["NDRE_fall_flow_to_mat"] = df["NDRE_mean_flowering"] - df["NDRE_mean_maturity"]
    if {"NDMI_mean_grain_fill", "NDMI_mean_maturity"}.issubset(df.columns):
        new_cols["NDMI_drydown_gf_to_mat"] = df["NDMI_mean_grain_fill"] - df["NDMI_mean_maturity"]
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    _assert_no_label_leakage(df)
    return df


def _assert_no_label_leakage(df: pd.DataFrame) -> None:
    leaked = [c for c in ("is_c4", "is_c4_x_NDVI_peak_stage", "is_c4_x_EVI_senescence_rate",
                          "is_soja_second_season", "planting_month") if c in df.columns]
    if leaked:
        raise ValueError(f"Label/target-derived features present (leakage): {leaked}")


def add_null_indicators(X: pd.DataFrame) -> pd.DataFrame:
    null_rates = X.isnull().mean()
    high_null_cols = null_rates[null_rates >= NULL_INDICATOR_THRESHOLD].index.tolist()
    if not high_null_cols:
        return X
    indicators = {f"{col}_is_null": X[col].isnull().astype(np.float32) for col in high_null_cols}
    X = pd.concat([X, pd.DataFrame(indicators, index=X.index)], axis=1)
    logger.info("Added %d null-indicator columns (threshold >= %.0f%%)",
                len(high_null_cols), NULL_INDICATOR_THRESHOLD * 100)
    return X


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data(
    db_path: str,
    min_stages: int = 0,
    planting_year: int | None = None,
    n_per_crop: int | None = None,
    fallback_year: int | None = None,
    exclude_crops: list[str] | None = None,
    date_mode: str = "full",
    second_season_months: list[int] | None = None,
    drop_sar: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, list[str], LabelEncoder, pd.DataFrame]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM phenology_features", conn)
    if df.empty:
        raise ValueError("phenology_features table is empty")

    if exclude_crops:
        exclude = {c.strip().upper() for c in exclude_crops}
        before = len(df)
        df = df[~df["crop_label"].str.upper().isin(exclude)].reset_index(drop=True)
        logger.info("Excluded crops %s: %d -> %d samples", sorted(exclude), before, len(df))
        if df.empty:
            raise ValueError("No rows left after excluding crops")

    for col in df.columns:
        if col not in TEXT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    valid_planting = pd.to_datetime(df["planting_date"], errors="coerce").notna()
    df = df[valid_planting].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d/%d rows without a valid planting_date", dropped, before)
    if df.empty:
        raise ValueError("No rows left after filtering for valid planting_date")

    if min_stages > 0:
        before = len(df)
        df = df[df["stages_covered"] >= min_stages].reset_index(drop=True)
        logger.info("Filtered stages_covered >= %d: %d -> %d samples", min_stages, before, len(df))

    years_col = pd.to_datetime(df["planting_date"], errors="coerce").dt.year

    if planting_year is not None and n_per_crop is not None:
        primary = df[years_col == planting_year]
        parts = []
        for crop, g in primary.groupby("crop_label"):
            taken = g.sample(min(len(g), n_per_crop), random_state=42)
            parts.append(taken)
            gap = n_per_crop - len(taken)
            if gap > 0 and fallback_year is not None:
                fb = df[(years_col == fallback_year) & (df["crop_label"] == crop)]
                if len(fb) > 0:
                    parts.append(fb.sample(min(len(fb), gap), random_state=42))
        df = pd.concat(parts, ignore_index=True)
        logger.info("Sampled up to %d per crop -> %d samples total", n_per_crop, len(df))
    elif planting_year is not None:
        before = len(df)
        df = df[years_col == planting_year].reset_index(drop=True)
        logger.info("Filtered planting_year == %d: %d -> %d samples", planting_year, before, len(df))
    elif n_per_crop is not None:
        parts = [g.sample(min(len(g), n_per_crop), random_state=42) for _, g in df.groupby("crop_label")]
        df = pd.concat(parts, ignore_index=True)
        logger.info("Sampled up to %d per crop -> %d samples total", n_per_crop, len(df))

    # -- Season metadata (for weighting + diagnostics, NEVER used as a feature) ----
    ss_months = second_season_months or DEFAULT_SECOND_SEASON_MONTHS
    pmonth = pd.to_datetime(df["planting_date"], errors="coerce").dt.month
    crop_upper = df["crop_label"].str.upper()
    meta = pd.DataFrame({
        "crop_label": df["crop_label"].values,
        "planting_month": pmonth.values,
        "is_soja_second_season": ((crop_upper == "SOJA") & pmonth.isin(ss_months)).values,
    })

    df = engineer_features(df, date_mode=date_mode)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].astype(np.float32)
    X = add_null_indicators(X)
    feature_cols = list(X.columns)

    # A/B diagnostic: drop all SAR features AND their null-indicators. SAR is
    # ~64% missing for second-season SOJA and "SAR absent" is a spurious season
    # proxy (P(SOJA|no-SAR, Jan-Mar)=61% vs base 24%); dropping it forces the
    # SOJA/MILHO split onto the well-populated optical red-edge/PSRI/NDMI signal.
    if drop_sar:
        sar_prefixes = tuple(f"{idx}_" for idx in SAR_INDICES)
        sar_cols = [c for c in X.columns if c.startswith(sar_prefixes)]
        X = X.drop(columns=sar_cols)
        feature_cols = list(X.columns)
        logger.info("Dropped %d SAR feature/indicator columns (--no-sar-features)", len(sar_cols))

    le = LabelEncoder()
    y = le.fit_transform(df["crop_label"])

    logger.info("Loaded %d samples, %d features, %d classes: %s",
                len(df), len(feature_cols), len(le.classes_), list(le.classes_))
    logger.info("date_mode=%s | date features present: %s",
                date_mode, [c for c in DATE_FEATURES if c in feature_cols] or "none")
    logger.info("Second-season SOJA in this set: %d / %d SOJA",
                int(meta["is_soja_second_season"].sum()), int((crop_upper == "SOJA").sum()))
    logger.info("Null rate: %.1f%%", X.isna().mean().mean() * 100)
    return X, y, feature_cols, le, meta


# ── Sample Weighting (B) ─────────────────────────────────────────────────────

def compute_sample_weights(
    y: np.ndarray,
    le: LabelEncoder,
    meta: pd.DataFrame,
    class_balance: bool = True,
    soja_ss_weight: float = 6.0,
) -> np.ndarray:
    """Per-sample weights: optional inverse-frequency class balancing, then a heavy
    multiplier on second-season SOJA (the failing sub-population). Mean is normalised
    to 1 to keep the effective learning-rate scale stable."""
    n = len(y)
    w = np.ones(n, dtype=np.float64)

    if class_balance:
        K = len(le.classes_)
        counts = np.bincount(y, minlength=K)
        for c in range(K):
            if counts[c] > 0:
                w[y == c] = n / (K * counts[c])

    ss_mask = meta["is_soja_second_season"].to_numpy()
    n_ss = int(ss_mask.sum())
    if soja_ss_weight != 1.0 and n_ss > 0:
        w[ss_mask] *= soja_ss_weight
        logger.info("Up-weighted %d second-season SOJA samples by %.1fx", n_ss, soja_ss_weight)
    elif soja_ss_weight != 1.0:
        logger.warning("soja_ss_weight=%.1f requested but 0 second-season SOJA present", soja_ss_weight)

    w *= n / w.sum()  # normalise mean weight to 1
    return w.astype(np.float32)


# ── Weighted out-of-fold predictions (manual CV) ─────────────────────────────

def weighted_oof(
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray | None,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Manual StratifiedKFold OOF. Sample weights are applied to TRAIN folds only;
    scoring is unweighted (we want macro-F1 to reward the rare classes). Returns
    (oof_pred, oof_proba)."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    n_classes = len(np.unique(y))
    oof_pred = np.empty(len(y), dtype=int)
    oof_proba = np.zeros((len(y), n_classes), dtype=np.float64)
    for tr, va in cv.split(X, y):
        model = xgb.XGBClassifier(**params)
        fit_kw = {}
        if w is not None:
            fit_kw["sample_weight"] = w[tr]
        model.fit(X.iloc[tr], y[tr], **fit_kw)
        proba = model.predict_proba(X.iloc[va])
        oof_proba[va] = proba
        oof_pred[va] = proba.argmax(axis=1)
    return oof_pred, oof_proba


# ── Feature Selection ────────────────────────────────────────────────────────

def select_features(X, y, feature_names, w=None, keep_ratio=0.60):
    quick_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        tree_method="hist", random_state=42, n_jobs=-1,
        objective="multi:softprob", eval_metric="mlogloss",
    )
    quick_model.fit(X, y, sample_weight=w)

    importance = quick_model.get_booster().get_score(importance_type="gain")
    feat_scores = {}
    for key, val in importance.items():
        if key.startswith("f") and key[1:].isdigit():
            i = int(key[1:])
            name = feature_names[i] if i < len(feature_names) else key
        else:
            name = key
        feat_scores[name] = val
    for f in feature_names:
        feat_scores.setdefault(f, 0.0)

    sorted_feats = sorted(feat_scores.items(), key=lambda x: x[1], reverse=True)
    n_keep = max(10, int(len(sorted_feats) * keep_ratio))
    selected = [f for f, _ in sorted_feats[:n_keep]]
    logger.info("Feature selection: kept %d / %d features", len(selected), len(feature_names))
    return X[selected], selected


# ── Optuna Hyperparameter Tuning (weighted CV) ───────────────────────────────

def tune_xgb(X, y, w, n_trials=120, n_folds=5):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
            "objective": "multi:softprob", "eval_metric": "mlogloss",
            "tree_method": "hist", "random_state": 42, "n_jobs": -1,
        }
        oof_pred, _ = weighted_oof(params, X, y, w, n_folds)
        return f1_score(y, oof_pred, average="macro")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="xgb_v6",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=15))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("Best XGB F1 macro (weighted-train CV): %.4f", study.best_value)
    logger.info("Best XGB params: %s", study.best_params)
    return study.best_params


# ── Season-stratified diagnostic ─────────────────────────────────────────────

def season_diagnostic(y_true, y_pred, le, meta) -> dict:
    """The metric that actually matters: how second-season SOJA fares, and the
    SOJA<->MILHO confusion that random CV hides."""
    classes = list(le.classes_)
    name_pred = np.array(classes)[y_pred]
    name_true = np.array(classes)[y_true]
    ss = meta["is_soja_second_season"].to_numpy()
    is_soja = name_true == "SOJA"
    main = is_soja & ~ss

    def _recall(mask):
        if mask.sum() == 0:
            return None
        return float((name_pred[mask] == name_true[mask]).mean())

    diag = {
        "soja_total": int(is_soja.sum()),
        "soja_main_season": int(main.sum()),
        "soja_second_season": int((is_soja & ss).sum()),
        "soja_main_recall": _recall(main),
        "soja_second_season_recall": _recall(is_soja & ss),
        "soja_to_milho": int(((name_true == "SOJA") & (name_pred == "MILHO")).sum()),
        "milho_to_soja": int(((name_true == "MILHO") & (name_pred == "SOJA")).sum()),
        "second_season_soja_to_milho": int(((is_soja & ss) & (name_pred == "MILHO")).sum()),
    }
    logger.info("── Season diagnostic (out-of-fold) ──")
    logger.info("  SOJA recall  main-season: %s  |  second-season: %s  (n=%d)",
                _fmt(diag["soja_main_recall"]), _fmt(diag["soja_second_season_recall"]),
                diag["soja_second_season"])
    logger.info("  SOJA->MILHO: %d (of which second-season: %d)  |  MILHO->SOJA: %d",
                diag["soja_to_milho"], diag["second_season_soja_to_milho"], diag["milho_to_soja"])
    return diag


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


# ── Abstain policy (E) ───────────────────────────────────────────────────────

def build_abstain_policy(y_true, oof_proba, le, target_acc=0.93) -> dict:
    """Max-probability gate. Sweeps thresholds and picks the one that maximises
    coverage while keeping accuracy on the covered set >= target_acc. The inference
    layer should emit NAO_CLASSIFICAVEL when max prob < threshold."""
    conf = oof_proba.max(axis=1)
    pred = oof_proba.argmax(axis=1)
    correct = (pred == y_true)
    name_true = np.array(le.classes_)[y_true]
    name_pred = np.array(le.classes_)[pred]
    soja_milho = np.isin(name_true, ["SOJA", "MILHO"]) | np.isin(name_pred, ["SOJA", "MILHO"])

    table = []
    for t in np.round(np.arange(0.30, 0.96, 0.05), 2):
        cov_mask = conf >= t
        coverage = float(cov_mask.mean())
        cov_acc = float(correct[cov_mask].mean()) if cov_mask.any() else None
        sm = cov_mask & soja_milho
        sm_acc = float(correct[sm].mean()) if sm.any() else None
        table.append({"threshold": float(t), "coverage": round(coverage, 4),
                      "covered_accuracy": None if cov_acc is None else round(cov_acc, 4),
                      "soja_milho_accuracy": None if sm_acc is None else round(sm_acc, 4)})

    chosen = None
    for row in table:  # smallest threshold (=> max coverage) meeting target
        if row["covered_accuracy"] is not None and row["covered_accuracy"] >= target_acc:
            chosen = row
            break
    if chosen is None:
        chosen = max(table, key=lambda r: (r["covered_accuracy"] or 0))

    policy = {
        "metric": "max_softmax_probability",
        "abstain_label": ABSTAIN_LABEL,
        "rule": "predict crop only if max class probability >= threshold, else abstain",
        "target_covered_accuracy": target_acc,
        "threshold": chosen["threshold"],
        "coverage_at_threshold": chosen["coverage"],
        "covered_accuracy_at_threshold": chosen["covered_accuracy"],
        "soja_milho_accuracy_at_threshold": chosen["soja_milho_accuracy"],
        "sweep": table,
    }
    logger.info("── Abstain policy (E) ──")
    logger.info("  threshold=%.2f -> coverage=%.1f%%, covered acc=%s, SOJA/MILHO acc=%s",
                chosen["threshold"], chosen["coverage"] * 100,
                _fmt(chosen["covered_accuracy"]), _fmt(chosen["soja_milho_accuracy"]))
    return policy


# ── Training & Evaluation ────────────────────────────────────────────────────

def train_and_evaluate(
    X, y, feature_names, le, meta,
    n_folds=5, optuna_trials=120, output_dir="src/model/output_v6",
    excluded_crops=None, date_mode="full", class_balance=True,
    soja_ss_weight=6.0, abstain=True, abstain_target_acc=0.93,
    second_season_months=None,
):
    os.makedirs(output_dir, exist_ok=True)

    w = compute_sample_weights(y, le, meta, class_balance=class_balance, soja_ss_weight=soja_ss_weight)

    X_sel, sel_features = select_features(X, y, feature_names, w=w, keep_ratio=0.60)
    with open(os.path.join(output_dir, "selected_features.json"), "w") as f:
        json.dump(sel_features, f, indent=2)

    if optuna_trials > 0:
        logger.info("Tuning XGBoost with %d trials (weighted CV)...", optuna_trials)
        best_params = tune_xgb(X_sel, y, w, n_trials=optuna_trials, n_folds=n_folds)
        best_params.update({"objective": "multi:softprob", "eval_metric": "mlogloss",
                            "tree_method": "hist", "random_state": 42, "n_jobs": -1})
    else:
        best_params = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
                       "objective": "multi:softprob", "eval_metric": "mlogloss",
                       "tree_method": "hist", "random_state": 42, "n_jobs": -1}

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump({"xgboost": best_params}, f, indent=2, default=str)

    logger.info("Running %d-fold weighted OOF CV...", n_folds)
    y_pred, oof_proba = weighted_oof(best_params, X_sel, y, w, n_folds)

    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")
    f1w = f1_score(y, y_pred, average="weighted")
    logger.info("XGBoost — Acc: %.2f%% | F1 macro: %.4f | F1 weighted: %.4f", acc * 100, f1m, f1w)

    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
    print("\nXGBoost Classification Report:\n" + classification_report(y, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y, y_pred)
    _plot_confusion_matrix(cm, le.classes_, output_dir)

    diag = season_diagnostic(y, y_pred, le, meta)
    policy = build_abstain_policy(y, oof_proba, le, target_acc=abstain_target_acc) if abstain else None
    if policy:
        with open(os.path.join(output_dir, "abstain_policy.json"), "w") as f:
            json.dump(policy, f, indent=2)
        _plot_abstain_curve(policy, output_dir)

    logger.info("Training final model on all data (weighted)...")
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_sel, y, sample_weight=w)
    _plot_feature_importance(model, sel_features, output_dir)

    model_path = os.path.join(output_dir, "xgboost_crop_classifier.json")
    model.save_model(model_path)
    logger.info("XGB model saved: %s", model_path)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "version": "v6",
        "config": {
            "date_mode": date_mode, "class_balance": class_balance,
            "soja_second_season_weight": soja_ss_weight,
            "second_season_months": second_season_months or DEFAULT_SECOND_SEASON_MONTHS,
            "abstain": abstain, "abstain_target_acc": abstain_target_acc,
        },
        "excluded_crops": sorted({c.strip().upper() for c in excluded_crops}) if excluded_crops else [],
        "n_samples": int(len(y)),
        "n_features_original": int(X.shape[1]),
        "n_features_selected": int(X_sel.shape[1]),
        "n_classes": int(len(le.classes_)),
        "classes": list(le.classes_),
        "cv_folds": n_folds,
        "optuna_trials": optuna_trials,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1m, 4),
        "f1_weighted": round(f1w, 4),
        "null_rate": round(float(X.isna().mean().mean()), 4),
        "season_diagnostic": diag,
        "abstain_policy": policy,
        "per_class": {
            cls: {"precision": round(report[cls]["precision"], 4),
                  "recall": round(report[cls]["recall"], 4),
                  "f1": round(report[cls]["f1-score"], 4),
                  "support": int(report[cls]["support"])}
            for cls in le.classes_
        },
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", os.path.join(output_dir, "metrics.json"))
    return model


# ── Plots ────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm, class_names, output_dir):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (% per class)", fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _plot_abstain_curve(policy, output_dir):
    sweep = policy["sweep"]
    t = [r["threshold"] for r in sweep]
    cov = [r["coverage"] for r in sweep]
    acc = [r["covered_accuracy"] or np.nan for r in sweep]
    sm = [r["soja_milho_accuracy"] or np.nan for r in sweep]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, cov, "o-", label="coverage")
    ax.plot(t, acc, "s-", label="covered accuracy")
    ax.plot(t, sm, "^-", label="SOJA/MILHO accuracy")
    ax.axvline(policy["threshold"], color="red", ls="--", label=f"chosen t={policy['threshold']}")
    ax.set_xlabel("confidence threshold (max prob)"); ax.set_ylabel("fraction")
    ax.set_title("Abstain policy: coverage vs accuracy", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "abstain_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _resolve_feature_name(key, feature_names):
    if key.startswith("f") and key[1:].isdigit():
        i = int(key[1:])
        return feature_names[i] if i < len(feature_names) else key
    return key


def _plot_feature_importance(model, feature_names, output_dir, top_n=30):
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        return
    mapped = {_resolve_feature_name(k, feature_names): v for k, v in importance.items()}
    sorted_imp = sorted(mapped.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_imp][::-1]
    values = [x[1] for x in sorted_imp][::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(names, values, color=sns.color_palette("viridis", len(names)))
    ax.set_title(f"Top {top_n} Features by Gain", fontweight="bold")
    ax.set_xlabel("Gain")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost crop classifier v6 (second-season SOJA fix)")
    parser.add_argument("--db", default=os.path.join("src", "data", "features_v5", "features.db"))
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--fallback-year", type=int, default=None)
    parser.add_argument("--n-per-crop", type=int, default=None)
    parser.add_argument("--min-stages", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=120, help="Optuna trials (0 = use defaults)")
    parser.add_argument("--exclude-crops", nargs="*", default=None)
    parser.add_argument("--no-sar-features", dest="drop_sar", action="store_true", default=False,
                        help="Drop all SAR (VV/VH/CR/RVI) features and their null-indicators "
                             "(A/B diagnostic for SAR-missingness shortcut)")
    parser.add_argument("--tag", default=None, help="Suffix for the run folder name")

    parser.add_argument("--preset", choices=list(PRESETS), default="fix",
                        help="fix (B+C+E, default) | ablate-date (A diagnostic) | baseline (~v5)")
    # The four knobs default to None so a preset fills them unless explicitly overridden.
    parser.add_argument("--date-mode", choices=["full", "cyclic", "none"], default=None)
    parser.add_argument("--class-balance", dest="class_balance", action="store_true", default=None)
    parser.add_argument("--no-class-balance", dest="class_balance", action="store_false")
    parser.add_argument("--soja-ss-weight", type=float, default=None,
                        help="Multiplier for second-season SOJA samples (1.0 = off)")
    parser.add_argument("--abstain", dest="abstain", action="store_true", default=None)
    parser.add_argument("--no-abstain", dest="abstain", action="store_false")
    parser.add_argument("--abstain-target-acc", type=float, default=0.93)
    parser.add_argument("--ss-months", type=int, nargs="*", default=DEFAULT_SECOND_SEASON_MONTHS,
                        help="Planting months counted as second-season SOJA (default 1 2 3)")
    args = parser.parse_args()

    # Resolve preset, letting explicit flags win.
    cfg = dict(PRESETS[args.preset])
    if args.date_mode is not None:    cfg["date_mode"] = args.date_mode
    if args.class_balance is not None: cfg["class_balance"] = args.class_balance
    if args.soja_ss_weight is not None: cfg["soja_ss_weight"] = args.soja_ss_weight
    if args.abstain is not None:      cfg["abstain"] = args.abstain
    logger.info("Preset=%s -> %s", args.preset, cfg)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or args.preset
    output_dir = os.path.join("src", "model", "runs_v6", f"{run_ts}_{tag}")

    X, y, feature_names, le, meta = load_data(
        args.db, min_stages=args.min_stages, planting_year=args.year,
        n_per_crop=args.n_per_crop, fallback_year=args.fallback_year,
        exclude_crops=args.exclude_crops, date_mode=cfg["date_mode"],
        second_season_months=args.ss_months, drop_sar=args.drop_sar,
    )

    train_and_evaluate(
        X, y, feature_names, le, meta,
        n_folds=args.folds, optuna_trials=args.trials, output_dir=output_dir,
        excluded_crops=args.exclude_crops, date_mode=cfg["date_mode"],
        class_balance=cfg["class_balance"], soja_ss_weight=cfg["soja_ss_weight"],
        abstain=cfg["abstain"], abstain_target_acc=args.abstain_target_acc,
        second_season_months=args.ss_months,
    )
    logger.info("Run output: %s", output_dir)
