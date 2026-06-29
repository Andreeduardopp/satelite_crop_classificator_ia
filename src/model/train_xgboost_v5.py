import os
import sqlite3
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
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

STAGES = ["baseline", "emergence", "vegetative", "flowering", "grain_fill", "maturity"]
# Red-edge (NDRE/CIRE/MTCI), senescence (PSRI) and moisture (NDMI) indices added
# to target the AVEIA<->TRIGO confusion. Engineered features (deltas, peaks,
# greenup/senescence rates, drydown ratios) are generated for every index here.
INDICES = ["NDVI", "NDWI", "EVI", "NDRE", "CIRE", "MTCI", "PSRI", "NDMI"]
SAR_INDICES = ["VV", "VH", "CR", "RVI"]
STATS_BASE = ["mean", "median", "std", "p10", "p90"]

NULL_INDICATOR_THRESHOLD = 0.10


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Planting day-of-year (key for winter/summer crop separation)
    if "planting_date" in df.columns:
        doy = pd.to_datetime(df["planting_date"], errors="coerce").dt.dayofyear
        df["planting_doy"] = doy
        df["planting_doy_sin"] = np.sin(2 * np.pi * doy / 365)
        df["planting_doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # 2. Stage-to-stage deltas
    for idx in INDICES:
        for stat in ["mean", "median"]:
            for i in range(len(STAGES) - 1):
                s_cur = STAGES[i]
                s_nxt = STAGES[i + 1]
                col_cur = f"{idx}_{stat}_{s_cur}"
                col_nxt = f"{idx}_{stat}_{s_nxt}"
                if col_cur in df.columns and col_nxt in df.columns:
                    df[f"{idx}_{stat}_delta_{s_cur}_to_{s_nxt}"] = (
                        df[col_nxt] - df[col_cur]
                    )

    # 3. Peak stage indicator
    for idx in INDICES:
        stage_cols = [f"{idx}_mean_{s}" for s in STAGES if f"{idx}_mean_{s}" in df.columns]
        if stage_cols:
            subset = df[stage_cols]
            has_data = subset.notna().any(axis=1)
            stage_map = {f"{idx}_mean_{s}": float(i) for i, s in enumerate(STAGES)}
            df[f"{idx}_peak_stage"] = np.nan
            if has_data.any():
                df.loc[has_data, f"{idx}_peak_stage"] = (
                    subset.loc[has_data].idxmax(axis=1).map(stage_map)
                )
            df[f"{idx}_peak_value"] = subset.max(axis=1)
            df[f"{idx}_min_value"] = subset.min(axis=1)
            df[f"{idx}_amplitude"] = df[f"{idx}_peak_value"] - df[f"{idx}_min_value"]

    # 4. Greenup rate
    for idx in INDICES:
        baseline_col = f"{idx}_mean_baseline"
        if baseline_col in df.columns and f"{idx}_peak_value" in df.columns:
            peak_stage = df[f"{idx}_peak_stage"].replace(0, np.nan)
            df[f"{idx}_greenup_rate"] = (
                (df[f"{idx}_peak_value"] - df[baseline_col]) / peak_stage
            )

    # 5. Senescence rate
    for idx in INDICES:
        maturity_col = f"{idx}_mean_maturity"
        if maturity_col in df.columns and f"{idx}_peak_value" in df.columns:
            stages_after = (len(STAGES) - 1 - df[f"{idx}_peak_stage"]).replace(0, np.nan)
            df[f"{idx}_senescence_rate"] = (
                (df[f"{idx}_peak_value"] - df[maturity_col]) / stages_after
            )

    # 6. Cross-index ratios at key stages
    for stage in ["vegetative", "flowering", "grain_fill"]:
        ndvi_col = f"NDVI_mean_{stage}"
        ndwi_col = f"NDWI_mean_{stage}"
        evi_col = f"EVI_mean_{stage}"
        if ndvi_col in df.columns and evi_col in df.columns:
            denom = df[evi_col].replace(0, np.nan)
            df[f"NDVI_EVI_ratio_{stage}"] = df[ndvi_col] / denom
        if ndvi_col in df.columns and ndwi_col in df.columns:
            denom = df[ndwi_col].replace(0, np.nan)
            df[f"NDVI_NDWI_ratio_{stage}"] = df[ndvi_col] / denom

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
            p10 = f"{idx}_p10_{stage}"
            p90 = f"{idx}_p90_{stage}"
            if p10 in df.columns and p90 in df.columns:
                spreads.append(df[p90] - df[p10])
        if spreads:
            df[f"{idx}_mean_spread"] = pd.concat(spreads, axis=1).mean(axis=1)

    # 9. Cumulative index
    for idx in INDICES:
        mean_cols = [f"{idx}_mean_{s}" for s in STAGES if f"{idx}_mean_{s}" in df.columns]
        if mean_cols:
            df[f"{idx}_cumulative"] = df[mean_cols].sum(axis=1)

    # 10. Late-stage divergence features (targets TRIGO/AVEIA and FEIJAO/SOJA separation)
    for stage in ["grain_fill", "maturity"]:
        ndvi_col = f"NDVI_mean_{stage}"
        evi_col = f"EVI_mean_{stage}"
        ndwi_col = f"NDWI_mean_{stage}"
        ndvi_std = f"NDVI_std_{stage}"
        evi_std = f"EVI_std_{stage}"
        if ndvi_col in df.columns and evi_col in df.columns:
            df[f"NDVI_minus_EVI_{stage}"] = df[ndvi_col] - df[evi_col]
        if ndvi_col in df.columns and ndwi_col in df.columns:
            df[f"NDVI_minus_NDWI_{stage}"] = df[ndvi_col] - df[ndwi_col]
        if ndvi_std in df.columns and evi_std in df.columns:
            denom = df[evi_std].replace(0, np.nan)
            df[f"std_ratio_NDVI_EVI_{stage}"] = df[ndvi_std] / denom

    # 11. Full-cycle shape: ratio of early vs late means
    for idx in INDICES:
        early_cols = [f"{idx}_mean_{s}" for s in ["baseline", "emergence"] if f"{idx}_mean_{s}" in df.columns]
        late_cols = [f"{idx}_mean_{s}" for s in ["grain_fill", "maturity"] if f"{idx}_mean_{s}" in df.columns]
        if early_cols and late_cols:
            early_mean = df[early_cols].mean(axis=1)
            late_mean = df[late_cols].mean(axis=1)
            denom = late_mean.replace(0, np.nan)
            df[f"{idx}_early_late_ratio"] = early_mean / denom

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
                lambda x: stage_to_i.get(x.replace(f"{idx}_mean_", ""), np.nan)
            ).values
            cumsum = vs.cumsum(axis=1)
            half_max = cumsum.max(axis=1) / 2.0
            reached = (cumsum >= half_max.values.reshape(-1, 1)).idxmax(axis=1).map(
                lambda x: stage_to_i.get(x.replace(f"{idx}_mean_", ""), np.nan)
            )
            df.loc[valid, "NDVI_c4_cumulative_half_stage"] = reached.values
    for idx in ["NDVI", "EVI", "NDWI"]:
        for stage in ["vegetative", "flowering"]:
            bl = f"{idx}_mean_baseline"
            sc = f"{idx}_mean_{stage}"
            if bl in df.columns and sc in df.columns:
                denom = df[sc].replace(0, np.nan)
                ratio = (denom - df[bl]) / denom
                df[f"{idx}_rise_{stage}_vs_baseline"] = ratio

    # 13. Late-season drydown ratios — TRIGO senesces more completely than AVEIA:
    #     ratio captures relative water/greenness retention, not just absolute level.
    for idx in INDICES:
        gf_col = f"{idx}_mean_grain_fill"
        mat_col = f"{idx}_mean_maturity"
        if gf_col in df.columns and mat_col in df.columns:
            denom = df[gf_col].replace(0, np.nan)
            df[f"{idx}_maturity_vs_grainfill_ratio"] = df[mat_col] / denom

    # 14. Vegetative-to-maturity total drop — captures full drydown magnitude
    for idx in INDICES:
        veg_col = f"{idx}_mean_vegetative"
        mat_col = f"{idx}_mean_maturity"
        if veg_col in df.columns and mat_col in df.columns:
            df[f"{idx}_veg_to_maturity_drop"] = df[veg_col] - df[mat_col]

    _assert_no_label_leakage(df)
    return df


def _assert_no_label_leakage(df: pd.DataFrame) -> None:
    leaked = [c for c in ("is_c4", "is_c4_x_NDVI_peak_stage",
                          "is_c4_x_EVI_senescence_rate") if c in df.columns]
    if leaked:
        raise ValueError(f"Label-derived features present (target leakage): {leaked}")


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
) -> tuple[pd.DataFrame, np.ndarray, list[str], LabelEncoder]:
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
                    logger.info("  %s: %d from %d + %d from %d",
                                crop, len(taken), planting_year, min(len(fb), gap), fallback_year)
                else:
                    logger.info("  %s: %d from %d (no fallback available)",
                                crop, len(taken), planting_year)
            else:
                logger.info("  %s: %d from %d", crop, len(taken), planting_year)
        df = pd.concat(parts, ignore_index=True)
        logger.info("Sampled up to %d per crop -> %d samples total", n_per_crop, len(df))

    elif planting_year is not None:
        before = len(df)
        df = df[years_col == planting_year].reset_index(drop=True)
        logger.info("Filtered planting_year == %d: %d -> %d samples", planting_year, before, len(df))

    elif n_per_crop is not None:
        parts = [
            g.sample(min(len(g), n_per_crop), random_state=42)
            for _, g in df.groupby("crop_label")
        ]
        df = pd.concat(parts, ignore_index=True)
        logger.info("Sampled up to %d per crop -> %d samples total", n_per_crop, len(df))

    df = engineer_features(df)

    feature_cols = [
        c for c in df.columns
        if c not in TEXT_COLS and c not in {"stages_covered", "sar_backfill_done"}
    ]
    X = df[feature_cols].astype(np.float32)
    X = add_null_indicators(X)
    feature_cols = list(X.columns)

    le = LabelEncoder()
    y = le.fit_transform(df["crop_label"])

    logger.info(
        "Loaded %d samples, %d features, %d classes: %s",
        len(df), len(feature_cols), len(le.classes_), list(le.classes_),
    )
    logger.info("Null rate: %.1f%%", X.isna().mean().mean() * 100)
    return X, y, feature_cols, le


# ── Feature Selection ────────────────────────────────────────────────────────

def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    keep_ratio: float = 0.60,
) -> tuple[pd.DataFrame, list[str]]:
    quick_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        tree_method="hist", random_state=42, n_jobs=-1,
        objective="multi:softprob", eval_metric="mlogloss",
    )
    quick_model.fit(X, y)

    importance = quick_model.get_booster().get_score(importance_type="gain")
    feat_scores = {}
    for key, val in importance.items():
        if key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            name = feature_names[idx] if idx < len(feature_names) else key
        else:
            name = key
        feat_scores[name] = val

    for f in feature_names:
        if f not in feat_scores:
            feat_scores[f] = 0.0

    sorted_feats = sorted(feat_scores.items(), key=lambda x: x[1], reverse=True)
    n_keep = max(10, int(len(sorted_feats) * keep_ratio))
    selected = [f for f, _ in sorted_feats[:n_keep]]

    dropped = len(feature_names) - len(selected)
    logger.info("Feature selection: kept %d / %d features (dropped %d with lowest gain)",
                len(selected), len(feature_names), dropped)
    return X[selected], selected


# ── Optuna Hyperparameter Tuning ─────────────────────────────────────────────

def tune_xgb(
    X: pd.DataFrame,
    y: np.ndarray,
    n_trials: int = 120,
    n_folds: int = 5,
) -> dict:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

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
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_v5",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=15),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best XGB F1 macro: %.4f", study.best_value)
    logger.info("Best XGB params: %s", study.best_params)
    return study.best_params


# ── Training & Evaluation ────────────────────────────────────────────────────

def train_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    le: LabelEncoder,
    n_folds: int = 5,
    optuna_trials: int = 120,
    output_dir: str = "src/model/output_v5",
    excluded_crops: list[str] | None = None,
) -> xgb.XGBClassifier:
    os.makedirs(output_dir, exist_ok=True)

    X_sel, sel_features = select_features(X, y, feature_names, keep_ratio=0.60)
    logger.info("Selected features: %s", sel_features[:10])

    with open(os.path.join(output_dir, "selected_features.json"), "w") as f:
        json.dump(sel_features, f, indent=2)

    if optuna_trials > 0:
        logger.info("Tuning XGBoost with %d trials...", optuna_trials)
        best_params = tune_xgb(X_sel, y, n_trials=optuna_trials, n_folds=n_folds)
        best_params.update({
            "objective": "multi:softprob", "eval_metric": "mlogloss",
            "tree_method": "hist", "random_state": 42, "n_jobs": -1,
        })
    else:
        best_params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            "objective": "multi:softprob", "eval_metric": "mlogloss",
            "tree_method": "hist", "random_state": 42, "n_jobs": -1,
        }

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump({"xgboost": best_params}, f, indent=2, default=str)

    model = xgb.XGBClassifier(**best_params)

    logger.info("Running %d-fold CV...", n_folds)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_sel, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")
    f1w = f1_score(y, y_pred, average="weighted")
    logger.info("XGBoost — Acc: %.2f%% | F1 macro: %.4f | F1 weighted: %.4f",
                acc * 100, f1m, f1w)

    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
    report_str = classification_report(y, y_pred, target_names=le.classes_)
    print(f"\nXGBoost Classification Report:\n" + report_str)

    cm = confusion_matrix(y, y_pred)
    _plot_confusion_matrix(cm, le.classes_, output_dir)

    logger.info("Training final model on all data...")
    model.fit(X_sel, y)

    _plot_feature_importance(model, sel_features, output_dir)
    _plot_feature_importance_by_category(model, sel_features, output_dir)

    model_path = os.path.join(output_dir, "xgboost_crop_classifier.json")
    model.save_model(model_path)
    logger.info("XGB model saved: %s", model_path)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "version": "v5",
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
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }
            for cls in le.classes_
        },
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", metrics_path)

    return model


# ── Plots ────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, class_names: list, output_dir: str):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (% per class)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


def _resolve_feature_name(key: str, feature_names: list[str]) -> str:
    if key.startswith("f") and key[1:].isdigit():
        idx = int(key[1:])
        return feature_names[idx] if idx < len(feature_names) else key
    return key


def _plot_feature_importance(model: xgb.XGBClassifier, feature_names: list[str],
                             output_dir: str, top_n: int = 30):
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        return

    mapped = {_resolve_feature_name(k, feature_names): v for k, v in importance.items()}
    sorted_imp = sorted(mapped.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_imp][::-1]
    values = [x[1] for x in sorted_imp][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(names, values, color=sns.color_palette("viridis", len(names)))
    ax.set_title(f"Top {top_n} Features by Gain", fontsize=14, fontweight="bold")
    ax.set_xlabel("Gain")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


def _plot_feature_importance_by_category(model: xgb.XGBClassifier, feature_names: list[str],
                                         output_dir: str):
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        return

    mapped = {_resolve_feature_name(k, feature_names): v for k, v in importance.items()}

    categories = {
        "Optical base features": 0.0,
        "SAR base features": 0.0,
        "Stage deltas": 0.0,
        "Peak/amplitude": 0.0,
        "Greenup/senescence": 0.0,
        "Cross-index ratios": 0.0,
        "Temporal CV": 0.0,
        "Within-field spread": 0.0,
        "Cumulative": 0.0,
        "Planting date": 0.0,
        "Late-stage divergence": 0.0,
        "Early/late ratio": 0.0,
        "Drydown ratios": 0.0,
        "Null indicators": 0.0,
        "Geo/area": 0.0,
        "Other": 0.0,
    }

    for name, gain in mapped.items():
        if name.endswith("_is_null"):
            categories["Null indicators"] += gain
        elif "planting_doy" in name:
            categories["Planting date"] += gain
        elif name in ("latitude", "longitude", "area_hectares"):
            categories["Geo/area"] += gain
        elif "early_late_ratio" in name:
            categories["Early/late ratio"] += gain
        elif "maturity_vs_grainfill_ratio" in name or "veg_to_maturity_drop" in name:
            categories["Drydown ratios"] += gain
        elif "NDVI_minus_" in name or "std_ratio_" in name:
            categories["Late-stage divergence"] += gain
        elif "delta_" in name:
            categories["Stage deltas"] += gain
        elif any(k in name for k in ["peak_", "amplitude", "min_value"]):
            categories["Peak/amplitude"] += gain
        elif "greenup" in name or "senescence" in name:
            categories["Greenup/senescence"] += gain
        elif "ratio_" in name:
            categories["Cross-index ratios"] += gain
        elif "temporal_cv" in name:
            categories["Temporal CV"] += gain
        elif "mean_spread" in name:
            categories["Within-field spread"] += gain
        elif "cumulative" in name:
            categories["Cumulative"] += gain
        elif any(f"{idx}_{st}_{sg}" == name for idx in SAR_INDICES for st in STATS_BASE for sg in STAGES):
            categories["SAR base features"] += gain
        elif any(f"{idx}_{st}_{sg}" == name for idx in INDICES for st in STATS_BASE for sg in STAGES):
            categories["Optical base features"] += gain
        else:
            categories["Other"] += gain

    categories = {k: v for k, v in categories.items() if v > 0}
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_cats][::-1]
    values = [x[1] for x in sorted_cats][::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(names))
    ax.barh(names, values, color=colors)
    ax.set_title("Feature Importance by Category (total gain)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Gain")
    plt.tight_layout()
    path = os.path.join(output_dir, "importance_by_category.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost crop classifier v5")
    parser.add_argument("--db", default=os.path.join("src", "data", "features_v5", "features.db"),
                        help="Path to features SQLite DB")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--fallback-year", type=int, default=None)
    parser.add_argument("--n-per-crop", type=int, default=None)
    parser.add_argument("--min-stages", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=120,
                        help="Optuna trials for XGBoost (default 120)")
    parser.add_argument("--exclude-crops", nargs="*", default=None,
                        help="Crop labels to drop before training (e.g. ARROZ CAFE)")
    parser.add_argument("--tag", default=None,
                        help="Suffix appended to the run folder name (e.g. 5crops)")
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_ts}_{args.tag}" if args.tag else run_ts
    output_dir = os.path.join("src", "model", "runs_v5", run_name)

    X, y, feature_names, le = load_data(
        args.db,
        min_stages=args.min_stages,
        planting_year=args.year,
        n_per_crop=args.n_per_crop,
        fallback_year=args.fallback_year,
        exclude_crops=args.exclude_crops,
    )

    model = train_and_evaluate(
        X, y, feature_names, le,
        n_folds=args.folds,
        optuna_trials=args.trials,
        output_dir=output_dir,
        excluded_crops=args.exclude_crops,
    )
    logger.info("Run output: %s", output_dir)
