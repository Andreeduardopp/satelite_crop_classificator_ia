import os
import sys
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
INDICES = ["NDVI", "NDWI", "EVI"]
STATS_BASE = ["mean", "median", "std", "p10", "p90"]


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Stage-to-stage deltas (greenup / senescence rate)
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

    # 2. Peak stage indicator (which stage has the max mean value per index)
    for idx in INDICES:
        stage_cols = [f"{idx}_mean_{s}" for s in STAGES if f"{idx}_mean_{s}" in df.columns]
        if stage_cols:
            df[f"{idx}_peak_stage"] = df[stage_cols].idxmax(axis=1)
            df[f"{idx}_peak_stage"] = df[f"{idx}_peak_stage"].map(
                {f"{idx}_mean_{s}": i for i, s in enumerate(STAGES)}
            )
            df[f"{idx}_peak_value"] = df[stage_cols].max(axis=1)
            df[f"{idx}_min_value"] = df[stage_cols].min(axis=1)
            df[f"{idx}_amplitude"] = df[f"{idx}_peak_value"] - df[f"{idx}_min_value"]

    # 3. Greenup rate: (peak - baseline) / peak_stage_index
    for idx in INDICES:
        baseline_col = f"{idx}_mean_baseline"
        if baseline_col in df.columns and f"{idx}_peak_value" in df.columns:
            peak_stage = df[f"{idx}_peak_stage"].replace(0, np.nan)
            df[f"{idx}_greenup_rate"] = (
                (df[f"{idx}_peak_value"] - df[baseline_col]) / peak_stage
            )

    # 4. Senescence rate: (peak - maturity) / (n_stages - peak_stage)
    for idx in INDICES:
        maturity_col = f"{idx}_mean_maturity"
        if maturity_col in df.columns and f"{idx}_peak_value" in df.columns:
            stages_after = (len(STAGES) - 1 - df[f"{idx}_peak_stage"]).replace(0, np.nan)
            df[f"{idx}_senescence_rate"] = (
                (df[f"{idx}_peak_value"] - df[maturity_col]) / stages_after
            )

    # 5. Cross-index ratios at key stages
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

    # 6. Coefficient of variation across stages (temporal variability)
    for idx in INDICES:
        mean_cols = [f"{idx}_mean_{s}" for s in STAGES if f"{idx}_mean_{s}" in df.columns]
        if len(mean_cols) >= 3:
            row_mean = df[mean_cols].mean(axis=1)
            row_std = df[mean_cols].std(axis=1)
            df[f"{idx}_temporal_cv"] = row_std / row_mean.replace(0, np.nan)

    # 7. Within-field heterogeneity (mean p90-p10 spread across stages)
    for idx in INDICES:
        spreads = []
        for stage in STAGES:
            p10 = f"{idx}_p10_{stage}"
            p90 = f"{idx}_p90_{stage}"
            if p10 in df.columns and p90 in df.columns:
                spreads.append(df[p90] - df[p10])
        if spreads:
            df[f"{idx}_mean_spread"] = pd.concat(spreads, axis=1).mean(axis=1)

    # 8. Cumulative index (sum of means across stages — proxy for total biomass)
    for idx in INDICES:
        mean_cols = [f"{idx}_mean_{s}" for s in STAGES if f"{idx}_mean_{s}" in df.columns]
        if mean_cols:
            df[f"{idx}_cumulative"] = df[mean_cols].sum(axis=1)

    return df


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data(
    db_path: str,
    min_stages: int = 0,
    use_feature_engineering: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], LabelEncoder]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM phenology_features", conn)
    if df.empty:
        raise ValueError("phenology_features table is empty")

    for col in df.columns:
        if col not in TEXT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if min_stages > 0:
        before = len(df)
        df = df[df["stages_covered"] >= min_stages].reset_index(drop=True)
        logger.info("Filtered stages_covered >= %d: %d -> %d samples", min_stages, before, len(df))

    if use_feature_engineering:
        df = engineer_features(df)
        logger.info("Feature engineering: added derived features")

    feature_cols = [
        c for c in df.columns
        if c not in TEXT_COLS and c not in {"stages_covered"}
    ]
    X = df[feature_cols].astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["crop_label"])

    logger.info(
        "Loaded %d samples, %d features (%d base + %d engineered), %d classes: %s",
        len(df), len(feature_cols),
        91, len(feature_cols) - 91,
        len(le.classes_), list(le.classes_),
    )
    logger.info("Null rate: %.1f%%", X.isna().mean().mean() * 100)
    return df, X, y, feature_cols, le


# ── Optuna Hyperparameter Tuning ─────────────────────────────────────────────

def tune_hyperparameters(
    X: pd.DataFrame,
    y: np.ndarray,
    n_trials: int = 80,
    n_folds: int = 5,
) -> dict:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
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
    study = optuna.create_study(direction="maximize", study_name="xgb_crop_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best F1 macro: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    return study.best_params


# ── Training & Evaluation ────────────────────────────────────────────────────

def train_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    le: LabelEncoder,
    n_folds: int = 5,
    optuna_trials: int = 0,
    output_dir: str = "src/model/output_v2",
) -> xgb.XGBClassifier:
    os.makedirs(output_dir, exist_ok=True)

    if optuna_trials > 0:
        logger.info("Tuning hyperparameters with %d Optuna trials...", optuna_trials)
        best_params = tune_hyperparameters(X, y, n_trials=optuna_trials, n_folds=n_folds)

        best_params.update({
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        })
        model = xgb.XGBClassifier(**best_params)

        params_path = os.path.join(output_dir, "best_params.json")
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2, default=str)
        logger.info("Best params saved: %s", params_path)
    else:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    logger.info("Running %d-fold stratified cross-validation...", n_folds)
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")

    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    logger.info("CV Accuracy:    %.2f%%", acc * 100)
    logger.info("CV F1 (macro):  %.4f", f1_macro)
    logger.info("CV F1 (weighted): %.4f", f1_weighted)

    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
    report_str = classification_report(y, y_pred, target_names=le.classes_)
    print("\n" + report_str)

    cm = confusion_matrix(y, y_pred)
    _plot_confusion_matrix(cm, le.classes_, output_dir)

    logger.info("Training final model on all data...")
    model.fit(X, y)

    _plot_feature_importance(model, feature_names, output_dir)
    _plot_feature_importance_by_category(model, feature_names, output_dir)

    model_path = os.path.join(output_dir, "xgboost_crop_classifier.json")
    model.save_model(model_path)
    logger.info("Model saved: %s", model_path)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(le.classes_)),
        "classes": list(le.classes_),
        "cv_folds": n_folds,
        "optuna_trials": optuna_trials,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
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
        "Base features": 0.0,
        "Stage deltas": 0.0,
        "Peak/amplitude": 0.0,
        "Greenup/senescence": 0.0,
        "Cross-index ratios": 0.0,
        "Temporal CV": 0.0,
        "Within-field spread": 0.0,
        "Cumulative": 0.0,
        "Other": 0.0,
    }

    for name, gain in mapped.items():
        if "delta_" in name:
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
        elif any(f"{idx}_{st}_{sg}" == name for idx in INDICES for st in STATS_BASE for sg in STAGES) or name == "area_hectares":
            categories["Base features"] += gain
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
    db_path = os.path.join("src", "data", "features", "features.db")
    output_dir = os.path.join("src", "model", "output_v2")

    df, X, y, feature_names, le = load_data(
        db_path,
        min_stages=3,
        use_feature_engineering=True,
    )

    model = train_and_evaluate(
        X, y, feature_names, le,
        n_folds=5,
        optuna_trials=80,
        output_dir=output_dir,
    )
