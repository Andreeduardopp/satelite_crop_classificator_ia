import os
import sys
import sqlite3
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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

FEATURE_COLS = None  # derived at runtime from the DB


def load_data(db_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], LabelEncoder]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM phenology_features", conn)
    if df.empty:
        raise ValueError("phenology_features table is empty")

    for col in df.columns:
        if col not in TEXT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feature_cols = [c for c in df.columns if c not in TEXT_COLS and c not in {"stages_covered"}]
    X = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["crop_label"])

    logger.info("Loaded %d samples, %d features, %d classes: %s",
                len(df), X.shape[1], len(le.classes_), list(le.classes_))
    logger.info("Null rate: %.1f%%", np.isnan(X).mean() * 100)
    return df, X, y, feature_cols, le


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    le: LabelEncoder,
    n_folds: int = 5,
    output_dir: str = "src/model/output",
) -> xgb.XGBClassifier:
    os.makedirs(output_dir, exist_ok=True)

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
    X_df = pd.DataFrame(X, columns=feature_names)
    model.fit(X_df, y)

    _plot_feature_importance(model, feature_names, output_dir)

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
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "null_rate": round(float(np.isnan(X).mean()), 4),
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


def _plot_feature_importance(model: xgb.XGBClassifier, feature_names: list[str], output_dir: str, top_n: int = 30):
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        return

    mapped = {}
    for key, val in importance.items():
        if key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            name = feature_names[idx] if idx < len(feature_names) else key
        else:
            name = key
        mapped[name] = val

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


if __name__ == "__main__":
    db_path = os.path.join("src", "data", "features", "features.db")
    output_dir = os.path.join("src", "model", "output")

    df, X, y, feature_names, le = load_data(db_path)
    model = train_and_evaluate(X, y, feature_names, le, n_folds=5, output_dir=output_dir)
