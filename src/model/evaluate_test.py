import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import re
import sqlite3
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
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

# v6 model was trained on v5-pipeline features (red-edge NDRE/CIRE/MTCI/PSRI/NDMI +
# SAR + null-stage interpolation). Test features MUST be extracted with the same
# pipeline and engineered with the same v6 functions or the column space won't match.
from src.pipelines import phenology_feature_pipeline_v5 as pipeline_mod
from src.pipelines.phenology_feature_pipeline_v5 import PhenologyFeaturePipeline
from src.model.train_xgboost_v6 import (
    engineer_features,
    add_null_indicators,
    season_diagnostic,
    TEXT_COLS,
    NON_FEATURE_COLS,
    DEFAULT_SECOND_SEASON_MONTHS,
    ABSTAIN_LABEL,
)
from src.data_ingestion.request_sentinel_v1 import SentinelHubService

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Typical full-cycle length per crop (days from planting to harvest).
# Used to estimate planting_date when the filename contains "plantio_nan".
CROP_CYCLE_DAYS = {
    "SOJA": 130, "MILHO": 140, "FEIJAO": 90, "ARROZ": 145,
    "TRIGO": 135, "AVEIA": 130, "CAFE": 270,
}

FILENAME_REGEX_TEST = re.compile(
    r"(.+?)_(.+)_plantio_([\d]{2}-[\d]{2}-[\d]{2}|nan)_colheita_([\d]{2}-[\d]{2}-[\d]{2}|nan)\.kml",
    re.IGNORECASE,
)


def _parse_metadata_test(filename: str) -> dict | None:
    """Extended parser that handles 'nan' planting/harvest dates."""
    match = FILENAME_REGEX_TEST.match(filename)
    if not match:
        return None

    crop_raw, raw_id, plantio_str, colheita_str = match.groups()
    crop_key = pipeline_mod._normalize_crop_name(crop_raw)
    if crop_key is None:
        return None

    dt_plantio = None
    dt_colheita = None

    if plantio_str != "nan":
        try:
            dt_plantio = datetime.strptime(plantio_str, "%d-%m-%y")
        except ValueError:
            pass

    if colheita_str != "nan":
        try:
            dt_colheita = datetime.strptime(colheita_str, "%d-%m-%y")
        except ValueError:
            pass

    cycle = CROP_CYCLE_DAYS.get(crop_key, 140)
    if dt_plantio is None and dt_colheita is not None:
        dt_plantio = dt_colheita - timedelta(days=cycle)
    if dt_colheita is None and dt_plantio is not None:
        dt_colheita = dt_plantio + timedelta(days=cycle)

    if dt_plantio is None:
        return None

    return {
        "crop_label": crop_key,
        "field_id": f"{crop_key}_{raw_id}",
        "planting_date": dt_plantio,
        "harvest_date": dt_colheita,
    }


# ── Step 1: Feature Extraction ──────────────────────────────────────────────

def extract_test_features(
    test_kml_dir: str,
    output_dir: str,
    max_workers: int = 2,
    max_per_crop: int | None = None,
) -> pd.DataFrame:
    original_parser = pipeline_mod.parse_metadata_from_filename
    pipeline_mod.parse_metadata_from_filename = _parse_metadata_test
    try:
        service = SentinelHubService()
        pipeline = PhenologyFeaturePipeline(
            service,
            output_dir=output_dir,
            max_workers=max_workers,
            use_stats_api=True,
            fetch_sar=True,
        )
        df = pipeline.process_directory(
            root_dir=test_kml_dir,
            save_tiffs=False,
            max_per_crop=max_per_crop,
        )
    finally:
        pipeline_mod.parse_metadata_from_filename = original_parser
    return df


# ── Step 2: Evaluate ────────────────────────────────────────────────────────

def evaluate_model(
    run_dir: str,
    test_db_path: str,
    output_dir: str,
    min_stages: int = 3,
    exclude_field_ids: set[str] | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # -- Load the training run's config first (drives date_mode + season months) --
    with open(os.path.join(run_dir, "metrics.json")) as f:
        train_metrics = json.load(f)
    cfg = train_metrics.get("config", {})
    date_mode = cfg.get("date_mode", "full")
    ss_months = cfg.get("second_season_months", DEFAULT_SECOND_SEASON_MONTHS)
    logger.info("Eval against run=%s | date_mode=%s | second_season_months=%s",
                run_dir, date_mode, ss_months)

    # -- Load test features from DB -------------------------------------------
    with sqlite3.connect(test_db_path) as conn:
        df = pd.read_sql("SELECT * FROM phenology_features", conn)

    if df.empty:
        logger.error("Test features table is empty")
        return

    logger.info("Test data loaded: %d samples", len(df))

    # -- Sanity check: the v6 model needs red-edge indices in the test DB ------
    if not any(c.startswith("NDRE_") for c in df.columns):
        logger.error(
            "Test DB has no red-edge (NDRE/CIRE/MTCI/PSRI/NDMI) columns — it was "
            "built with an old pipeline. Re-extract with phenology_feature_pipeline_v5 "
            "before evaluating the v6 model (run without --skip-extraction)."
        )
        return

    # -- Drop leaked fields (e.g. augmentation fields that also live in test) --
    if exclude_field_ids:
        before = len(df)
        df = df[~df["field_id"].isin(exclude_field_ids)].reset_index(drop=True)
        logger.info("Excluded %d leaked field(s) from test set", before - len(df))

    for col in df.columns:
        if col not in TEXT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if min_stages > 0:
        before = len(df)
        df = df[df["stages_covered"] >= min_stages].reset_index(drop=True)
        logger.info("Filtered stages_covered >= %d: %d -> %d", min_stages, before, len(df))

    # -- Same feature engineering as training (same date_mode!) ----------------
    df = engineer_features(df, date_mode=date_mode)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].astype(np.float32)
    X = add_null_indicators(X)

    # -- Select features used during training ---------------------------------
    with open(os.path.join(run_dir, "selected_features.json")) as f:
        selected_features = json.load(f)

    missing = [feat for feat in selected_features if feat not in X.columns]
    if missing:
        logger.warning(
            "%d features from training missing in test data, filling with NaN: %s",
            len(missing), missing[:5],
        )
        for feat in missing:
            X[feat] = np.nan

    X_sel = X[selected_features].copy()

    # -- Ground truth ---------------------------------------------------------
    le = LabelEncoder()
    le.classes_ = np.array(train_metrics["classes"])

    known_mask = df["crop_label"].isin(le.classes_)
    if not known_mask.all():
        n_drop = (~known_mask).sum()
        logger.warning("Dropping %d samples with unknown crop labels", n_drop)
        # Filter BOTH frames by the same boolean mask before any index reset —
        # resetting df first and then indexing X_sel by df.index silently grabs
        # the wrong (first-M positional) rows and misaligns X from y.
        X_sel = X_sel[known_mask.to_numpy()].reset_index(drop=True)
        df = df[known_mask].reset_index(drop=True)

    y_true = le.transform(df["crop_label"])

    # -- Season metadata (for the diagnostic that actually matters) ------------
    pmonth = pd.to_datetime(df["planting_date"], errors="coerce").dt.month
    crop_upper = df["crop_label"].str.upper()
    meta = pd.DataFrame({
        "crop_label": df["crop_label"].values,
        "planting_month": pmonth.values,
        "is_soja_second_season": ((crop_upper == "SOJA") & pmonth.isin(ss_months)).values,
    })

    # -- Load model & predict -------------------------------------------------
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(run_dir, "xgboost_crop_classifier.json"))

    proba = model.predict_proba(X_sel)
    y_pred = proba.argmax(axis=1)

    # -- Metrics --------------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")

    logger.info("TEST  Accuracy:    %.2f%%", acc * 100)
    logger.info("TEST  F1 macro:    %.4f", f1m)
    logger.info("TEST  F1 weighted: %.4f", f1w)

    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=le.classes_)
    print(f"\nTEST Classification Report:\n{report_str}")

    # -- Season-stratified diagnostic (THE safrinha number) -------------------
    diag = season_diagnostic(y_true, y_pred, le, meta)

    # -- Abstain policy (E): apply the trained gate to the test predictions ----
    abstain = _apply_abstain_policy(run_dir, proba, y_true, le, meta)

    # -- Confusion matrix -----------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, le.classes_, output_dir)

    # -- Save test metrics ----------------------------------------------------
    test_metrics = {
        "timestamp": datetime.now().isoformat(),
        "train_run": run_dir,
        "n_test_samples": int(len(y_true)),
        "n_features": int(X_sel.shape[1]),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1m, 4),
        "f1_weighted": round(f1w, 4),
        "null_rate": round(float(X_sel.isna().mean().mean()), 4),
        "season_diagnostic": diag,
        "abstain": abstain,
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }
            for cls in le.classes_
            if cls in report
        },
        "train_vs_test": {
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": round(acc, 4),
            "train_f1_macro": train_metrics["f1_macro"],
            "test_f1_macro": round(f1m, 4),
            "overfit_gap_acc": round(train_metrics["accuracy"] - acc, 4),
            "overfit_gap_f1": round(train_metrics["f1_macro"] - f1m, 4),
            "train_second_season_soja_recall":
                train_metrics.get("season_diagnostic", {}).get("soja_second_season_recall"),
            "test_second_season_soja_recall": diag.get("soja_second_season_recall"),
        },
    }

    metrics_out = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_out, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Saved: %s", metrics_out)

    # -- Train vs Test comparison table ---------------------------------------
    print("\nTrain vs Test per-class F1:")
    print(f"{'Crop':<10} {'Train F1':>10} {'Test F1':>10} {'Delta':>10}")
    print("-" * 42)
    for cls in le.classes_:
        train_f1 = train_metrics["per_class"].get(cls, {}).get("f1", 0)
        test_f1 = report.get(cls, {}).get("f1-score", 0)
        delta = test_f1 - train_f1
        print(f"{cls:<10} {train_f1:>10.4f} {test_f1:>10.4f} {delta:>+10.4f}")


def _apply_abstain_policy(run_dir, proba, y_true, le, meta) -> dict | None:
    """Apply the trained max-softmax-prob gate to the test predictions and report
    coverage / covered accuracy, plus how second-season SOJA fares under the gate."""
    policy_path = os.path.join(run_dir, "abstain_policy.json")
    if not os.path.exists(policy_path):
        logger.info("No abstain_policy.json in run — skipping abstain evaluation")
        return None
    with open(policy_path) as f:
        policy = json.load(f)
    t = policy["threshold"]

    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    covered = conf >= t
    correct = pred == y_true

    coverage = float(covered.mean())
    covered_acc = float(correct[covered].mean()) if covered.any() else None

    ss = meta["is_soja_second_season"].to_numpy()
    name_true = np.array(le.classes_)[y_true]
    name_pred = np.array(le.classes_)[pred]
    ss_covered = ss & covered
    ss_recall_covered = (
        float((name_pred[ss_covered] == "SOJA").mean()) if ss_covered.any() else None
    )

    out = {
        "threshold": t,
        "abstain_label": ABSTAIN_LABEL,
        "coverage": round(coverage, 4),
        "covered_accuracy": None if covered_acc is None else round(covered_acc, 4),
        "n_abstained": int((~covered).sum()),
        "second_season_soja_total": int(ss.sum()),
        "second_season_soja_abstained": int((ss & ~covered).sum()),
        "second_season_soja_recall_covered":
            None if ss_recall_covered is None else round(ss_recall_covered, 4),
    }
    logger.info("── Abstain policy applied to TEST (t=%.2f) ──", t)
    logger.info("  coverage=%.1f%% | covered acc=%s | abstained=%d",
                coverage * 100, _fmt(covered_acc), out["n_abstained"])
    logger.info("  2nd-season SOJA: %d total, %d abstained, covered recall=%s",
                out["second_season_soja_total"], out["second_season_soja_abstained"],
                _fmt(ss_recall_covered))
    return out


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


# ── Plots ────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, class_names, output_dir: str):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("TEST Confusion Matrix (counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("TEST Confusion Matrix (% per class)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(output_dir, "test_confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained v6 model on held-out test KMLs")
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Training run directory (e.g. src/model/runs_v6/20260620_055235_aug_febmar)",
    )
    parser.add_argument(
        "--test-kml-dir", type=str,
        default=os.path.join("src", "data", "dataset_split", "test"),
    )
    parser.add_argument(
        "--test-features-dir", type=str,
        default=os.path.join("src", "data", "features_test_v5"),
    )
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip feature extraction, use existing test features.db")
    parser.add_argument("--max-per-crop", type=int, default=None,
                        help="Max KMLs to process per crop (e.g. 50 for a quick sample)")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--min-stages", type=int, default=3)
    parser.add_argument("--exclude-field-ids", type=str, default=None,
                        help="Comma-separated field_ids to drop from the test set "
                             "(e.g. fields that leaked into training augmentation)")
    args = parser.parse_args()

    test_db_path = os.path.join(args.test_features_dir, "features.db")
    eval_output_dir = os.path.join(args.run_dir, "test_eval")
    exclude_ids = (
        {s.strip() for s in args.exclude_field_ids.split(",") if s.strip()}
        if args.exclude_field_ids else None
    )

    if not args.skip_extraction:
        logger.info("Step 1/2: Extracting features from test KMLs (v5 pipeline)...")
        extract_test_features(
            args.test_kml_dir, args.test_features_dir,
            args.max_workers, args.max_per_crop,
        )
    else:
        if not os.path.exists(test_db_path):
            raise FileNotFoundError(
                f"Test features DB not found: {test_db_path}\n"
                "Run without --skip-extraction first."
            )
        logger.info("Skipping extraction, using existing: %s", test_db_path)

    logger.info("Step 2/2: Evaluating model...")
    evaluate_model(args.run_dir, test_db_path, eval_output_dir,
                   min_stages=args.min_stages, exclude_field_ids=exclude_ids)
