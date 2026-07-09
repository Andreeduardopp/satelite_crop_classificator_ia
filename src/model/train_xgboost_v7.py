"""XGBoost crop classifier v7 — trained on the v6 DENSE temporal series.

v6 (the model) consumed 6 coarse phenological stages. v7 consumes the v6 *pipeline*'s
dense grid: dekadal (P10D) bins over the full season + fine (P5D) bins over
flowering->maturity. The AVEIA-vs-TRIGO pilot showed this dense series carries ~+0.10 AUC
of oats/wheat senescence-timing signal the stages discard (see AVEIA_TRIGO_DISCRIMINATION.md).

This trainer reuses v6's whole training/eval machinery (weighted OOF CV, class-balance +
safrinha-SOJA up-weight, abstain gate, season diagnostic, plots) via train_and_evaluate;
it only swaps the feature builder to the dense schema. First cut uses the RAW dense bins
(they already hit 0.894 AUC on the pair) + cyclic planting date; engineered dense-timing
features are a later iteration.

Usage:
    python src/model/train_xgboost_v7.py --train-db src/data/features_v6/features.db \
        --test-db src/data/features_test_v6/features.db --tag dense_5crop --trials 0
"""
import os
import sys
import json
import argparse
import logging
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from train_xgboost_v6 import (
    train_and_evaluate, add_null_indicators, season_diagnostic,
    DEFAULT_SECOND_SEASON_MONTHS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")

# Bookkeeping/text columns that must never be features (mirrors the v6 dense schema).
NON_FEATURE = {"field_id", "crop_label", "planting_date", "interpolated",
               "dekads_covered", "fine_covered"}


def _load(db_path, exclude_crops=None, min_cov=3, second_season_months=None, dense_timing=False):
    months = second_season_months or DEFAULT_SECOND_SEASON_MONTHS
    df = pd.read_sql("SELECT * FROM phenology_features", sqlite3.connect(db_path))
    if exclude_crops:
        ex = {c.strip().upper() for c in exclude_crops}
        df = df[~df["crop_label"].str.upper().isin(ex)].reset_index(drop=True)
    for c in df.columns:
        if c not in {"field_id", "crop_label", "planting_date", "interpolated"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if dense_timing:
        from features_v7_dense import add_dense_timing_features
        df = add_dense_timing_features(df)

    pdate = pd.to_datetime(df["planting_date"], errors="coerce")
    df = df[pdate.notna()].reset_index(drop=True)
    pdate = pd.to_datetime(df["planting_date"], errors="coerce")
    if min_cov > 0 and "dekads_covered" in df.columns:
        df = df[df["dekads_covered"].fillna(0) >= min_cov].reset_index(drop=True)
        pdate = pd.to_datetime(df["planting_date"], errors="coerce")

    # cyclic planting date (drops the raw monotonic DOY -> no early-year shortcut)
    doy = pdate.dt.dayofyear
    df["planting_doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["planting_doy_cos"] = np.cos(2 * np.pi * doy / 365)

    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    X = df[feat_cols].astype(np.float32)
    # corrupted SAR responses can carry inf (log-scaled zero backscatter);
    # map to NaN so XGBoost treats them as missing instead of aborting
    X = X.replace([np.inf, -np.inf], np.nan)
    X = add_null_indicators(X)

    le = LabelEncoder()
    y = le.fit_transform(df["crop_label"])
    crop_u = df["crop_label"].str.upper()
    meta = pd.DataFrame({
        "crop_label": df["crop_label"].values,
        "planting_month": pdate.dt.month.values,
        "is_soja_second_season": ((crop_u == "SOJA") & pdate.dt.month.isin(months)).values,
    })
    logger.info("Loaded %s: %d fields, %d features, classes=%s",
                os.path.basename(os.path.dirname(db_path)), len(y), X.shape[1], list(le.classes_))
    return X, y, list(X.columns), le, meta


def evaluate(run_dir, test_db, exclude_crops=None, min_cov=3, dense_timing=False):
    """Apply the trained v7 model to a dense test DB; report metrics + AVEIA/TRIGO confusion."""
    with open(os.path.join(run_dir, "metrics.json")) as f:
        classes = json.load(f)["classes"]
    with open(os.path.join(run_dir, "selected_features.json")) as f:
        sel = json.load(f)

    X, y, feats, le, meta = _load(test_db, exclude_crops=exclude_crops, min_cov=min_cov,
                                  dense_timing=dense_timing)
    le = LabelEncoder(); le.classes_ = np.array(classes)
    known = meta["crop_label"].isin(classes).to_numpy()
    if not known.all():
        logger.info("Dropping %d test rows with crops outside the model", int((~known).sum()))
        X, meta = X[known].reset_index(drop=True), meta[known].reset_index(drop=True)
    y = le.transform(meta["crop_label"])

    for c in sel:
        if c not in X.columns:
            X[c] = np.nan
    X = X[sel]

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(run_dir, "xgboost_crop_classifier.json"))
    proba = model.predict_proba(X)
    y_pred = proba.argmax(axis=1)

    # the test DB may lack some model classes (e.g. the 2026 test set has no TRIGO
    # until the December batch) — pin labels so report/matrix stay aligned to the model
    labels = list(range(len(le.classes_)))
    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")
    print(f"\nTEST  acc={acc:.4f}  macro-F1={f1m:.4f}  (n={len(y)})")
    print(classification_report(y, y_pred, labels=labels, target_names=le.classes_,
                                zero_division=0))
    diag = season_diagnostic(y, y_pred, le, meta)

    cls = list(le.classes_)
    if "AVEIA" in cls and "TRIGO" in cls:
        cm = confusion_matrix(y, y_pred, labels=range(len(cls)))
        ai, ti = cls.index("AVEIA"), cls.index("TRIGO")
        swaps = cm[ai][ti] + cm[ti][ai]
        print(f"AVEIA<->TRIGO swaps: {swaps}  (AVEIA->TRIGO {cm[ai][ti]}, TRIGO->AVEIA {cm[ti][ai]})")

    cm_full = confusion_matrix(y, y_pred, labels=labels)
    support = cm_full.sum(axis=1)
    out = {"accuracy": round(acc, 4), "f1_macro": round(f1m, 4), "n_test": int(len(y)),
           "season_diagnostic": diag,
           "per_class": {c: {"recall": round(cm_full[i][i] / support[i], 4)}
                         for i, c in enumerate(le.classes_) if support[i] > 0}}
    os.makedirs(os.path.join(run_dir, "test_eval"), exist_ok=True)
    with open(os.path.join(run_dir, "test_eval", "test_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train XGBoost v7 on the dense v6 temporal series")
    ap.add_argument("--train-db", default="src/data/features_v6/features.db")
    ap.add_argument("--test-db", default="src/data/features_test_v6/features.db")
    ap.add_argument("--exclude-crops", nargs="*", default=None)
    ap.add_argument("--min-cov", type=int, default=3, help="min dekads_covered")
    ap.add_argument("--trials", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--soja-ss-weight", type=float, default=6.0)
    ap.add_argument("--runs-dir", default=os.path.join("src", "model", "runs_v7"))
    ap.add_argument("--tag", default="dense")
    ap.add_argument("--dense-timing", action="store_true", default=False,
                    help="add engineered dense-timing features (Option 1)")
    ap.add_argument("--no-eval", dest="do_eval", action="store_false", default=True)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.runs_dir, f"{ts}_{args.tag}")

    X, y, feats, le, meta = _load(args.train_db, exclude_crops=args.exclude_crops,
                                  min_cov=args.min_cov, dense_timing=args.dense_timing)
    train_and_evaluate(
        X, y, feats, le, meta,
        n_folds=args.folds, optuna_trials=args.trials, output_dir=out_dir,
        excluded_crops=args.exclude_crops, date_mode="cyclic", class_balance=True,
        soja_ss_weight=args.soja_ss_weight, abstain=True,
    )
    logger.info("Run output: %s", out_dir)

    if args.do_eval and os.path.exists(args.test_db):
        evaluate(out_dir, args.test_db, exclude_crops=args.exclude_crops,
                 min_cov=args.min_cov, dense_timing=args.dense_timing)
