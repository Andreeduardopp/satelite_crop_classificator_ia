"""Evaluate a trained v7-family run against the 2026 production benchmark.

The benchmark is the frozen kml_test_2026 sample (50/crop from the most recent completed
season; see PLAN_V10_2025_2026.md §2), extracted at `src/data/features_test_2026/`.
Currently 250 of 350 fields (TRIGO/AVEIA arrive with the December 2026 batch). It replaces
the retired 2024-heavy kml_test_sample_250 benchmark (2026-07-09; the old zip is kept only
so sampling scripts can exclude its field_ids).

Rows whose crop is outside the model's classes are dropped automatically, so the same DB
serves the 5/6/7-crop ladder (n=200 for 5-crop, 250 for 6/7-crop).

Writes `<run_dir>/bench2026_eval/bench_metrics.json` (accuracy, per-class precision/recall,
confusion matrix, abstain-threshold sweep). Promotion bar: >=0.95 covered accuracy at
>=0.85 coverage.

Usage:
    python src/model/eval_bench2026.py src/model/best_models/5_culturas_no_aveia
"""
import os
import sys
import json
import argparse
import logging

import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_score, recall_score)

sys.path.insert(0, os.path.dirname(__file__))
from train_xgboost_v7 import _load

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")

BENCH_DB = os.path.join("src", "data", "features_test_2026", "features.db")
SWEEP_THRESHOLDS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
OUT_NAME = "bench2026_eval"


def eval_bench(run_dir, bench_db=BENCH_DB, min_cov=0, dense_timing=False):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        classes = json.load(f)["classes"]
    with open(os.path.join(run_dir, "selected_features.json")) as f:
        sel = json.load(f)

    X, _, _, _, meta = _load(bench_db, min_cov=min_cov, dense_timing=dense_timing)
    known = meta["crop_label"].isin(classes).to_numpy()
    if not known.all():
        logger.info("Dropping %d bench rows with crops outside the model", int((~known).sum()))
        X, meta = X[known].reset_index(drop=True), meta[known].reset_index(drop=True)
    y = np.array([classes.index(c) for c in meta["crop_label"]])

    for c in sel:
        if c not in X.columns:
            X[c] = np.nan
    X = X[sel]

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(run_dir, "xgboost_crop_classifier.json"))
    proba = model.predict_proba(X)
    y_pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)

    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")
    labels = list(range(len(classes)))
    prec = precision_score(y, y_pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=labels)
    present = cm.sum(axis=1) > 0

    sweep = []
    for t in SWEEP_THRESHOLDS:
        covered = conf >= t
        cov = covered.mean()
        cov_acc = accuracy_score(y[covered], y_pred[covered]) if covered.any() else None
        sweep.append({"threshold": t, "coverage": round(float(cov), 4),
                      "covered_accuracy": round(float(cov_acc), 4) if cov_acc is not None else None})

    out = {
        "benchmark": "kml_test_2026 (features_test_2026; TRIGO/AVEIA pending December batch)",
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1m), 4),
        "n": int(len(y)),
        "per_class": {c: {"precision": round(float(p), 4), "recall": round(float(r), 4)}
                      for c, p, r, ok in zip(classes, prec, rec, present) if ok},
        "confusion_matrix": {"labels": classes,
                             "rows_true": [[int(v) for v in row] for row in cm]},
        "abstain_sweep": sweep,
    }
    os.makedirs(os.path.join(run_dir, OUT_NAME), exist_ok=True)
    out_path = os.path.join(run_dir, OUT_NAME, "bench_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nBENCH2026  acc={acc:.4f}  macro-F1={f1m:.4f}  (n={len(y)})")
    for c, p, r, ok in zip(classes, prec, rec, present):
        if ok:
            print(f"  {c:<8} precision={p:.2f}  recall={r:.2f}")
    print(f"Saved: {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a run on the 2026 production benchmark")
    ap.add_argument("run_dir", help="run folder with xgboost_crop_classifier.json + selected_features.json")
    ap.add_argument("--bench-db", default=BENCH_DB)
    ap.add_argument("--min-cov", type=int, default=0)
    ap.add_argument("--dense-timing", action="store_true", default=False)
    args = ap.parse_args()
    eval_bench(args.run_dir, bench_db=args.bench_db, min_cov=args.min_cov,
               dense_timing=args.dense_timing)
