"""AVEIA-vs-TRIGO separability probe.

Goal: before designing v7 discriminator features, find out *where the signal is*
for the only pair that matters (78% of the 5-crop benchmark's errors are AVEIA<->TRIGO).

It trains a 2-class XGBoost on the existing v5 features and breaks feature gain down
two ways:
  - by INDEX FAMILY  : red-edge {NDRE,CIRE,MTCI,PSRI,NDMI} vs broadband {NDVI,NDWI,EVI}
                       vs SAR {VV,VH,CR,RVI} vs DATE {planting_doy*}
  - by FEATURE TYPE  : level (per-stage stat) vs timing (rates/peaks/deltas/drydown)
                       vs cross-index (over/minus between two indices)

If red-edge LEVEL features don't dominate, the C3/C4 red-edge logic that fixed
SOJA<->MILHO will NOT transfer, and v7 should invest in timing + SAR instead.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd, sqlite3
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from train_xgboost_v6 import engineer_features, add_null_indicators, NON_FEATURE_COLS, TEXT_COLS
from features_v7 import add_v7_timing_features

TRAIN_DB = "src/data/features_v5/features.db"
TEST_DB = "src/data/features_test_v5/features.db"
PAIR = ["AVEIA", "TRIGO"]
RED_EDGE = {"NDRE", "CIRE", "MTCI", "PSRI", "NDMI"}
BROADBAND = {"NDVI", "NDWI", "EVI"}
SAR = {"VV", "VH", "CR", "RVI"}
TIMING_HINTS = ("peak_stage", "_rate", "greenup", "senescence", "drydown", "_delta",
                "_rise_", "vs_baseline", "_ratio", "_drop", "_fall_", "amplitude",
                "_vs_", "peak_value", "min_value")


def load(db, v7=False):
    df = pd.read_sql("SELECT * FROM phenology_features", sqlite3.connect(db))
    df = df[df["crop_label"].str.upper().isin(PAIR)].reset_index(drop=True)
    for c in df.columns:
        if c not in TEXT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[pd.to_datetime(df["planting_date"], errors="coerce").notna()]
    df = df[df["stages_covered"] >= 3].reset_index(drop=True)
    df = engineer_features(df, date_mode="cyclic")
    if v7:
        df = add_v7_timing_features(df)
    X = df[[c for c in df.columns if c not in NON_FEATURE_COLS]].astype(np.float32)
    X = add_null_indicators(X)
    y = (df["crop_label"].str.upper() == "TRIGO").astype(int).to_numpy()  # 1=TRIGO 0=AVEIA
    return X, y


def evaluate(v7, label):
    Xtr, ytr = load(TRAIN_DB, v7=v7)
    Xte, yte = load(TEST_DB, v7=v7)
    for f in Xtr.columns:
        if f not in Xte.columns:
            Xte[f] = np.nan
    Xte = Xte[Xtr.columns]
    mdl = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                            tree_method="hist", n_jobs=-1, random_state=42)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(mdl, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    cv_auc = roc_auc_score(ytr, oof)
    cv_acc = accuracy_score(ytr, (oof >= 0.5).astype(int))
    mdl.fit(Xtr, ytr)
    pte = mdl.predict_proba(Xte)[:, 1]
    te_auc = roc_auc_score(yte, pte)
    te_acc = accuracy_score(yte, (pte >= 0.5).astype(int))
    print(f"[{label:<10}] feats={Xtr.shape[1]:<4}  CV: acc={cv_acc:.4f} AUC={cv_auc:.4f}   "
          f"TEST: acc={te_acc:.4f} AUC={te_auc:.4f}")
    return mdl, Xtr, cv_auc, te_auc


def family(feat):
    head = feat.split("_")[0]
    if head == "planting":
        return "DATE"
    if head in RED_EDGE:
        return "RED_EDGE"
    if head in BROADBAND:
        return "BROADBAND"
    if head in SAR:
        return "SAR"
    return "OTHER"


def ftype(feat):
    if feat.endswith("_is_null"):
        return "null_ind"
    if "_over_" in feat or "_minus_" in feat:
        return "cross"
    if any(h in feat for h in TIMING_HINTS):
        return "timing"
    return "level"


def _v7_feature_names():
    import features_v7 as fv7
    names = set()
    for idx in fv7.V7_INDICES:
        for suf in ("late_curvature", "drydown_accel", "grainfill_retention", "maturity_retention",
                    "senescence_depth_norm", "senescence_onset_stage", "greenup_halfmax_stage",
                    "rise_fall_span_diff"):
            names.add(f"{idx}_{suf}")
    return names | {"REDEDGE_senescence_depth_consensus", "REDEDGE_late_curvature_consensus"}


def breakdown(mdl):
    """Gain analysis of a fitted model: by index family, feature type, and v7 contribution."""
    gain = mdl.get_booster().get_score(importance_type="gain")
    total = sum(gain.values())
    rows = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
    rank_of = {f: i + 1 for i, (f, _) in enumerate(rows)}

    print(f"\n== Top 15 features by gain (of {len(gain)} used) ==")
    for f, g in rows[:15]:
        print(f"{f:<42}{100*g/total:>6.1f}%  {family(f)} / {ftype(f)}")

    print("\n== Gain by INDEX FAMILY ==")
    fam: dict[str, float] = {}
    for f, g in gain.items():
        fam[family(f)] = fam.get(family(f), 0) + g
    for k, v in sorted(fam.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k:<10} {100*v/total:>5.1f}%")

    print("\n== Gain by FEATURE TYPE ==")
    typ: dict[str, float] = {}
    for f, g in gain.items():
        typ[ftype(f)] = typ.get(ftype(f), 0) + g
    for k, v in sorted(typ.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k:<10} {100*v/total:>5.1f}%")

    v7_names = _v7_feature_names()
    v7_gain = sum(g for f, g in gain.items() if f in v7_names)
    used = sum(1 for f in gain if f in v7_names)
    print(f"\n== New v7 timing features: {100*v7_gain/total:.1f}% of gain ({used} used) ==")
    for f, g in sorted(((f, g) for f, g in gain.items() if f in v7_names),
                       key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  #{rank_of[f]:<4} {f:<40} {100*g/total:.2f}%")


def main():
    print("AVEIA-vs-TRIGO separability — baseline (v6 features) vs +v7 timing features\n")
    _, _, base_cv, base_te = evaluate(v7=False, label="baseline")
    mdl, Xtr, v7_cv, v7_te = evaluate(v7=True, label="+v7 timing")
    print(f"\nDelta from v7 timing features:  CV AUC {v7_cv-base_cv:+.4f}   TEST AUC {v7_te-base_te:+.4f}")
    breakdown(mdl)


if __name__ == "__main__":
    main()
