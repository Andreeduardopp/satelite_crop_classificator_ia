"""Path-B pilot probe: does the v6 DENSE temporal series beat the 0.857 AUC ceiling?

Compares AVEIA-vs-TRIGO separability of:
  * v6 dense features (dekadal *_d{k} + fine *_f{k} series), and
  * v5 stage features (engineer_features) on the SAME fields,
matched by field_id so the small pilot sample can't bias the comparison. Averaged
over several CV seeds because the pilot is intentionally small (de-risk before a full
re-extraction).

Geographic columns (lat/lon/area) are dropped from BOTH so we measure the spectral +
temporal signal, not a geographic quirk of this particular pilot draw.
"""
import os, sys, warnings
import numpy as np, pandas as pd, sqlite3
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_xgboost_v6 import engineer_features, add_null_indicators, NON_FEATURE_COLS, TEXT_COLS
from features_v7_dense import add_dense_timing_features

V6_DB = sys.argv[1] if len(sys.argv) > 1 else "src/data/features_v6_matched/features.db"
V5_DB = "src/data/features_v5/features.db"
PAIR = ["AVEIA", "TRIGO"]
GEO_DROP = {"area_hectares", "latitude", "longitude", "dekads_covered", "fine_covered"}
V6_TEXT = {"field_id", "crop_label", "planting_date", "interpolated"}
SEEDS = [42, 7, 123, 2024, 99]


def _cv(X, y, n_est=400, depth=5):
    aucs, accs = [], []
    for s in SEEDS:
        mdl = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                tree_method="hist", n_jobs=-1, random_state=s)
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        oof = cross_val_predict(mdl, X, y, cv=skf, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, oof))
        accs.append(accuracy_score(y, (oof >= 0.5).astype(int)))
    return np.mean(aucs), np.std(aucs), np.mean(accs)


def load_v6(timing=False):
    df = pd.read_sql("SELECT * FROM phenology_features", sqlite3.connect(V6_DB))
    df = df[df["crop_label"].str.upper().isin(PAIR)].reset_index(drop=True)
    if timing:
        df = add_dense_timing_features(df)
    feat = [c for c in df.columns if c not in V6_TEXT and c not in GEO_DROP]
    X = df[feat].apply(pd.to_numeric, errors="coerce").astype("float32")
    y = (df["crop_label"].str.upper() == "TRIGO").astype(int).to_numpy()
    return X, y, df["field_id"], feat


def load_v5(field_ids):
    df = pd.read_sql("SELECT * FROM phenology_features", sqlite3.connect(V5_DB))
    df = df[df["field_id"].isin(set(field_ids))].reset_index(drop=True)
    for c in df.columns:
        if c not in TEXT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = engineer_features(df, date_mode="cyclic")
    drop = NON_FEATURE_COLS | {"latitude", "longitude", "area_hectares"}
    X = df[[c for c in df.columns if c not in drop]].astype("float32")
    X = add_null_indicators(X)
    y = (df["crop_label"].str.upper() == "TRIGO").astype(int).to_numpy()
    return X, y, df["field_id"]


def main():
    Xv6, yv6, ids6, feat6 = load_v6(timing=False)
    print(f"v6 pilot: {len(yv6)} fields (AVEIA {np.sum(yv6==0)}, TRIGO {np.sum(yv6==1)}), "
          f"{Xv6.shape[1]} dense features")
    auc6, sd6, acc6 = _cv(Xv6, yv6)
    print(f"  v6 DENSE (raw bins)    : AUC {auc6:.4f} +/-{sd6:.4f}   acc {acc6:.4f}")

    # Option 1: raw bins + engineered dense-timing features
    Xt, yt, _, featt = load_v6(timing=True)
    auct, sdt, acct = _cv(Xt, yt)
    print(f"  v6 DENSE + timing ({Xt.shape[1]-Xv6.shape[1]:+d}f): AUC {auct:.4f} +/-{sdt:.4f}   acc {acct:.4f}")
    print(f"  -> timing delta: AUC {auct-auc6:+.4f}")

    # matched v5 baseline on the same fields
    Xv5, yv5, ids5 = load_v5(ids6)
    n_match = len(ids5)
    print(f"\nmatched v5 fields found in features_v5: {n_match}/{len(ids6)}")
    if n_match >= 30:
        auc5, sd5, acc5 = _cv(Xv5, yv5)
        print(f"  v5 STAGE (same fields) : AUC {auc5:.4f} +/-{sd5:.4f}   acc {acc5:.4f}")
        print(f"\nDeltas vs v5 stage (matched): raw {auc6-auc5:+.4f} | +timing {auct-auc5:+.4f}")


if __name__ == "__main__":
    main()
