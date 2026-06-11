"""
train_lstm.py — LSTM Crop Classifier
=====================================
Reshapes stage-aggregated features into temporal sequences (6 timesteps)
and trains a bidirectional LSTM with a static feature branch.

Sequence: baseline → emergence → vegetative → flowering → grain_fill → maturity
Channels: NDVI, NDWI, EVI (×5 stats) + VV, VH, CR, RVI (×5 stats) = 35/timestep
Static:   planting_doy_sin, planting_doy_cos, latitude, longitude, area_hectares

Usage:
    python -m src.model.train_lstm \
        --db src/data/features/features.db \
        --n-per-crop 1000 --min-stages 3 \
        --epochs 100 --lr 0.001 --batch-size 64
"""

import os
import sqlite3
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# ── Constants ────────────────────────────────────────────────────────────────

TEXT_COLS = {"field_id", "crop_label", "planting_date"}
STAGES = ["baseline", "emergence", "vegetative", "flowering", "grain_fill", "maturity"]
TEMPORAL_INDICES = ["NDVI", "NDWI", "EVI", "VV", "VH", "CR", "RVI"]
STATS = ["mean", "median", "std", "p10", "p90"]
STATIC_COLS = ["latitude", "longitude", "area_hectares"]

SEQ_LEN = len(STAGES)                             # 6
CHANNELS_PER_STEP = len(TEMPORAL_INDICES) * len(STATS)  # 7 × 5 = 35

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────────────────────────

class CropSequenceDataset(Dataset):
    """Wraps pre-built numpy arrays into a PyTorch Dataset."""

    def __init__(self, sequences: np.ndarray, static: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)   # (N, 6, 35)
        self.static = torch.FloatTensor(static)         # (N, n_static)
        self.labels = torch.LongTensor(labels)           # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.static[idx], self.labels[idx]


# ── Model ────────────────────────────────────────────────────────────────────

class CropLSTM(nn.Module):
    """
    Bidirectional LSTM on stage sequences + static branch → FC classifier.

    Architecture:
        temporal_seq (N, 6, 35)  →  BiLSTM  →  (N, hidden*2)
        static_feat  (N, n_s)   →  FC       →  (N, 32)
        concat                  →  FC layers → (N, n_classes)
    """

    def __init__(
        self,
        n_channels: int = CHANNELS_PER_STEP,
        n_static: int = 5,
        n_classes: int = 7,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.temporal_norm = nn.LayerNorm(hidden_size * 2)

        self.static_branch = nn.Sequential(
            nn.Linear(n_static, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        combined_dim = hidden_size * 2 + 32

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # seq: (batch, 6, 35)
        lstm_out, _ = self.lstm(seq)          # (batch, 6, hidden*2)
        # Use last timestep output
        temporal = lstm_out[:, -1, :]          # (batch, hidden*2)
        temporal = self.temporal_norm(temporal)

        static_out = self.static_branch(static)  # (batch, 32)

        combined = torch.cat([temporal, static_out], dim=1)
        return self.classifier(combined)


# ── Data Loading & Reshaping ─────────────────────────────────────────────────

def load_and_reshape(
    db_path: str,
    min_stages: int = 3,
    planting_year: int | None = None,
    n_per_crop: int | None = None,
    fallback_year: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    """
    Loads DB, builds temporal sequences (N, 6, 35) and static features (N, 5).
    Returns: sequences, static_features, labels, label_encoder, channel_names
    """
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
        logger.info("Filtered stages >= %d: %d -> %d", min_stages, before, len(df))

    # ── Sampling ─────────────────────────────────────────────────────────
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
    elif planting_year is not None:
        df = df[years_col == planting_year].reset_index(drop=True)
    elif n_per_crop is not None:
        parts = [
            g.sample(min(len(g), n_per_crop), random_state=42)
            for _, g in df.groupby("crop_label")
        ]
        df = pd.concat(parts, ignore_index=True)

    logger.info("Samples: %d", len(df))

    # ── Build temporal sequences ─────────────────────────────────────────
    # Channel order: for each index, [mean, median, std, p10, p90]
    channel_names = []
    for idx in TEMPORAL_INDICES:
        for stat in STATS:
            channel_names.append(f"{idx}_{stat}")

    n = len(df)
    sequences = np.zeros((n, SEQ_LEN, CHANNELS_PER_STEP), dtype=np.float32)

    for t, stage in enumerate(STAGES):
        for c, (idx, stat) in enumerate(
            [(i, s) for i in TEMPORAL_INDICES for s in STATS]
        ):
            col_name = f"{idx}_{stat}_{stage}"
            if col_name in df.columns:
                sequences[:, t, c] = df[col_name].values.astype(np.float32)

    # Replace NaN with 0 in sequences (will be normalized later)
    nan_rate = np.isnan(sequences).mean()
    logger.info("Sequence NaN rate: %.1f%%", nan_rate * 100)
    sequences = np.nan_to_num(sequences, nan=0.0)

    # ── Static features ──────────────────────────────────────────────────
    doy = pd.to_datetime(df["planting_date"], errors="coerce").dt.dayofyear
    static = np.column_stack([
        np.sin(2 * np.pi * doy.values / 365),  # planting_doy_sin
        np.cos(2 * np.pi * doy.values / 365),  # planting_doy_cos
        df["latitude"].values,
        df["longitude"].values,
        df["area_hectares"].values if "area_hectares" in df.columns
            else np.zeros(n),
    ]).astype(np.float32)
    static = np.nan_to_num(static, nan=0.0)

    static_names = ["planting_doy_sin", "planting_doy_cos", "latitude", "longitude", "area_hectares"]

    # ── Labels ───────────────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(df["crop_label"])

    logger.info(
        "Built sequences: %s, static: %s, classes: %s",
        sequences.shape, static.shape, list(le.classes_)
    )

    return sequences, static, y, le, channel_names


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: CropLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    for seq, static, labels in loader:
        seq, static, labels = seq.to(DEVICE), static.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(seq, static)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: CropLSTM,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    for seq, static, labels in loader:
        seq, static = seq.to(DEVICE), static.to(DEVICE)
        logits = model(seq, static)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def normalize_sequences(
    train_seq: np.ndarray, val_seq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel z-normalization fitted on train."""
    # Reshape to (N*6, 35) for fitting
    n_tr = train_seq.shape[0]
    flat_tr = train_seq.reshape(-1, train_seq.shape[2])
    mean = flat_tr.mean(axis=0)
    std = flat_tr.std(axis=0)
    std[std < 1e-8] = 1.0

    train_norm = (train_seq - mean) / std
    val_norm = (val_seq - mean) / std
    return train_norm.astype(np.float32), val_norm.astype(np.float32), mean, std


def normalize_static(
    train_st: np.ndarray, val_st: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_st)
    val_norm = scaler.transform(val_st)
    return train_norm.astype(np.float32), val_norm.astype(np.float32), scaler


# ── Cross-validated Training Pipeline ────────────────────────────────────────

def train_and_evaluate(
    sequences: np.ndarray,
    static: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    n_folds: int = 5,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    hidden_size: int = 128,
    n_layers: int = 2,
    dropout: float = 0.3,
    patience: int = 15,
    output_dir: str = "src/model/runs_lstm",
):
    os.makedirs(output_dir, exist_ok=True)

    n_classes = len(le.classes_)
    n_static = static.shape[1]
    class_names = list(le.classes_)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred_all = np.full(len(y), -1, dtype=int)
    fold_scores = []

    logger.info("Training on %s | %d folds | %d epochs | lr=%.4f | bs=%d",
                DEVICE, n_folds, epochs, lr, batch_size)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(sequences, y)):
        logger.info("══ Fold %d/%d ══", fold_idx + 1, n_folds)

        # ── Normalize ────────────────────────────────────────────────────
        seq_tr, seq_va, seq_mean, seq_std = normalize_sequences(
            sequences[train_idx], sequences[val_idx]
        )
        st_tr, st_va, st_scaler = normalize_static(
            static[train_idx], static[val_idx]
        )

        # ── Compute class weights ────────────────────────────────────────
        class_counts = np.bincount(y[train_idx], minlength=n_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * n_classes
        weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)

        # ── DataLoaders ──────────────────────────────────────────────────
        train_ds = CropSequenceDataset(seq_tr, st_tr, y[train_idx])
        val_ds = CropSequenceDataset(seq_va, st_va, y[val_idx])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                                num_workers=0, pin_memory=True)

        # ── Model ────────────────────────────────────────────────────────
        model = CropLSTM(
            n_channels=CHANNELS_PER_STEP,
            n_static=n_static,
            n_classes=n_classes,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-6
        )

        # ── Training loop ────────────────────────────────────────────────
        best_f1 = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                preds, labels = evaluate(model, val_loader)
                val_f1 = f1_score(labels, preds, average="macro")
                val_acc = accuracy_score(labels, preds)
                scheduler.step(val_f1)
                cur_lr = optimizer.param_groups[0]["lr"]

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 5

                if (epoch + 1) % 20 == 0 or epoch == 0:
                    logger.info(
                        "  Epoch %3d | loss=%.4f | val_acc=%.2f%% | val_f1=%.4f | lr=%.1e | best=%.4f",
                        epoch + 1, train_loss, val_acc * 100, val_f1, cur_lr, best_f1
                    )

                if no_improve >= patience:
                    logger.info("  Early stop at epoch %d (patience=%d)", epoch + 1, patience)
                    break

        # ── Evaluate best model ──────────────────────────────────────────
        model.load_state_dict(best_state)
        preds, labels = evaluate(model, val_loader)
        y_pred_all[val_idx] = preds

        fold_f1 = f1_score(labels, preds, average="macro")
        fold_acc = accuracy_score(labels, preds)
        fold_scores.append(fold_f1)
        logger.info("  Fold %d best: acc=%.2f%% f1=%.4f", fold_idx + 1, fold_acc * 100, fold_f1)

    # ── Overall results ──────────────────────────────────────────────────
    logger.info("═══ Overall Results ═══")

    acc = accuracy_score(y, y_pred_all)
    f1m = f1_score(y, y_pred_all, average="macro")
    f1w = f1_score(y, y_pred_all, average="weighted")

    logger.info("Accuracy:    %.2f%%", acc * 100)
    logger.info("F1 macro:    %.4f", f1m)
    logger.info("F1 weighted: %.4f", f1w)
    logger.info("Fold F1s:    %s", [f"{s:.4f}" for s in fold_scores])

    report = classification_report(y, y_pred_all, target_names=class_names, output_dict=True)
    report_str = classification_report(y, y_pred_all, target_names=class_names)
    print(f"\nLSTM Classification Report:\n{report_str}")

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred_all)
    _plot_confusion_matrix(cm, class_names, output_dir)

    # ── Train final model on all data ────────────────────────────────────
    logger.info("Training final model on all data...")

    # Normalize all data
    flat_all = sequences.reshape(-1, sequences.shape[2])
    seq_mean = flat_all.mean(axis=0)
    seq_std = flat_all.std(axis=0)
    seq_std[seq_std < 1e-8] = 1.0
    seq_norm = ((sequences - seq_mean) / seq_std).astype(np.float32)

    st_scaler = StandardScaler()
    st_norm = st_scaler.fit_transform(static).astype(np.float32)

    full_ds = CropSequenceDataset(seq_norm, st_norm, y)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)

    final_model = CropLSTM(
        n_channels=CHANNELS_PER_STEP,
        n_static=static.shape[1],
        n_classes=n_classes,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
    ).to(DEVICE)

    class_counts = np.bincount(y, minlength=n_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * n_classes
    weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(min(epochs, 80)):
        loss = train_one_epoch(final_model, full_loader, criterion, optimizer)
        if (epoch + 1) % 20 == 0:
            logger.info("  Final model epoch %d | loss=%.4f", epoch + 1, loss)

    # Save model + normalization params
    torch.save({
        "model_state": final_model.state_dict(),
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "static_scaler_mean": st_scaler.mean_,
        "static_scaler_scale": st_scaler.scale_,
        "classes": class_names,
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "dropout": dropout,
        "channel_names": [f"{idx}_{s}" for idx in TEMPORAL_INDICES for s in STATS],
    }, os.path.join(output_dir, "lstm_crop_classifier.pt"))
    logger.info("Model saved: %s", os.path.join(output_dir, "lstm_crop_classifier.pt"))

    # ── Save metrics ─────────────────────────────────────────────────────
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "version": "lstm_v1",
        "model": "BiLSTM",
        "device": str(DEVICE),
        "n_samples": int(len(y)),
        "sequence_shape": [SEQ_LEN, CHANNELS_PER_STEP],
        "n_static": int(static.shape[1]),
        "n_classes": n_classes,
        "classes": class_names,
        "hyperparams": {
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
        },
        "cv_folds": n_folds,
        "fold_f1_scores": [round(s, 4) for s in fold_scores],
        "accuracy": round(acc, 4),
        "f1_macro": round(f1m, 4),
        "f1_weighted": round(f1w, 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }
            for cls in class_names
        },
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved")

    # ── Comparison ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LSTM vs XGBoost V3 BASELINE (0.8759)")
    print("=" * 60)
    print(f"  Overall F1 macro:     {f1m:.4f}")
    for cls in class_names:
        f1_val = report[cls]["f1-score"]
        print(f"  {cls:<10} F1:       {f1_val:.4f}")
    print("=" * 60)


# ── Plots ────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, class_names: list, output_dir: str):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("LSTM Confusion Matrix (counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("LSTM Confusion Matrix (% per class)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM crop classifier")
    parser.add_argument("--db", default=os.path.join("src", "data", "features", "features.db"))
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--fallback-year", type=int, default=None)
    parser.add_argument("--n-per-crop", type=int, default=None)
    parser.add_argument("--min-stages", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("src", "model", "runs_lstm", run_ts)

    sequences, static, y, le, channel_names = load_and_reshape(
        args.db,
        min_stages=args.min_stages,
        planting_year=args.year,
        n_per_crop=args.n_per_crop,
        fallback_year=args.fallback_year,
    )

    train_and_evaluate(
        sequences, static, y, le,
        n_folds=args.folds,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        patience=args.patience,
        output_dir=output_dir,
    )
    logger.info("LSTM run output: %s", output_dir)
