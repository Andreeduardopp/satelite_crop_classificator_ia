"""
train_cnn1d.py — 1D-CNN Crop Classifier
=========================================
Multi-scale 1D convolutions over phenological stage sequences.
Better suited than LSTM for short sequences (6 timesteps) because
CNN kernels capture local patterns without needing long-range memory.

Architecture:
    Multi-scale conv blocks (kernel 2 + kernel 3) → concat → global pool
    + static branch → FC classifier → 7 classes

Usage:
    python -m src.model.train_cnn1d \
        --db src/data/features/features.db \
        --n-per-crop 1000 --min-stages 3
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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

# Reuse data loading and dataset from LSTM script
from src.model.train_lstm import (
    load_and_reshape,
    CropSequenceDataset,
    normalize_sequences,
    normalize_static,
    CHANNELS_PER_STEP,
    SEQ_LEN,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1D → BatchNorm → ReLU → Dropout"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class CropCNN1D(nn.Module):
    """
    Multi-scale 1D-CNN for short phenological sequences.

    Input:  (batch, 6, 35)  →  transpose to (batch, 35, 6)
    Branch A: kernel=2 captures adjacent-stage transitions
    Branch B: kernel=3 captures 3-stage patterns (e.g., veg→flower→grain)
    Both branches → global avg pool + global max pool → concat with static → FC
    """

    def __init__(
        self,
        n_channels: int = CHANNELS_PER_STEP,
        n_static: int = 5,
        n_classes: int = 7,
        base_filters: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Branch A: kernel=2 (adjacent-stage transitions)
        self.branch_a = nn.Sequential(
            ConvBlock(n_channels, base_filters, kernel=2, dropout=dropout),
            ConvBlock(base_filters, base_filters * 2, kernel=2, dropout=dropout),
        )

        # Branch B: kernel=3 (three-stage patterns)
        self.branch_b = nn.Sequential(
            ConvBlock(n_channels, base_filters, kernel=3, dropout=dropout),
            ConvBlock(base_filters, base_filters * 2, kernel=3, dropout=dropout),
        )

        # After global pooling: avg + max per branch = 4 × (base_filters*2)
        pool_dim = base_filters * 2 * 4

        # Static branch
        self.static_branch = nn.Sequential(
            nn.Linear(n_static, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        combined_dim = pool_dim + 64

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # seq: (batch, 6, 35) → transpose to (batch, 35, 6) for Conv1d
        x = seq.transpose(1, 2)

        # Multi-scale branches
        a = self.branch_a(x)  # (batch, filters*2, T')
        b = self.branch_b(x)  # (batch, filters*2, T')

        # Global pooling (avg + max for each branch)
        a_avg = a.mean(dim=2)
        a_max = a.max(dim=2).values
        b_avg = b.mean(dim=2)
        b_max = b.max(dim=2).values

        temporal = torch.cat([a_avg, a_max, b_avg, b_max], dim=1)

        static_out = self.static_branch(static)

        combined = torch.cat([temporal, static_out], dim=1)
        return self.classifier(combined)


# ── Training helpers ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
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
def predict(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for seq, static, labels in loader:
        seq, static = seq.to(DEVICE), static.to(DEVICE)
        preds = model(seq, static).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# ── Main pipeline ────────────────────────────────────────────────────────────

def train_and_evaluate(
    sequences: np.ndarray,
    static: np.ndarray,
    y: np.ndarray,
    le,
    n_folds: int = 5,
    epochs: int = 120,
    lr: float = 0.001,
    batch_size: int = 64,
    base_filters: int = 128,
    dropout: float = 0.3,
    patience: int = 20,
    output_dir: str = "src/model/runs_cnn1d",
):
    os.makedirs(output_dir, exist_ok=True)

    n_classes = len(le.classes_)
    n_static = static.shape[1]
    class_names = list(le.classes_)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred_all = np.full(len(y), -1, dtype=int)
    fold_scores = []

    logger.info("1D-CNN on %s | %d folds | %d epochs | lr=%.4f | filters=%d",
                DEVICE, n_folds, epochs, lr, base_filters)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(sequences, y)):
        logger.info("══ Fold %d/%d ══", fold_idx + 1, n_folds)

        seq_tr, seq_va, _, _ = normalize_sequences(sequences[train_idx], sequences[val_idx])
        st_tr, st_va, _ = normalize_static(static[train_idx], static[val_idx])

        class_counts = np.bincount(y[train_idx], minlength=n_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * n_classes
        weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)

        train_ds = CropSequenceDataset(seq_tr, st_tr, y[train_idx])
        val_ds = CropSequenceDataset(seq_va, st_va, y[val_idx])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                                num_workers=0, pin_memory=True)

        model = CropCNN1D(
            n_channels=CHANNELS_PER_STEP,
            n_static=n_static,
            n_classes=n_classes,
            base_filters=base_filters,
            dropout=dropout,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6,
        )

        best_f1 = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            scheduler.step(epoch)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                preds, labels = predict(model, val_loader)
                val_f1 = f1_score(labels, preds, average="macro")
                val_acc = accuracy_score(labels, preds)

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 5

                if (epoch + 1) % 20 == 0 or epoch == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "  Epoch %3d | loss=%.4f | val_acc=%.2f%% | val_f1=%.4f | lr=%.1e | best=%.4f",
                        epoch + 1, train_loss, val_acc * 100, val_f1, cur_lr, best_f1
                    )

                if no_improve >= patience:
                    logger.info("  Early stop at epoch %d", epoch + 1)
                    break

        model.load_state_dict(best_state)
        preds, labels = predict(model, val_loader)
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
    print(f"\n1D-CNN Classification Report:\n{report_str}")

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred_all)
    _plot_confusion_matrix(cm, class_names, output_dir)

    # ── Train final model on all data ────────────────────────────────────
    logger.info("Training final model on all data...")

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

    final_model = CropCNN1D(
        n_channels=CHANNELS_PER_STEP, n_static=static.shape[1],
        n_classes=n_classes, base_filters=base_filters, dropout=dropout,
    ).to(DEVICE)

    class_counts = np.bincount(y, minlength=n_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * n_classes
    weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(min(epochs, 100)):
        loss = train_one_epoch(final_model, full_loader, criterion, optimizer)
        if (epoch + 1) % 20 == 0:
            logger.info("  Final epoch %d | loss=%.4f", epoch + 1, loss)

    torch.save({
        "model_state": final_model.state_dict(),
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "static_scaler_mean": st_scaler.mean_,
        "static_scaler_scale": st_scaler.scale_,
        "classes": class_names,
        "base_filters": base_filters,
        "dropout": dropout,
    }, os.path.join(output_dir, "cnn1d_crop_classifier.pt"))
    logger.info("Model saved")

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "version": "cnn1d_v1",
        "model": "MultiScale_1D_CNN",
        "device": str(DEVICE),
        "n_samples": int(len(y)),
        "sequence_shape": [SEQ_LEN, CHANNELS_PER_STEP],
        "n_static": int(static.shape[1]),
        "n_classes": n_classes,
        "classes": class_names,
        "hyperparams": {
            "base_filters": base_filters,
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

    print("\n" + "=" * 60)
    print("  1D-CNN vs XGBoost V3 (0.8759) vs LSTM (0.8296)")
    print("=" * 60)
    print(f"  Overall F1 macro:     {f1m:.4f}")
    for cls in class_names:
        print(f"  {cls:<10} F1:       {report[cls]['f1-score']:.4f}")
    print("=" * 60)


def _plot_confusion_matrix(cm, class_names, output_dir):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("1D-CNN Confusion Matrix (counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("1D-CNN Confusion Matrix (% per class)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train 1D-CNN crop classifier")
    parser.add_argument("--db", default=os.path.join("src", "data", "features", "features.db"))
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--fallback-year", type=int, default=None)
    parser.add_argument("--n-per-crop", type=int, default=None)
    parser.add_argument("--min-stages", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-filters", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("src", "model", "runs_cnn1d", run_ts)

    sequences, static, y, le, _ = load_and_reshape(
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
        base_filters=args.base_filters,
        dropout=args.dropout,
        patience=args.patience,
        output_dir=output_dir,
    )
    logger.info("1D-CNN run output: %s", output_dir)
