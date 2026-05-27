import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

STAGES = ["baseline", "emergence", "vegetative", "flowering", "grain_fill", "maturity"]
INDICES = ["NDVI", "NDWI", "EVI"]
STATS_ALL = ["mean", "median", "std", "p10", "p90"]


def load_features(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM phenology_features", conn)
    if df.empty:
        raise ValueError("Table phenology_features is empty")
    text_cols = {"field_id", "crop_label", "planting_date"}
    for col in df.columns:
        if col not in text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── 1. Phenological profiles (mean per crop per stage) ────────────────────────

def plot_phenological_profiles(df: pd.DataFrame, output_dir: str):
    crops = sorted(df["crop_label"].unique())
    palette = sns.color_palette("tab10", len(crops))
    color_map = dict(zip(crops, palette))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle("Phenological Profiles by Crop (median of field means)", fontsize=16, fontweight="bold")

    for ax, idx in zip(axes, INDICES):
        ax.set_title(idx, fontsize=14, fontweight="semibold")
        ax.set_xlabel("Phenological Stage")
        ax.set_ylabel("Index Value")

        for crop in crops:
            sub = df[df["crop_label"] == crop]
            means = []
            for stage in STAGES:
                col = f"{idx}_mean_{stage}"
                if col in sub.columns:
                    means.append(sub[col].median())
                else:
                    means.append(np.nan)
            ax.plot(STAGES, means, marker="o", linewidth=2.5, markersize=7,
                    label=crop, color=color_map[crop])

        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels([s.replace("_", "\n") for s in STAGES], fontsize=9)
        ax.legend(fontsize=9, title="Crop")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "phenological_profiles.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 2. Within-field variability (p10–p90 envelope) ───────────────────────────

def plot_variability_envelopes(df: pd.DataFrame, output_dir: str):
    crops = sorted(df["crop_label"].unique())
    palette = sns.color_palette("tab10", len(crops))
    color_map = dict(zip(crops, palette))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle("Within-Field Variability (p10–p90 envelope, median across fields)",
                 fontsize=16, fontweight="bold")

    x = np.arange(len(STAGES))
    for ax, idx in zip(axes, INDICES):
        ax.set_title(idx, fontsize=14, fontweight="semibold")
        ax.set_xlabel("Phenological Stage")
        ax.set_ylabel("Index Value")

        for i, crop in enumerate(crops):
            sub = df[df["crop_label"] == crop]
            meds, lo, hi = [], [], []
            for stage in STAGES:
                med_col = f"{idx}_median_{stage}"
                p10_col = f"{idx}_p10_{stage}"
                p90_col = f"{idx}_p90_{stage}"
                meds.append(sub[med_col].median() if med_col in sub.columns else np.nan)
                lo.append(sub[p10_col].median() if p10_col in sub.columns else np.nan)
                hi.append(sub[p90_col].median() if p90_col in sub.columns else np.nan)

            color = color_map[crop]
            offset = (i - len(crops) / 2) * 0.06
            ax.fill_between(x + offset, lo, hi, alpha=0.15, color=color)
            ax.plot(x + offset, meds, marker="o", linewidth=2, markersize=6,
                    label=crop, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in STAGES], fontsize=9)
        ax.legend(fontsize=9, title="Crop")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "variability_envelopes.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 3. Data quality: stages_covered distribution ─────────────────────────────

def plot_data_quality(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Data Quality Overview", fontsize=16, fontweight="bold")

    # 3a. stages_covered histogram per crop
    ax = axes[0]
    crops = sorted(df["crop_label"].unique())
    palette = sns.color_palette("tab10", len(crops))
    for crop, color in zip(crops, palette):
        sub = df[df["crop_label"] == crop]["stages_covered"]
        counts = sub.value_counts().reindex(range(7), fill_value=0)
        ax.bar(counts.index + crops.index(crop) * 0.12 - 0.3,
               counts.values, width=0.12, label=crop, color=color)
    ax.set_xlabel("stages_covered")
    ax.set_ylabel("Number of Fields")
    ax.set_title("Stages Covered per Crop")
    ax.legend(fontsize=8, title="Crop")
    ax.set_xticks(range(7))

    # 3b. Null rate heatmap (stage × index)
    ax = axes[1]
    null_matrix = []
    labels_y = []
    for idx in INDICES:
        row = []
        for stage in STAGES:
            col = f"{idx}_mean_{stage}"
            if col in df.columns:
                row.append(df[col].isna().mean())
            else:
                row.append(1.0)
        null_matrix.append(row)
        labels_y.append(idx)

    im = ax.imshow(null_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([s.replace("_", "\n") for s in STAGES], fontsize=9)
    ax.set_yticks(range(len(INDICES)))
    ax.set_yticklabels(labels_y)
    ax.set_title("Null Rate (stage x index)")
    for i in range(len(INDICES)):
        for j in range(len(STAGES)):
            ax.text(j, i, f"{null_matrix[i][j]:.0%}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Null Rate")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "data_quality.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 4. Feature correlation heatmap ───────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    feature_cols = [
        f"{idx}_{stat}_{stage}"
        for idx in INDICES
        for stage in STAGES
        for stat in STATS_ALL
        if f"{idx}_{stat}_{stage}" in df.columns
    ]
    if len(feature_cols) < 2:
        print("  Skipping correlation heatmap (not enough feature columns)")
        return

    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=True, yticklabels=True, ax=ax,
                cbar_kws={"shrink": 0.6, "label": "Pearson r"})
    ax.set_title("Feature Correlation Matrix (90 features)", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=5)
    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 5. Box plots: per-stage distributions by crop ────────────────────────────

def plot_stage_boxplots(df: pd.DataFrame, output_dir: str):
    for idx in INDICES:
        melted = []
        for stage in STAGES:
            col = f"{idx}_mean_{stage}"
            if col not in df.columns:
                continue
            tmp = df[["crop_label", col]].dropna(subset=[col]).copy()
            tmp = tmp.rename(columns={col: "value"})
            tmp["stage"] = stage
            melted.append(tmp)

        if not melted:
            continue

        melted_df = pd.concat(melted, ignore_index=True)

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=melted_df, x="stage", y="value", hue="crop_label",
                    order=STAGES, palette="tab10", ax=ax, fliersize=2)
        ax.set_title(f"{idx} Mean Distribution by Crop and Stage",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Phenological Stage")
        ax.set_ylabel(f"{idx} Mean")
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels([s.replace("_", "\n") for s in STAGES])
        ax.legend(title="Crop", fontsize=8, title_fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"boxplot_{idx.lower()}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ── 6. Summary stats table ───────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n--- Dataset Summary ---")
    print(f"Total fields: {len(df)}")
    print(f"Crops: {sorted(df['crop_label'].unique())}")
    print(f"\nFields per crop:")
    print(df["crop_label"].value_counts().sort_index().to_string())
    print(f"\nstages_covered distribution:")
    print(df["stages_covered"].value_counts().sort_index().to_string())
    print(f"\nMean stages_covered per crop:")
    print(df.groupby("crop_label")["stages_covered"].mean().round(1).to_string())
    print(f"\nArea (hectares) per crop:")
    print(df.groupby("crop_label")["area_hectares"].describe()[["mean", "min", "max"]].round(1).to_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all_plots(db_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print(f"Loading data from {db_path}...")
    df = load_features(db_path)
    print(f"Loaded {len(df)} fields, {df['crop_label'].nunique()} crops\n")

    print_summary(df)

    print("\nGenerating plots...")
    plot_phenological_profiles(df, output_dir)
    plot_variability_envelopes(df, output_dir)
    plot_data_quality(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_stage_boxplots(df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    db_file = os.path.join("src", "data", "features", "features.db")
    out_dir = os.path.join("src", "analysis", "plots", "features")
    generate_all_plots(db_file, out_dir)
