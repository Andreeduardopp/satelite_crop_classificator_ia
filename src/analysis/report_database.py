import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

STAGES = ["baseline", "emergence", "vegetative", "flowering", "grain_fill", "maturity"]
OPTICAL_INDICES = ["NDVI", "NDWI", "EVI"]
SAR_INDICES = ["VV", "VH", "CR", "RVI"]
ALL_INDICES = OPTICAL_INDICES + SAR_INDICES
STATS = ["mean", "median", "std", "p10", "p90"]


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


# --1. Text summary ─────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("=" * 70)
    print("DATABASE REPORT")
    print("=" * 70)
    print(f"\nTotal fields: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Crops ({df['crop_label'].nunique()}): {sorted(df['crop_label'].unique())}")

    print(f"\n-- Fields per crop --")
    print(df["crop_label"].value_counts().sort_index().to_string())

    print(f"\n--stages_covered distribution --")
    print(df["stages_covered"].value_counts().sort_index().to_string())

    print(f"\n--Mean stages_covered per crop --")
    print(df.groupby("crop_label")["stages_covered"].mean().round(2).to_string())

    print(f"\n--Area (hectares) per crop --")
    print(df.groupby("crop_label")["area_hectares"]
          .describe()[["mean", "min", "max", "std"]].round(2).to_string())

    # Overall null %
    feature_cols = [c for c in df.columns if any(
        c.startswith(f"{idx}_") for idx in ALL_INDICES
    )]
    total_cells = len(df) * len(feature_cols)
    null_cells = df[feature_cols].isna().sum().sum()
    print(f"\n--Overall null rate --")
    print(f"Feature cells: {total_cells:,}")
    print(f"Null cells:    {null_cells:,} ({null_cells / total_cells:.1%})")

    opt_cols = [c for c in feature_cols if any(c.startswith(f"{idx}_") for idx in OPTICAL_INDICES)]
    sar_cols = [c for c in feature_cols if any(c.startswith(f"{idx}_") for idx in SAR_INDICES)]
    opt_null = df[opt_cols].isna().sum().sum()
    sar_null = df[sar_cols].isna().sum().sum()
    print(f"  Optical: {opt_null:,}/{len(df)*len(opt_cols):,} ({opt_null/(len(df)*len(opt_cols)):.1%})")
    print(f"  SAR:     {sar_null:,}/{len(df)*len(sar_cols):,} ({sar_null/(len(df)*len(sar_cols)):.1%})")

    # SAR backfill coverage
    sar_any = df[[f"VV_mean_{s}" for s in STAGES if f"VV_mean_{s}" in df.columns]].notna().any(axis=1).sum()
    print(f"\n--SAR backfill coverage --")
    print(f"Fields with any SAR data: {sar_any}/{len(df)} ({sar_any/len(df):.1%})")
    for crop in sorted(df["crop_label"].unique()):
        sub = df[df["crop_label"] == crop]
        sar_crop = sub[[f"VV_mean_{s}" for s in STAGES if f"VV_mean_{s}" in sub.columns]].notna().any(axis=1).sum()
        print(f"  {crop}: {sar_crop}/{len(sub)} ({sar_crop/len(sub):.1%})")
    print()


# --2. Null rate heatmap (optical + SAR, per crop) ──────────────────────────

def plot_null_heatmap(df: pd.DataFrame, output_dir: str):
    crops = sorted(df["crop_label"].unique())
    n_crops = len(crops)

    fig, axes = plt.subplots(n_crops + 1, 1, figsize=(18, 3.5 * (n_crops + 1)))
    fig.suptitle("Null Rate per Index × Stage", fontsize=18, fontweight="bold", y=0.995)

    def _build_matrix(sub):
        matrix = []
        labels = []
        for idx in ALL_INDICES:
            row = []
            for stage in STAGES:
                col = f"{idx}_mean_{stage}"
                row.append(sub[col].isna().mean() if col in sub.columns else 1.0)
            matrix.append(row)
            labels.append(idx)
        return np.array(matrix), labels

    # All crops combined
    mat, ylabels = _build_matrix(df)
    ax = axes[0]
    sns.heatmap(mat, annot=True, fmt=".0%", cmap="YlOrRd", vmin=0, vmax=1,
                xticklabels=[s.replace("_", "\n") for s in STAGES],
                yticklabels=ylabels, ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title(f"ALL CROPS (n={len(df)})", fontsize=13, fontweight="bold")

    # Per crop
    for i, crop in enumerate(crops):
        sub = df[df["crop_label"] == crop]
        mat, ylabels = _build_matrix(sub)
        ax = axes[i + 1]
        sns.heatmap(mat, annot=True, fmt=".0%", cmap="YlOrRd", vmin=0, vmax=1,
                    xticklabels=[s.replace("_", "\n") for s in STAGES],
                    yticklabels=ylabels, ax=ax, cbar_kws={"shrink": 0.6})
        ax.set_title(f"{crop} (n={len(sub)})", fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(output_dir, "null_rate_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --3. Null rate per column (bar chart, top-N worst) ────────────────────────

def plot_null_per_column(df: pd.DataFrame, output_dir: str):
    feature_cols = [c for c in df.columns if any(
        c.startswith(f"{idx}_") for idx in ALL_INDICES
    )]
    null_pct = df[feature_cols].isna().mean().sort_values(ascending=False)

    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    fig.suptitle("Null Rate per Feature Column", fontsize=18, fontweight="bold")

    # Top 40 worst columns
    top = null_pct.head(40)
    colors = ["#d32f2f" if v > 0.5 else "#f57c00" if v > 0.25 else "#4caf50" for v in top.values]
    ax = axes[0]
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Null Rate")
    ax.set_title("Top 40 Columns with Highest Null Rate", fontsize=14)
    ax.invert_yaxis()
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="50%")
    ax.axvline(0.25, color="orange", linestyle="--", alpha=0.5, label="25%")
    ax.legend()

    # Distribution of null rates
    ax = axes[1]
    ax.hist(null_pct.values, bins=30, color="#1976d2", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Null Rate")
    ax.set_ylabel("Number of Columns")
    ax.set_title(f"Distribution of Null Rates across {len(feature_cols)} Feature Columns", fontsize=14)
    ax.axvline(null_pct.median(), color="red", linestyle="--",
               label=f"Median: {null_pct.median():.1%}")
    ax.legend(fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "null_per_column.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --4. Value distributions per index (violin plots, across stages) ──────────

def plot_value_distributions(df: pd.DataFrame, output_dir: str):
    for idx in ALL_INDICES:
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

        fig, ax = plt.subplots(figsize=(16, 7))
        sns.violinplot(data=melted_df, x="stage", y="value", hue="crop_label",
                       order=STAGES, palette="tab10", ax=ax, inner="quartile",
                       cut=0, density_norm="width")
        ax.set_title(f"{idx} - Mean Distribution by Crop and Stage",
                     fontsize=15, fontweight="bold")
        ax.set_xlabel("Phenological Stage", fontsize=12)
        ax.set_ylabel(f"{idx} Mean", fontsize=12)
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels([s.replace("_", "\n") for s in STAGES])
        ax.legend(title="Crop", fontsize=8, title_fontsize=9, loc="upper left",
                  bbox_to_anchor=(1.01, 1))
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"violin_{idx.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# --5. SAR vs Optical coverage comparison ───────────────────────────────────

def plot_sar_vs_optical_coverage(df: pd.DataFrame, output_dir: str):
    crops = sorted(df["crop_label"].unique())

    opt_coverage = []
    sar_coverage = []
    for stage in STAGES:
        opt_col = f"NDVI_mean_{stage}"
        sar_col = f"VV_mean_{stage}"
        opt_coverage.append(df[opt_col].notna().mean() if opt_col in df.columns else 0)
        sar_coverage.append(df[sar_col].notna().mean() if sar_col in df.columns else 0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Optical vs SAR Data Coverage", fontsize=18, fontweight="bold")

    # Overall coverage per stage
    ax = axes[0]
    x = np.arange(len(STAGES))
    w = 0.35
    ax.bar(x - w / 2, opt_coverage, w, label="Optical (NDVI)", color="#4caf50", alpha=0.85)
    ax.bar(x + w / 2, sar_coverage, w, label="SAR (VV)", color="#1976d2", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in STAGES])
    ax.set_ylabel("Coverage (fraction non-null)")
    ax.set_title("Coverage per Stage (all crops)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    for i, (ov, sv) in enumerate(zip(opt_coverage, sar_coverage)):
        ax.text(i - w / 2, ov + 0.02, f"{ov:.0%}", ha="center", fontsize=8)
        ax.text(i + w / 2, sv + 0.02, f"{sv:.0%}", ha="center", fontsize=8)

    # Per-crop coverage (stacked bar: optical-only, both, SAR-only, neither)
    ax = axes[1]
    categories = {"Both": [], "Optical only": [], "SAR only": [], "Neither": []}
    for crop in crops:
        sub = df[df["crop_label"] == crop]
        opt_any = sub[[f"NDVI_mean_{s}" for s in STAGES if f"NDVI_mean_{s}" in sub.columns]].notna().any(axis=1)
        sar_any = sub[[f"VV_mean_{s}" for s in STAGES if f"VV_mean_{s}" in sub.columns]].notna().any(axis=1)
        n = len(sub)
        categories["Both"].append(((opt_any & sar_any).sum()) / n)
        categories["Optical only"].append(((opt_any & ~sar_any).sum()) / n)
        categories["SAR only"].append(((~opt_any & sar_any).sum()) / n)
        categories["Neither"].append(((~opt_any & ~sar_any).sum()) / n)

    x = np.arange(len(crops))
    bottom = np.zeros(len(crops))
    colors = {"Both": "#2e7d32", "Optical only": "#66bb6a", "SAR only": "#42a5f5", "Neither": "#ef5350"}
    for cat, vals in categories.items():
        ax.bar(x, vals, bottom=bottom, label=cat, color=colors[cat], width=0.6)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(crops, rotation=30, ha="right")
    ax.set_ylabel("Fraction of fields")
    ax.set_title("Data Availability per Crop", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "sar_vs_optical_coverage.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --6. Correlation heatmap (optical + SAR, grouped) ────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    mean_cols = [
        f"{idx}_mean_{stage}"
        for idx in ALL_INDICES
        for stage in STAGES
        if f"{idx}_mean_{stage}" in df.columns
    ]
    if len(mean_cols) < 2:
        print("  Skipping correlation heatmap (not enough columns)")
        return

    corr = df[mean_cols].corr()

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=True, yticklabels=True, ax=ax,
                cbar_kws={"shrink": 0.5, "label": "Pearson r"})
    ax.set_title(f"Correlation Matrix — Mean Features ({len(mean_cols)} cols, optical + SAR)",
                 fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_heatmap_full.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --7. Stages covered distribution (per crop) ──────────────────────────────

def plot_stages_covered(df: pd.DataFrame, output_dir: str):
    crops = sorted(df["crop_label"].unique())
    palette = sns.color_palette("tab10", len(crops))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Stages Covered Analysis", fontsize=18, fontweight="bold")

    # Grouped bar chart
    ax = axes[0]
    max_stages = int(df["stages_covered"].max())
    x = np.arange(max_stages + 1)
    width = 0.8 / len(crops)
    for i, (crop, color) in enumerate(zip(crops, palette)):
        sub = df[df["crop_label"] == crop]["stages_covered"]
        counts = sub.value_counts().reindex(range(max_stages + 1), fill_value=0)
        ax.bar(x + i * width - 0.4 + width / 2, counts.values, width,
               label=crop, color=color)
    ax.set_xlabel("Stages Covered")
    ax.set_ylabel("Number of Fields")
    ax.set_title("Distribution of Stages Covered per Crop")
    ax.set_xticks(x)
    ax.legend(fontsize=8, title="Crop")

    # Box plot of stages_covered per crop
    ax = axes[1]
    sns.boxplot(data=df, x="crop_label", y="stages_covered", hue="crop_label",
                palette="tab10", order=crops, ax=ax, legend=False)
    ax.set_xlabel("Crop")
    ax.set_ylabel("Stages Covered")
    ax.set_title("Stages Covered per Crop (box plot)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "stages_covered.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --8. Per-stat breakdown (mean vs std vs p10 vs p90) for one index ─────────

def plot_stat_breakdown(df: pd.DataFrame, output_dir: str):
    for idx in ["NDVI", "VV"]:
        cols_exist = [f"{idx}_{s}_{st}" for s in STATS for st in STAGES
                      if f"{idx}_{s}_{st}" in df.columns]
        if len(cols_exist) < 5:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"{idx} — All Statistics per Stage (median across fields)",
                     fontsize=18, fontweight="bold")

        crops = sorted(df["crop_label"].unique())
        palette = sns.color_palette("tab10", len(crops))
        color_map = dict(zip(crops, palette))

        for ax, stage in zip(axes.flatten(), STAGES):
            ax.set_title(stage.replace("_", " ").title(), fontsize=13, fontweight="semibold")
            data_for_box = []
            labels_for_box = []
            for stat in STATS:
                col = f"{idx}_{stat}_{stage}"
                if col in df.columns:
                    vals = df[col].dropna()
                    if not vals.empty:
                        data_for_box.append(vals.values)
                        labels_for_box.append(stat)

            if data_for_box:
                bp = ax.boxplot(data_for_box, tick_labels=labels_for_box, patch_artist=True,
                                showfliers=False)
                stat_colors = ["#4caf50", "#2196f3", "#ff9800", "#9c27b0", "#f44336"]
                for patch, color in zip(bp["boxes"], stat_colors[:len(data_for_box)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(output_dir, f"stat_breakdown_{idx.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# --9. Area distribution per crop ───────────────────────────────────────────

def plot_area_distribution(df: pd.DataFrame, output_dir: str):
    crops = sorted(df["crop_label"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Field Area Distribution", fontsize=18, fontweight="bold")

    ax = axes[0]
    for crop in crops:
        sub = df[df["crop_label"] == crop]["area_hectares"].dropna()
        ax.hist(sub, bins=30, alpha=0.5, label=crop)
    ax.set_xlabel("Area (hectares)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Field Areas by Crop")
    ax.legend(fontsize=8)

    ax = axes[1]
    sns.boxplot(data=df, x="crop_label", y="area_hectares", hue="crop_label",
                palette="tab10", order=crops, ax=ax, showfliers=False, legend=False)
    ax.set_xlabel("Crop")
    ax.set_ylabel("Area (hectares)")
    ax.set_title("Field Area per Crop (box plot)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "area_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --10. SAR phenological profiles ──────────────────────────────────────────

def plot_sar_profiles(df: pd.DataFrame, output_dir: str):
    crops = sorted(df["crop_label"].unique())
    palette = sns.color_palette("tab10", len(crops))
    color_map = dict(zip(crops, palette))

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("SAR Phenological Profiles by Crop (median of field means)",
                 fontsize=16, fontweight="bold")

    for ax, idx in zip(axes, SAR_INDICES):
        ax.set_title(idx, fontsize=14, fontweight="semibold")
        ax.set_xlabel("Phenological Stage")
        ax.set_ylabel("Value" if idx in ("CR", "RVI") else "dB")

        for crop in crops:
            sub = df[df["crop_label"] == crop]
            means = []
            for stage in STAGES:
                col = f"{idx}_mean_{stage}"
                means.append(sub[col].median() if col in sub.columns else np.nan)
            if all(np.isnan(m) for m in means):
                continue
            ax.plot(STAGES, means, marker="o", linewidth=2.5, markersize=7,
                    label=crop, color=color_map[crop])

        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels([s.replace("_", "\n") for s in STAGES], fontsize=9)
        ax.legend(fontsize=8, title="Crop")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "sar_profiles.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --Main ────────────────────────────────────────────────────────────────────

def generate_report(db_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print(f"Loading data from {db_path}...")
    df = load_features(db_path)
    print(f"Loaded {len(df)} fields, {df['crop_label'].nunique()} crops\n")

    print_summary(df)

    print("Generating plots...")
    plot_null_heatmap(df, output_dir)
    plot_null_per_column(df, output_dir)
    plot_value_distributions(df, output_dir)
    plot_sar_vs_optical_coverage(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_stages_covered(df, output_dir)
    plot_stat_breakdown(df, output_dir)
    plot_area_distribution(df, output_dir)
    plot_sar_profiles(df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    db_file = os.path.join("src", "data", "features", "features.db")
    out_dir = os.path.join("src", "analysis", "plots", "report")
    generate_report(db_file, out_dir)
