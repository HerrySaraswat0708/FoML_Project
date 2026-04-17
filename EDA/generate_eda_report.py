from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import load_dataset
from utils.project_paths import OUTPUTS_DIR, ensure_dir


EDA_OUTPUT_DIR = ensure_dir(OUTPUTS_DIR / "eda")

DESCRIPTOR_COLUMNS = [
    "MolWt",
    "MolLogP",
    "MolMR",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumValenceElectrons",
    "NumAromaticRings",
    "RingCount",
    "TPSA",
    "LabuteASA",
    "BertzCT",
]


def save_solubility_distribution(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(frame["Solubility"], bins=40, color="#2f5d50", edgecolor="white", alpha=0.9)
    ax.axvline(frame["Solubility"].median(), color="#b03a2e", linestyle="--", linewidth=2, label="Median")
    ax.axvline(frame["Solubility"].mean(), color="#1f618d", linestyle="-.", linewidth=2, label="Mean")
    ax.set_title("Solubility Distribution")
    ax.set_xlabel("Solubility")
    ax.set_ylabel("Molecule Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "solubility_distribution.png", dpi=220)
    plt.close(fig)


def save_logp_scatter(frame: pd.DataFrame) -> None:
    x = frame["MolLogP"].to_numpy(dtype=float)
    y = frame["Solubility"].to_numpy(dtype=float)
    coeffs = np.polyfit(x, y, deg=1)
    trend = np.poly1d(coeffs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=12, alpha=0.22, color="#2874a6", edgecolors="none")
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    ax.plot(x_line, trend(x_line), color="#c0392b", linewidth=2.5, label="Linear trend")
    ax.set_title("MolLogP vs Solubility")
    ax.set_xlabel("MolLogP")
    ax.set_ylabel("Solubility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "mol_logp_vs_solubility.png", dpi=220)
    plt.close(fig)


def save_molwt_hexbin(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(
        frame["MolWt"].to_numpy(dtype=float),
        frame["Solubility"].to_numpy(dtype=float),
        gridsize=35,
        cmap="YlGnBu",
        mincnt=1,
    )
    ax.set_title("Molecular Weight vs Solubility Density")
    ax.set_xlabel("Molecular Weight")
    ax.set_ylabel("Solubility")
    colorbar = fig.colorbar(hb, ax=ax)
    colorbar.set_label("Count")
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "molwt_vs_solubility_density.png", dpi=220)
    plt.close(fig)


def save_group_counts(frame: pd.DataFrame) -> None:
    group_counts = frame["Group"].astype(str).value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(group_counts.index, group_counts.values, color="#7d6608")
    ax.set_title("Dataset Composition by Group")
    ax.set_xlabel("Group")
    ax.set_ylabel("Molecule Count")
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "group_counts.png", dpi=220)
    plt.close(fig)


def save_correlation_heatmap(frame: pd.DataFrame) -> None:
    corr = frame[DESCRIPTOR_COLUMNS + ["Solubility"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Descriptor Correlation Heatmap")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "descriptor_correlation_heatmap.png", dpi=220)
    plt.close(fig)


def save_top_correlations(frame: pd.DataFrame) -> pd.Series:
    correlations = (
        frame[DESCRIPTOR_COLUMNS + ["Solubility"]]
        .corr(numeric_only=True)["Solubility"]
        .drop(labels=["Solubility"])
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    top_corr = correlations.head(8).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#b03a2e" if value < 0 else "#1e8449" for value in top_corr.values]
    ax.barh(top_corr.index, top_corr.values, color=colors)
    ax.set_title("Top Descriptor Correlations With Solubility")
    ax.set_xlabel("Pearson Correlation")
    ax.set_ylabel("Descriptor")
    ax.axvline(0.0, color="black", linewidth=1)
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "top_descriptor_correlations.png", dpi=220)
    plt.close(fig)
    return correlations


def save_descriptor_panels(frame: pd.DataFrame) -> None:
    descriptors = ["MolLogP", "MolWt", "TPSA", "BertzCT"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for ax, column in zip(axes, descriptors, strict=False):
        x = frame[column].to_numpy(dtype=float)
        y = frame["Solubility"].to_numpy(dtype=float)
        coeffs = np.polyfit(x, y, deg=1)
        trend = np.poly1d(coeffs)
        x_line = np.linspace(float(x.min()), float(x.max()), 150)

        ax.scatter(x, y, s=8, alpha=0.18, color="#2874a6", edgecolors="none")
        ax.plot(x_line, trend(x_line), color="#c0392b", linewidth=2)
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.set_title(f"{column} vs Solubility (r={corr:.2f})", fontsize=10)
        ax.set_xlabel(column)
        ax.set_ylabel("Solubility")

    fig.suptitle("Key Descriptor Relationships With Solubility", fontsize=14)
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "descriptor_vs_solubility_panels.png", dpi=220)
    plt.close(fig)


def save_pca_projection(frame: pd.DataFrame) -> dict[str, float]:
    features = frame[DESCRIPTOR_COLUMNS].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=frame["Solubility"].to_numpy(dtype=float),
        s=10,
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_title("PCA Projection of Descriptor Space")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Solubility")
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "descriptor_pca_projection.png", dpi=220)
    plt.close(fig)
    return {
        "pc1_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_variance_ratio": float(pca.explained_variance_ratio_[1]),
    }


def save_solubility_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    low = frame.nsmallest(6, "Solubility")[["Name", "SMILES", "Solubility", "MolLogP", "MolWt"]].copy()
    high = frame.nlargest(6, "Solubility")[["Name", "SMILES", "Solubility", "MolLogP", "MolWt"]].copy()
    low["extreme_type"] = "lowest"
    high["extreme_type"] = "highest"
    extremes = pd.concat([low, high], ignore_index=True)
    extremes.to_csv(EDA_OUTPUT_DIR / "solubility_extremes.csv", index=False)

    plot_frame = extremes.copy()
    plot_frame["short_name"] = [
        name if len(name) <= 36 else f"{name[:33]}..."
        for name in plot_frame["Name"].astype(str)
    ]
    plot_frame = plot_frame.sort_values("Solubility")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#b03a2e" if kind == "lowest" else "#1e8449" for kind in plot_frame["extreme_type"]]
    ax.barh(plot_frame["short_name"], plot_frame["Solubility"], color=colors)
    ax.set_title("Most Extreme Solubility Examples")
    ax.set_xlabel("Solubility")
    ax.set_ylabel("Molecule")
    ax.axvline(0.0, color="black", linewidth=1)
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "solubility_extremes.png", dpi=220)
    plt.close(fig)
    return extremes


def save_summary(
    frame: pd.DataFrame,
    correlations: pd.Series,
    pca_summary: dict[str, float],
    extremes: pd.DataFrame,
) -> None:
    summary = {
        "num_rows": int(len(frame)),
        "num_columns": int(frame.shape[1]),
        "num_unique_smiles": int(frame["CanonicalSMILES"].nunique()),
        "solubility_mean": float(frame["Solubility"].mean()),
        "solubility_median": float(frame["Solubility"].median()),
        "solubility_std": float(frame["Solubility"].std()),
        "solubility_min": float(frame["Solubility"].min()),
        "solubility_max": float(frame["Solubility"].max()),
        "top_descriptor_correlations": {
            key: float(value) for key, value in correlations.head(5).items()
        },
        "pca": pca_summary,
    }
    (EDA_OUTPUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    stats_table = frame[DESCRIPTOR_COLUMNS + ["Solubility"]].describe().transpose()
    stats_table.to_csv(EDA_OUTPUT_DIR / "descriptor_summary_statistics.csv")

    group_counts = frame["Group"].astype(str).value_counts().rename_axis("Group").reset_index(name="Count")
    group_counts.to_csv(EDA_OUTPUT_DIR / "group_counts.csv", index=False)
    correlation_table = correlations.head(8).to_frame("correlation")
    lowest_names = extremes[extremes["extreme_type"] == "lowest"]["Name"].head(3).tolist()
    highest_names = extremes[extremes["extreme_type"] == "highest"]["Name"].head(3).tolist()

    summary_md = f"""# Dataset EDA Summary

- Rows: {len(frame):,}
- Columns: {frame.shape[1]}
- Unique canonical SMILES: {frame['CanonicalSMILES'].nunique():,}
- Solubility mean: {frame['Solubility'].mean():.3f}
- Solubility median: {frame['Solubility'].median():.3f}
- Solubility standard deviation: {frame['Solubility'].std():.3f}
- Solubility range: {frame['Solubility'].min():.3f} to {frame['Solubility'].max():.3f}

## Report-ready figures

- `solubility_distribution.png`
- `mol_logp_vs_solubility.png`
- `molwt_vs_solubility_density.png`
- `group_counts.png`
- `descriptor_correlation_heatmap.png`
- `top_descriptor_correlations.png`
- `descriptor_vs_solubility_panels.png`
- `descriptor_pca_projection.png`
- `solubility_extremes.png`

## Strongest descriptor correlations with solubility

```text
{correlation_table.to_string()}
```

## PCA overview

- PC1 explains {pca_summary['pc1_variance_ratio'] * 100:.2f}% of descriptor variance
- PC2 explains {pca_summary['pc2_variance_ratio'] * 100:.2f}% of descriptor variance

## Example solubility extremes

- Lowest-solubility examples: {", ".join(lowest_names)}
- Highest-solubility examples: {", ".join(highest_names)}
"""
    (EDA_OUTPUT_DIR / "summary.md").write_text(summary_md, encoding="utf-8")


def main() -> None:
    frame = load_dataset()
    plt.style.use("ggplot")

    save_solubility_distribution(frame)
    save_logp_scatter(frame)
    save_molwt_hexbin(frame)
    save_group_counts(frame)
    save_correlation_heatmap(frame)
    correlations = save_top_correlations(frame)
    save_descriptor_panels(frame)
    pca_summary = save_pca_projection(frame)
    extremes = save_solubility_outliers(frame)
    save_summary(frame, correlations, pca_summary, extremes)

    print(f"Saved EDA outputs to {EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
