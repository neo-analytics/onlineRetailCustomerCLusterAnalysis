import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURE_DPI, TOTAL_AMOUNT_COL, RFM_SCORE_BINS

plt.rcParams.update({"figure.dpi": FIGURE_DPI, "font.size": 10})


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, path: str) -> str:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[rfm] Saved → {os.path.basename(path)}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPUTE RFM TABLE
# ─────────────────────────────────────────────────────────────────────────────


def compute(df: pd.DataFrame) -> pd.DataFrame:

    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=(TOTAL_AMOUNT_COL, "sum"),
        )
        .reset_index()
    )

    # Score 1–5 (5 = best customer)
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"], RFM_SCORE_BINS, labels=[5, 4, 3, 2, 1]
    ).astype(int)

    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), RFM_SCORE_BINS, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"], RFM_SCORE_BINS, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    print(f"\n[rfm] RFM table → {len(rfm):,} customers")
    print(
        rfm[["Recency", "Frequency", "Monetary", "RFM_Score"]]
        .describe()
        .round(2)
        .to_string()
    )

    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 2. DISTRIBUTION PLOTS
# ─────────────────────────────────────────────────────────────────────────────


def plot_distributions(rfm: pd.DataFrame, output_dir: str) -> str:

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    specs = [
        ("Recency", "#E07B54"),
        ("Frequency", "#4C72B0"),
        ("Monetary", "#55A868"),
    ]
    for ax, (col, clr) in zip(axes, specs):
        data = rfm[col].clip(upper=rfm[col].quantile(0.99))
        ax.hist(data, bins=50, color=clr, edgecolor="white")
        ax.set_title(f"{col} Distribution", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("# Customers")

    plt.tight_layout()
    path = os.path.join(output_dir, "07_rfm_distributions.png")
    return _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. R × F HEATMAP
# ─────────────────────────────────────────────────────────────────────────────


def plot_heatmap(rfm: pd.DataFrame, output_dir: str) -> str:

    pivot = rfm.pivot_table(
        values="Monetary", index="R_Score", columns="F_Score", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": "Avg Monetary (£)"},
    )
    ax.set_title("Avg Monetary Value — R-Score vs F-Score", fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "08_rfm_heatmap.png")
    return _save(fig, path)


def run_rfm(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:

    print("\n── RFM Analysis ────────────────────────────────────────────")
    rfm = compute(df)
    os.makedirs(output_dir, exist_ok=True)
    plot_distributions(rfm, output_dir)
    plot_heatmap(rfm, output_dir)
    return rfm
