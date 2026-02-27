import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from config import (
    RANDOM_SEED,
    K_RANGE,
    MIN_CLUSTERS,
    N_INIT,
    SEGMENT_LABELS,
    PALETTE,
    FIGURE_DPI,
)


plt.rcParams.update({"figure.dpi": FIGURE_DPI, "font.size": 10})


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, path: str) -> str:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[segmentation] Saved → {os.path.basename(path)}")
    return path


def _scale(rfm: pd.DataFrame):

    features_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_log)
    return X_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. FIT KMEANS
# ─────────────────────────────────────────────────────────────────────────────


def fit(rfm: pd.DataFrame):

    rfm = rfm.copy()
    X_scaled, _ = _scale(rfm)

    inertias, sil_scores = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=N_INIT)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    # Select best K with a minimum floor for business utility
    raw_best = list(K_RANGE)[int(np.argmax(sil_scores))]
    best_k = max(MIN_CLUSTERS, raw_best)
    best_sil = (
        sil_scores[list(K_RANGE).index(best_k)]
        if best_k in K_RANGE
        else (
            silhouette_score(
                X_scaled,
                KMeans(
                    n_clusters=best_k, random_state=RANDOM_SEED, n_init=N_INIT
                ).fit_predict(X_scaled),
            )
        )
    )
    print(f"\n[segmentation] Best K = {best_k}  |  silhouette = {best_sil:.3f}")

    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=N_INIT)
    rfm["Cluster"] = km_final.fit_predict(X_scaled)

    return rfm, km_final, X_scaled, inertias, sil_scores


# ─────────────────────────────────────────────────────────────────────────────
# 2. LABEL CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────


def label_clusters(rfm: pd.DataFrame) -> pd.DataFrame:

    rfm = rfm.copy()
    means = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    means["score"] = (
        -means["Recency"] / means["Recency"].max()
        + means["Frequency"] / means["Frequency"].max()
        + means["Monetary"] / means["Monetary"].max()
    )
    ranked = means["score"].rank(ascending=False).astype(int)
    labels_map = {
        cluster: SEGMENT_LABELS.get(rank, f"Segment {rank}")
        for cluster, rank in ranked.items()
    }
    rfm["Segment"] = rfm["Cluster"].map(labels_map)

    print("\n[segmentation] Cluster → Segment mapping:")
    seg_counts = rfm.groupby("Segment")["CustomerID"].count()
    for seg, n in seg_counts.items():
        print(f"  {seg:<38} {n:,} customers")

    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────


def plot_cluster_selection(inertias, sil_scores, best_k, output_dir: str) -> str:

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(list(K_RANGE), inertias, marker="o", color="steelblue")
    axes[0].set_title("Elbow Method", fontweight="bold")
    axes[0].set_xlabel("Number of Clusters K")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(K_RANGE), sil_scores, marker="s", color="tomato")
    axes[1].axvline(
        x=best_k, color="green", linestyle="--", alpha=0.7, label=f"K={best_k} (chosen)"
    )
    axes[1].set_title("Silhouette Score", fontweight="bold")
    axes[1].set_xlabel("Number of Clusters K")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "09_cluster_selection.png")
    return _save(fig, path)


def plot_segment_overview(rfm: pd.DataFrame, output_dir: str) -> str:

    seg_counts = rfm["Segment"].value_counts()
    seg_revenue = rfm.groupby("Segment")["Monetary"].sum()
    colors = sns.color_palette(PALETTE, len(seg_counts))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(seg_counts.index, seg_counts.values, color=colors)
    axes[0].set_title("Customer Count per Segment", fontweight="bold")
    axes[0].set_xlabel("# Customers")

    axes[1].barh(seg_revenue.index, seg_revenue.values, color=colors)
    axes[1].set_title("Revenue per Segment", fontweight="bold")
    axes[1].set_xlabel("Revenue (£)")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    plt.tight_layout()

    path = os.path.join(output_dir, "10_segment_overview.png")
    return _save(fig, path)


def plot_rfm_by_segment(rfm: pd.DataFrame, output_dir: str) -> str:

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["Recency", "Frequency", "Monetary"]
    for ax, metric in zip(axes, metrics):
        data_plot = rfm.copy()
        if metric == "Monetary":
            cap = data_plot[metric].quantile(0.95)
            data_plot[metric] = data_plot[metric].clip(upper=cap)
        sns.boxplot(
            data=data_plot,
            x="Segment",
            y=metric,
            hue="Segment",
            palette=PALETTE,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"{metric} by Segment", fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = os.path.join(output_dir, "11_rfm_by_segment.png")
    return _save(fig, path)


def plot_pca(rfm: pd.DataFrame, X_scaled: np.ndarray, output_dir: str) -> str:

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X_scaled)
    segs = rfm["Segment"].unique()
    colors = sns.color_palette(PALETTE, len(segs))

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, seg in enumerate(segs):
        mask = rfm["Segment"] == seg
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=seg,
            alpha=0.5,
            s=15,
            color=colors[i],
        )

    ax.set_title("Customer Segments — PCA 2D Projection", fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    path = os.path.join(output_dir, "12_pca_segments.png")
    return _save(fig, path)


def run_k_means(rfm: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Full segmentation pipeline: fit → label → visualise.

    Parameters
    ----------
    rfm : pd.DataFrame
        Output of rfm.run_all() (or rfm.compute()).
    output_dir : str
        Directory where all PNGs will be saved.

    Returns
    -------
    pd.DataFrame
        RFM table enriched with 'Cluster' and 'Segment' columns.
    """
    print("\n── Segmentation ────────────────────────────────────────────")
    os.makedirs(output_dir, exist_ok=True)

    rfm_clustered, km, X_scaled, inertias, sil_scores = fit(rfm)
    rfm_labelled = label_clusters(rfm_clustered)

    best_k = rfm_labelled["Cluster"].nunique()
    plot_cluster_selection(inertias, sil_scores, best_k, output_dir)
    plot_segment_overview(rfm_labelled, output_dir)
    plot_rfm_by_segment(rfm_labelled, output_dir)
    plot_pca(rfm_labelled, X_scaled, output_dir)

    return rfm_labelled
