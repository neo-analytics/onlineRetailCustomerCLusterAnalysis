import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import PALETTE, FIGURE_DPI, TOP_N, TOTAL_AMOUNT_COL

plt.rcParams.update({"figure.dpi": FIGURE_DPI, "font.size": 10})


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: str) -> str:

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved → {os.path.basename(path)}")
    return path


def _gbp_fmt(x, _):

    return f"£{x/1e3:.0f}K"


# ─────────────────────────────────────────────────────────────────────────────
# 1. MONTHLY REVENUE TREND
# ─────────────────────────────────────────────────────────────────────────────
def plot_monthly_revenue(df: pd.DataFrame, output_dir: str) -> str:

    monthly = df.groupby("Month")[TOTAL_AMOUNT_COL].sum().reset_index()
    monthly["Month_str"] = monthly["Month"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        monthly["Month_str"],
        monthly[TOTAL_AMOUNT_COL],
        marker="o",
        color="steelblue",
        linewidth=2,
    )
    ax.fill_between(
        range(len(monthly)), monthly[TOTAL_AMOUNT_COL], alpha=0.15, color="steelblue"
    )
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["Month_str"], rotation=45, ha="right")
    ax.set_title("Monthly Revenue Trend", fontsize=13, fontweight="bold")
    ax.set_ylabel("Revenue (£)")
    ax.set_xlabel("Month")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_gbp_fmt))
    plt.tight_layout()

    path = os.path.join(output_dir, "01_monthly_revenue_trend.png")
    return _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEMPORAL PATTERNS — DAY OF WEEK + HOUR
# ─────────────────────────────────────────────────────────────────────────────
def plot_temporal_patterns(df: pd.DataFrame, output_dir: str) -> str:

    dow_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    dow_counts = df.groupby("DayOfWeek")["InvoiceNo"].nunique().reindex(dow_order)
    hourly = df.groupby("Hour")["InvoiceNo"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].bar(
        dow_counts.index, dow_counts.values, color=sns.color_palette(PALETTE, 7)
    )
    axes[0].set_title("Orders by Day of Week", fontweight="bold")
    axes[0].set_ylabel("Unique Orders")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(hourly.index, hourly.values, color="coral", edgecolor="white")
    axes[1].set_title("Orders by Hour of Day", fontweight="bold")
    axes[1].set_xlabel("Hour (24 h)")
    axes[1].set_ylabel("Unique Orders")

    plt.tight_layout()
    path = os.path.join(output_dir, "02_temporal_patterns.png")
    return _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOP COUNTRIES BY REVENUE (excluding UK)
# ─────────────────────────────────────────────────────────────────────────────
def plot_top_countries(df: pd.DataFrame, output_dir: str) -> str:

    top = (
        df[df["Country"] != "United Kingdom"]
        .groupby("Country")[TOTAL_AMOUNT_COL]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_N)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        top.index[::-1], top.values[::-1], color=sns.color_palette("Blues_r", TOP_N)
    )
    ax.set_title(f"Top {TOP_N} Countries by Revenue (excl. UK)", fontweight="bold")
    ax.set_xlabel("Revenue (£)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_gbp_fmt))
    plt.tight_layout()

    path = os.path.join(output_dir, "03_top_countries.png")
    return _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOP PRODUCTS BY REVENUE
# ─────────────────────────────────────────────────────────────────────────────
def plot_top_products(df: pd.DataFrame, output_dir: str) -> str:

    top = (
        df.groupby("Description")[TOTAL_AMOUNT_COL]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_N)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top.index[::-1], top.values[::-1], color=sns.color_palette("Set1", TOP_N))
    ax.set_title(f"Top {TOP_N} Products by Revenue", fontweight="bold")
    ax.set_xlabel("Revenue (£)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_gbp_fmt))
    plt.tight_layout()

    path = os.path.join(output_dir, "04_top_products.png")
    return _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ORDER VALUE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def plot_order_value_dist(df: pd.DataFrame, output_dir: str) -> str:

    order_values = df.groupby("InvoiceNo")[TOTAL_AMOUNT_COL].sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log1p(order_values), bins=60, color="mediumseagreen", edgecolor="white")
    ax.set_title("Distribution of Order Values (log scale)", fontweight="bold")
    ax.set_xlabel("log(1 + Order Value £)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    path = os.path.join(output_dir, "05_order_value_distribution.png")
    return _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. REPEAT vs ONE-TIME CUSTOMERS
# ─────────────────────────────────────────────────────────────────────────────
def plot_repeat_vs_onetime(df: pd.DataFrame, output_dir: str) -> str:

    freq = df.groupby("CustomerID")["InvoiceNo"].nunique()
    repeat = int((freq > 1).sum())
    one_time = int((freq == 1).sum())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [repeat, one_time],
        labels=[f"Repeat\n({repeat:,})", f"One-time\n({one_time:,})"],
        colors=["#4C72B0", "#DD8452"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white"),
    )
    ax.set_title("Repeat vs One-time Customers", fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "06_repeat_vs_onetime.png")
    return _save(fig, path)


def run_eda(df: pd.DataFrame, output_dir: str) -> dict:

    os.makedirs(output_dir, exist_ok=True)
    print("\n── EDA ─────────────────────────────────────────────────────")

    order_values = df.groupby("InvoiceNo")[TOTAL_AMOUNT_COL].sum()
    metrics = {
        "total_revenue": round(df[TOTAL_AMOUNT_COL].sum(), 2),
        "n_orders": df["InvoiceNo"].nunique(),
        "n_customers": df["CustomerID"].nunique(),
        "n_products": df["StockCode"].nunique(),
        "avg_order_value": round(order_values.mean(), 2),
        "top_country": df["Country"].value_counts().idxmax(),
    }
    print(f"  Total Revenue   : £{metrics['total_revenue']:,.2f}")
    print(f"  Total Orders    : {metrics['n_orders']:,}")
    print(f"  Customers       : {metrics['n_customers']:,}")
    print(f"  Products        : {metrics['n_products']:,}")
    print(f"  Avg Order Value : £{metrics['avg_order_value']:,.2f}")
    print(f"  Top Country     : {metrics['top_country']}")

    plot_monthly_revenue(df, output_dir)
    plot_temporal_patterns(df, output_dir)
    plot_top_countries(df, output_dir)
    plot_top_products(df, output_dir)
    plot_order_value_dist(df, output_dir)
    plot_repeat_vs_onetime(df, output_dir)

    return metrics
