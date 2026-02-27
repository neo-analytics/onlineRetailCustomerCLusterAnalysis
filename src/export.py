import os
import pandas as pd


def save_customer_segments(rfm: pd.DataFrame, output_dir: str) -> str:

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "customer_segments.csv")
    rfm.to_csv(path, index=False)
    print(f"[exporter] Saved → customer_segments.csv  ({len(rfm):,} rows)")
    return path


def save_segment_summary(rfm: pd.DataFrame, output_dir: str) -> str:

    summary = (
        rfm.groupby("Segment")
        .agg(
            Count=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        )
        .round(2)
        .sort_values("Total_Revenue", ascending=False)
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "segment_summary.csv")
    summary.to_csv(path)
    print(f"[exporter] Saved → segment_summary.csv")
    return path


def print_segment_summary(rfm: pd.DataFrame) -> None:

    summary = (
        rfm.groupby("Segment")
        .agg(
            Count=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        )
        .round(2)
        .sort_values("Total_Revenue", ascending=False)
    )
    print("\n── Segment Summary ─────────────────────────────────────────")
    print(summary.to_string())
    print()


def run_export(rfm: pd.DataFrame, output_dir: str) -> dict:

    print("\n── Exporting Results ───────────────────────────────────────")
    print_segment_summary(rfm)
    return {
        "customer_segments": save_customer_segments(rfm, output_dir),
        "segment_summary": save_segment_summary(rfm, output_dir),
    }
