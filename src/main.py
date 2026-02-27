import argparse
import os
import sys
import time

from config import DATA_PATH, OUTPUT_DIR
from data_preprocessing import run_preprocessing, print_summary
from eda import run_eda
from rfm_analysis import run_rfm
from k_means import run_k_means
from export import run_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online Retail — Customer Purchase Pattern Analysis"
    )
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help=f"Path to the combined CSV file (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help=f"Output directory for charts, CSVs, and report (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip Word report generation (faster for iterative runs)",
    )
    return parser.parse_args()


def main(data_path: str, output_dir: str, skip_report: bool = False) -> None:

    t_start = time.time()

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "═" * 60)
    print("  ONLINE RETAIL – CUSTOMER ANALYSIS PIPELINE")
    print("═" * 60)

    print("\n[STEP 1/6] Data Loading & Preprocessing")
    df = run_preprocessing(data_path)
    print_summary(df)

    print("\n[STEP 2/6] Exploratory Data Analysis")
    metrics = run_eda(df, output_dir)

    print("\n[STEP 3/6] RFM Analysis")
    rfm_df = run_rfm(df, output_dir)

    print("\n[STEP 4/6] Customer Segmentation")
    rfm_segmented = run_k_means(rfm_df, output_dir)

    print("\n[STEP 5/6] Exporting Results")
    exported_paths = run_export(rfm_segmented, output_dir)

    elapsed = time.time() - t_start
    print("\n" + "═" * 60)
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print("═" * 60)
    print(f"  All outputs saved to: {output_dir}")
    print()
    for label, path in exported_paths.items():
        print(f"    {label:<22} → {os.path.basename(path)}")
    print()


if __name__ == "__main__":
    args = parse_args()

    main(
        data_path=args.data,
        output_dir=args.output,
        skip_report=args.skip_report,
    )
