import pandas as pd
from config import ENCODING, REQUIRED_COLUMNS, TOTAL_AMOUNT_COL


def load_raw(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, encoding=ENCODING)
    df.columns = df.columns.str.strip()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[preprocessing] Missing columns: {missing}")

    print(f"[preprocessing] Loaded  → {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    n_raw = len(df)

    df.dropna(subset=["CustomerID"], inplace=True)
    print(f"[preprocessing] Drop missing CustomerID → {n_raw - len(df):,} rows removed")

    n_before = len(df)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    print(
        f"[preprocessing] Drop cancellations     → {n_before - len(df):,} rows removed"
    )

    n_before = len(df)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    print(
        f"[preprocessing] Drop non-positive vals → {n_before - len(df):,} rows removed"
    )

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    df[TOTAL_AMOUNT_COL] = df["Quantity"] * df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M")
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    print(f"[preprocessing] Clean shape → {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df.reset_index(drop=True)


def run_preprocessing(path: str) -> pd.DataFrame:
    """Load and clean in one call."""
    return clean(load_raw(path))


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise overview of the cleaned dataset to stdout."""
    print("\n" + "═" * 60)
    print("  DATASET SUMMARY")
    print("═" * 60)
    print(
        f"  Date range     : {df['InvoiceDate'].min().date()}  →  {df['InvoiceDate'].max().date()}"
    )
    print(f"  Rows           : {len(df):,}")
    print(f"  Unique invoices: {df['InvoiceNo'].nunique():,}")
    print(f"  Unique customers: {df['CustomerID'].nunique():,}")
    print(f"  Unique products : {df['StockCode'].nunique():,}")
    print(f"  Countries      : {df['Country'].nunique()}")
    print(f"  Total revenue  : £{df[TOTAL_AMOUNT_COL].sum():,.2f}")
    print("═" * 60 + "\n")
