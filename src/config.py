import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DATA_PATH = os.path.join(DATA_DIR, "OnlineRetail.csv")

# ─── Data / Cleaning ──────────────────────────────────────────────────────────
ENCODING = "unicode_escape"

# Columns that must be present after loading
REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

# ─── Feature Engineering ──────────────────────────────────────────────────────
TOTAL_AMOUNT_COL = "TotalAmount"  # Quantity × UnitPrice

# ─── EDA ──────────────────────────────────────────────────────────────────────
PALETTE = "Set2"
FIGURE_DPI = 130
TOP_N = 10

# ─── RFM ──────────────────────────────────────────────────────────────────────
RFM_SCORE_BINS = 5  # quantile bins for R / F / M scoring

# ─── Clustering ───────────────────────────────────────────────────────────────
RANDOM_SEED = 42
K_RANGE = range(2, 9)
MIN_CLUSTERS = 4
N_INIT = 10  # KMeans restarts

SEGMENT_LABELS = {
    1: "Champions",
    2: "Loyal Customers",
    3: "At-Risk Customers",
    4: "Promising / Potential Loyalists",
    5: "Lost / Hibernating",
    6: "Need Attention",
    7: "Cannot Lose Them",
    8: "Recent Customers",
}
