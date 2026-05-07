# Online Retail — Customer Purchase Pattern Analysis

---

## Project Structure

```
retail_analysis/
│
├── requirements.txt               ← Python dependencies
├── README.md
│
├── data/
│   └── online_retail.csv          ← Combined dataset (you must provide this)
│
├── notebooks/                     ← All analysis modules (imported by main.ipynb)
│   ├── main.ipynb                 ← Single entry point — runs the full pipeline
│   ├── config.ipynb               ← All constants, paths & hyperparameters
│   ├── data_processing.ipynb      ← Data loading, cleaning, feature engineering
│   ├── eda.ipynb                  ← Exploratory Data Analysis charts
│   ├── rfm_analysis.ipynb         ← RFM scoring & visualisations
│   ├── k_means.ipynb              ← K-Means clustering, labelling & charts
│   └── export.ipynb               ← CSV export & console summaries
│
├── src/                           ← Python modules 
│   ├── main.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── rfm_analysis.py
│   ├── k_means.py
│   └── export.py
│
└── outputs/                       ← Auto-created; all generated files land here
    ├── 01_monthly_revenue_trend.png
    ├── 02_temporal_patterns.png
    ├── 03_top_countries.png
    ├── 04_top_products.png
    ├── 05_order_value_distribution.png
    ├── 06_repeat_vs_onetime.png
    ├── 07_rfm_distributions.png
    ├── 08_rfm_heatmap.png
    ├── 09_cluster_selection.png
    ├── 10_segment_overview.png
    ├── 11_rfm_by_segment.png
    ├── 12_pca_segments.png
    ├── customer_segments.csv
    └── segment_summary.csv

```

---

## Dataset

| Column        | Type        | Description                                   |
| ------------- | ----------- | --------------------------------------------- |
| `InvoiceNo`   | Categorical | Invoice identifier; prefix `C` = cancellation |
| `StockCode`   | Categorical | Product/item identifier                       |
| `Description` | Text        | Product name                                  |
| `Quantity`    | Numeric     | Units purchased per line                      |
| `InvoiceDate` | DateTime    | Date and time of transaction                  |
| `UnitPrice`   | Numeric     | Price per unit in GBP (£)                     |
| `CustomerID`  | Categorical | Unique customer identifier                    |
| `Country`     | Categorical | Country of purchase                           |

> **Note:** `online_retail.csv` is the pre-combined file. The merging of the two source parts is intentionally excluded from this codebase per assignment requirements.

---

## Setup

### Python dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline (Jupyter Notebook)

Open and execute `notebooks/main.ipynb`:

```bash
jupyter notebook notebooks/main.ipynb
```

Alternatively, run from command line:

```bash
python src/main.py
```

---

## Module Reference

## Module Reference (Jupyter Notebooks)

| Notebook                | Responsibility                                                                  |
| ----------------------- | ------------------------------------------------------------------------------- |
| `config.ipynb`          | All constants: paths, colours, hyperparameters (edit here to tune)              |
| `data_processing.ipynb` | `load_raw()`, `clean()`, `run_preprocessing()`, `print_summary()`               |
| `eda.ipynb`             | `run_eda()` + individual `plot_*()` functions for each chart                    |
| `rfm_analysis.ipynb`    | `compute()` builds the RFM table; `plot_distributions()`, `plot_heatmap()`      |
| `k_means.ipynb`         | `fit()` → KMeans; `label_clusters()` → segment names; 4 plot functions          |
| `export.ipynb`          | `save_customer_segments()`, `save_segment_summary()`, `print_segment_summary()` |
| `main.ipynb`            | Orchestrates all notebooks via `%run` magic and executes the full pipeline      |

Each notebook can be run independently after `config.ipynb`, or run all via `main.ipynb`.

### Notebook Import Structure

Notebooks use Jupyter's `%run` magic to import functions and constants from each other:

- **config.ipynb** → Base configuration (no dependencies)
- **data_processing.ipynb** → `%run config.ipynb`
- **eda.ipynb** → `%run config.ipynb`
- **rfm_analysis.ipynb** → `%run config.ipynb`
- **k_means.ipynb** → `%run config.ipynb`
- **export.ipynb** → Standalone (no notebook dependencies)
- **main.ipynb** → `%run config.ipynb`, `%run data_processing.ipynb`, `%run eda.ipynb`, `%run rfm_analysis.ipynb`, `%run k_means.ipynb`, `%run export.ipynb`

---

## Pipeline Steps

```
Step 0  config.ipynb                Run to load all constants and paths
Step 1  data_processing.ipynb       Load CSV → clean → engineer features
Step 2  eda.ipynb                   6 insight charts + key metrics dict
Step 3  rfm_analysis.ipynb          Compute R/F/M scores + 2 charts
Step 4  k_means.ipynb               K-Means → label clusters + 4 charts
Step 5  export.ipynb                Save customer_segments.csv + segment_summary.csv

Or simply: Run main.ipynb to execute all steps automatically
```

---

## Analysis Overview

### EDA Insights

| Chart                    | Key Finding                                        |
| ------------------------ | -------------------------------------------------- |
| Monthly Revenue          | Peaks in November 2011 (pre-holiday spike)         |
| Day/Hour                 | Thursdays + 10 AM–2 PM = peak purchase window      |
| Top Countries            | Netherlands, Ireland, Germany lead internationally |
| Top Products             | Regency Cakestand, Jumbo Bag dominate revenue      |
| Order Value Distribution | Right-skewed; bulk buyers inflate the mean         |
| Repeat vs One-time       | ~72% repeat buyers — strong retention signal       |

### Customer Segments (K=4)

| Segment         | % Customers | % Revenue | Strategy                                |
| --------------- | ----------- | --------- | --------------------------------------- |
| Champions       | 16%         | ~65%      | VIP rewards, early access, advocacy     |
| Loyal Customers | 27%         | ~24%      | Upsell, bundle promotions               |
| At-Risk         | 19%         | ~5%       | Win-back campaigns, personalised offers |
| Promising       | 37%         | ~6%       | Re-engagement, "We miss you" campaigns  |

---

## Configuration

All tunable parameters are centralised in `src/config.py`:

```python
# Clustering
MIN_CLUSTERS = 4        # minimum K for business utility
K_RANGE      = range(2, 9)
RANDOM_SEED  = 42

# RFM
RFM_SCORE_BINS = 5      # quantile bins (1–5 scoring)

# EDA
TOP_N = 10              # top N countries / products in charts
```

---

## Output Files

| File                    | Description                                               |
| ----------------------- | --------------------------------------------------------- |
| `customer_segments.csv` | Full RFM scores + segment label per customer (4,338 rows) |
| `segment_summary.csv`   | Aggregated stats per segment                              |
| `*.png` (12 files)      | All EDA, RFM, and segmentation charts                     |

---

## Author

Nirmaldas Patel
