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
├── src/                           ← All analysis modules (imported by main.py)
|   ├── main.py                    ← Single entry point — runs the full pipeline
│   ├── config.py                   ← All constants, paths & hyperparameters
│   ├── preprocessing.py           ← Data loading, cleaning, feature engineering
│   ├── eda.py                     ← Exploratory Data Analysis charts
│   ├── rfm.py                     ← RFM scoring & visualisations
│   ├── segmentation.py            ← K-Means clustering, labelling & charts
│   └── exporter.py                ← CSV export & console summaries
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

| Column        | Type          | Description                                   |
|---------------|---------------|-----------------------------------------------|
| `InvoiceNo`   | Categorical   | Invoice identifier; prefix `C` = cancellation   |
| `StockCode`   | Categorical   | Product/item identifier                        |
| `Description` | Text          | Product name                                  |
| `Quantity`    | Numeric       | Units purchased per line                      |
| `InvoiceDate` | DateTime      | Date and time of transaction                  |
| `UnitPrice`   | Numeric       | Price per unit in GBP (£)                     |
| `CustomerID`  | Categorical   | Unique customer identifier                     |
| `Country`     | Categorical   | Country of purchase                           |

> **Note:** `online_retail.csv` is the pre-combined file. The merging of the two source parts is intentionally excluded from this codebase per assignment requirements.

---

## Setup

### Python dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline
```bash
python main.py
```

---

## Module Reference

| Module                    | Responsibility                                                                    |
|---------------------------|-----------------------------------------------------------------------------------|
| `config.py`                | All constants: paths, colours, hyperparameters (edit here to tune)                |
| `data_preprocessing.py`   | `load_raw()`, `clean()`, `load_and_clean()`, `print_summary()`                    |
| `eda.py`                  | `run_all()` + individual `plot_*()` functions for each chart                      |
| `rfm_analysis.py`         | `compute()` builds the RFM table; `plot_distributions()`, `plot_heatmap()`        |
| `k_means.py`              | `fit()` → KMeans; `label_clusters()` → segment names; 4 plot functions             |
| `export.py`               | `save_customer_segments()`, `save_segment_summary()`, `print_segment_summary()`   |

Each module exposes a `run_*()` function that `main.py` calls in sequence.

---

## Pipeline Steps

```
Step 1  preprocessing.load_and_clean()   Load CSV → clean → engineer features
Step 2  eda.run_all()                    6 insight charts + key metrics dict
Step 3  rfm.run_all()                    Compute R/F/M scores + 2 charts
Step 4  segmentation.run_all()           K-Means → label clusters + 4 charts
Step 5  exporter.run_all()               Save customer_segments.csv + segment_summary.csv
Step 6  report_builder.build()           Embed all 12 charts into Word report
```

---

## Analysis Overview

### EDA Insights
| Chart                     | Key Finding                                           |
|---------------------------|-------------------------------------------------------|
| Monthly Revenue           | Peaks in November 2011 (pre-holiday spike)            |
| Day/Hour                  | Thursdays + 10 AM–2 PM = peak purchase window         |
| Top Countries             | Netherlands, Ireland, Germany lead internationally    |
| Top Products              | Regency Cakestand, Jumbo Bag dominate revenue         |
| Order Value Distribution  | Right-skewed; bulk buyers inflate the mean             |
| Repeat vs One-time        | ~72% repeat buyers — strong retention signal          |

### Customer Segments (K=4)
| Segment           | % Customers | % Revenue | Strategy                                |
|-------------------|-------------|-----------|-----------------------------------------|
| Champions         | 16%         | ~65%      | VIP rewards, early access, advocacy     |
| Loyal Customers   | 27%         | ~24%      | Upsell, bundle promotions               |
| At-Risk           | 19%         | ~5%       | Win-back campaigns, personalised offers |
| Promising         | 37%         | ~6%       | Re-engagement, "We miss you" campaigns  |

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

| File                      | Description                                               |
|---------------------------|-----------------------------------------------------------|
| `customer_segments.csv`   | Full RFM scores + segment label per customer (4,338 rows) |
| `segment_summary.csv`     | Aggregated stats per segment                              |
| `*.png` (12 files)         | All EDA, RFM, and segmentation charts                     |

---

## Author
Nirmaldas Patel