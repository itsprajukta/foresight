"""Central configuration for Project FORESIGHT.

Every tunable decision lives here so the client can see (and change) it in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "reports" / "figures"

# Raw extract file names exactly as delivered by NorthBay (two arrived as .xlsx despite the .csv in the name).
RAW_FILES = {
    "sales_daily": "sales_daily.csv",
    "inventory_snapshots": "inventory_snapshots.csv",
    "sku_master": "sku_master.csv.xlsx",
    "calendar": "calendar.csv.xlsx",
}

SEED = 42

# Forecast framing (Section 07, step 1: fix horizon and metric before modelling)
HORIZON_WEEKS = 8            # brief: "defined horizon (e.g., 6-8 weeks)"
SEASON_LENGTH = 52           # seasonal-naive = same week last year
WEEK_START = "MON"           # weeks run Monday-Sunday

# Rolling-origin backtest: origin = index of the last week the model is allowed to see.
BACKTEST_FIRST_ORIGIN = 55   # first origin with >1 year of history behind every test week
BACKTEST_STEP = 4            # a new origin every 4 weeks

# Prediction interval reported with the forecast (Section 7.2 shows an 80% band)
INTERVAL_LOWER_Q = 0.10
INTERVAL_UPPER_Q = 0.90

# Risk scoring (Section 08)
RISK_THRESHOLD = 0.50        # quadrant split on the decisioning grid

LGBM_PARAMS = dict(
    objective="l1",          # optimises absolute error, which is what WAPE measures
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)
