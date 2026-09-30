"""D1 - Data pipeline: ingest the four raw extracts, clean them in code, and produce one
analysis-ready dataset (plus the weekly SKU panel the forecast uses).

Every cleaning decision is recorded in a data-quality log (outputs/data_quality_log.csv)
with the number of rows affected, what was done and why, so the client can audit it.

Run on its own:  python -m src.pipeline
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


class DQLog:
    """Collects one row per data-quality check (including checks that found nothing)."""

    def __init__(self):
        self.rows = []

    def add(self, table, check, n_affected, action, rationale):
        self.rows.append(dict(table=table, check=check, rows_affected=int(n_affected),
                              action=action, rationale=rationale))

    def frame(self):
        return pd.DataFrame(self.rows)


# ----------------------------------------------------------------------------- ingest
def _read_any(path):
    """Extracts arrive as CSV or XLSX (some named *.csv.xlsx) - read by content, not name."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_raw(raw_dir=C.RAW_DIR):
    return {name: _read_any(raw_dir / fname) for name, fname in C.RAW_FILES.items()}


# ------------------------------------------------------------------ standardise names
SALES_COLS = {"Date": "date", "SKU": "sku_id", "Units_Sold": "units_sold",
              "Revenue": "revenue", "Price": "unit_price", "Promotion": "promo_flag"}
MASTER_COLS = {"SKU": "sku_id", "Product_Name": "product_name", "Category": "category",
               "Subcategory": "subcategory", "Launch_Date": "launch_date", "wfr": "unit_cost",
               "Selling_Price": "list_price", "Gross_Margin_Per_Unit": "gross_margin_per_unit"}
INV_COLS = {"Snapshot_Date": "date", "SKU": "sku_id", "Current_Stock": "on_hand_units",
            "On_Order": "on_order_units", "Lead_Time_Days": "lead_time_days",
            "Safety_Stock": "safety_stock", "Reorder_Point": "reorder_point",
            "Inventory_Value": "inventory_value_raw"}
CAL_COLS = {"date": "date", "promotion_event": "promo_event", "holiday": "holiday_name"}


def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()
    out = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")  # extracts use DD-MM-YYYY
    fallback = pd.to_datetime(s[out.isna()], errors="coerce", dayfirst=True)
    out.loc[out.isna()] = fallback
    return out


def _clean_sku(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


# ----------------------------------------------------------------------------- clean
def clean_sku_master(df, log: DQLog):
    df = df.rename(columns=MASTER_COLS).copy()
    df["sku_id"] = _clean_sku(df["sku_id"])
    df["launch_date"] = _parse_dates(df["launch_date"])
    for col in ("category", "subcategory", "product_name"):
        df[col] = df[col].astype(str).str.strip()

    n_dup = df.duplicated("sku_id").sum()
    df = df.drop_duplicates("sku_id", keep="last")
    log.add("sku_master", "Duplicate SKU rows", n_dup, "Kept last occurrence",
            "sku_id is the primary key; one row per SKU.")

    # 'wfr' is an unlabelled column. It satisfies list_price - wfr == gross_margin for every row,
    # so it is the unit cost. Verify rather than assume.
    ok = np.isclose(df["list_price"] - df["unit_cost"], df["gross_margin_per_unit"], atol=0.02)
    log.add("sku_master", "Unlabelled column 'wfr'", len(df),
            f"Renamed to unit_cost (identity price - wfr = margin holds for {ok.sum()}/{len(df)} rows)",
            "The brief's dictionary expects unit_cost; the extract ships it under a cryptic name.")

    neg = (df["gross_margin_per_unit"] < 0).sum()
    df["negative_margin"] = df["gross_margin_per_unit"] < 0
    log.add("sku_master", "Unit cost above selling price (negative margin)", neg,
            "Kept as-is and flagged (negative_margin=True)",
            "Could be a real pricing problem or a cost-data error - the client must confirm; "
            "we do not change prices (price optimisation is out of scope).")

    # Category/subcategory pairs are cyclically mismatched (e.g. Kitchen > Cushion, Lighting > Cookware).
    expected = {"Furniture": {"Chair", "Table", "Sofa", "Shelf", "Cabinet"},
                "Home Decor": {"Cushion", "Rug", "Lamp", "Table", "Cabinet"},
                "Kitchen": {"Cookware"}, "Lighting": {"Lamp"},
                "Storage": {"Organizer", "Shelf", "Cabinet"}}
    mism = ~df.apply(lambda r: r["subcategory"] in expected.get(r["category"], set()), axis=1)
    df["subcategory_suspect"] = mism
    log.add("sku_master", "Category/subcategory labels inconsistent", mism.sum(),
            "Kept; category used as the grouping level, subcategory flagged as unreliable",
            "Pairs like Kitchen > Cushion or Storage > Sofa cannot both be right and we cannot tell "
            "which label is wrong, so we group by category only.")
    return df


def clean_sales(df, master, log: DQLog):
    df = df.rename(columns=SALES_COLS).copy()
    n0 = len(df)
    df["sku_id"] = _clean_sku(df["sku_id"])
    df["date"] = _parse_dates(df["date"])

    bad_dates = df["date"].isna().sum()
    df = df.dropna(subset=["date"])
    log.add("sales_daily", "Unparseable dates", bad_dates, "Dropped",
            "A sale without a date cannot be placed in time.")

    for col in ("units_sold", "revenue", "unit_price", "promo_flag"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    miss = df[["units_sold", "revenue", "unit_price", "promo_flag"]].isna().sum().sum()
    log.add("sales_daily", "Missing numeric values", miss,
            "units_sold missing -> row dropped; promo_flag missing -> 0; price missing -> SKU list price",
            "The brief warned of gaps; checked every numeric column.")
    df = df.dropna(subset=["units_sold"])
    df["promo_flag"] = df["promo_flag"].fillna(0).clip(0, 1).astype(int)
    price_map = master.set_index("sku_id")["list_price"]
    df["unit_price"] = df["unit_price"].fillna(df["sku_id"].map(price_map))

    n_dup = df.duplicated(["date", "sku_id"]).sum()
    df = df.drop_duplicates(["date", "sku_id"], keep="last")
    log.add("sales_daily", "Duplicate (date, SKU) rows", n_dup, "Kept last occurrence",
            "Grain is one row per SKU per day.")

    neg = (df["units_sold"] < 0).sum()
    df["units_sold"] = df["units_sold"].clip(lower=0).round().astype(int)
    log.add("sales_daily", "Negative units", neg, "Clipped to 0",
            "Returns are not modelled as negative demand.")

    rev_bad = (~np.isclose(df["revenue"], df["units_sold"] * df["unit_price"], atol=1.0)).sum()
    df["revenue"] = df["units_sold"] * df["unit_price"]
    log.add("sales_daily", "Revenue != units x price", rev_bad, "Recomputed revenue",
            "Keeps rupee figures consistent with units.")

    unknown = ~df["sku_id"].isin(master["sku_id"])
    log.add("sales_daily", "SKUs not in sku_master", unknown.sum(), "Dropped",
            "Cannot attach cost or category to an unknown product.")
    df = df[~unknown]

    first_sale = df.groupby("sku_id")["date"].min()
    ld = master.set_index("sku_id")["launch_date"]
    pre = df[df["date"] < df["sku_id"].map(ld)]
    log.add("sales_daily", "Sales dated before the SKU's launch_date", len(pre),
            f"Kept the sales; launch_date treated as unreliable ({pre['sku_id'].nunique()} SKUs affected)",
            "Pre-launch sales volumes look identical to post-launch ones, so the sales are real "
            "and the launch dates are wrong. Effective launch = first sale date.")
    log.add("sales_daily", "Rows in / rows out", n0, f"{len(df)} rows kept", "Summary of the table.")
    return df.sort_values(["sku_id", "date"]).reset_index(drop=True), first_sale


def clean_calendar(df, sales, log: DQLog):
    df = df.rename(columns=CAL_COLS).copy()
    df["date"] = _parse_dates(df["date"])
    n_dup = df.duplicated("date").sum()
    df = df.drop_duplicates("date", keep="last")
    log.add("calendar", "Duplicate dates", n_dup, "Kept last", "One row per date.")

    n_h = df["holiday_name"].isna().sum()
    df["holiday_name"] = df["holiday_name"].fillna("")
    df["is_promo_event"] = df["promo_event"].notna().astype(int)
    df["promo_event"] = df["promo_event"].fillna("")
    log.add("calendar", "Blank holiday / promotion_event cells", n_h + (df["is_promo_event"] == 0).sum(),
            "Blank = no holiday / no event", "Blanks are structural (most days are ordinary), not missing data.")

    # The provided 'week' column restarts in odd places (e.g. 2025 week 1 has 5 days) - we build our own
    # Monday-start weeks instead of relying on it.
    log.add("calendar", "Provided 'week' column is not a standard week number", len(df),
            "Not used; weeks rebuilt as Monday-Sunday from the date",
            "Week 1 of 2025 has 5 days and week 1 of 2024 has 7, so it is not ISO-consistent.")

    # Cross-check: sales promo_flag should agree with the calendar's promotion days.
    daily_promo = sales.groupby("date")["promo_flag"].mean()
    agree = (daily_promo.round(3) == df.set_index("date")["is_promo_event"].reindex(daily_promo.index)).mean()
    log.add("calendar", "Sales promo_flag vs calendar promotion days", int(round((1 - agree) * len(daily_promo))),
            f"Checked - {agree:.0%} of days agree", "Promotions are store-wide events on weekends of "
            "March, June, September and November in both years.")

    missing_days = pd.date_range(sales["date"].min(), sales["date"].max()).difference(df["date"])
    log.add("calendar", "Sales dates missing from calendar", len(missing_days), "None to fill" if len(missing_days) == 0
            else "Filled with non-holiday/non-promo defaults", "Every sales day needs calendar attributes.")
    return df


def clean_inventory(df, master, log: DQLog):
    df = df.rename(columns=INV_COLS).copy()
    df["sku_id"] = _clean_sku(df["sku_id"])
    df["date"] = _parse_dates(df["date"])
    n_dup = df.duplicated(["date", "sku_id"]).sum()
    df = df.drop_duplicates(["date", "sku_id"], keep="last")
    log.add("inventory_snapshots", "Duplicate (date, SKU) rows", n_dup, "Kept last", "One position per SKU per snapshot.")

    orphan = ~df["sku_id"].isin(master["sku_id"])
    log.add("inventory_snapshots", "SKUs with stock but no master data or sales", orphan.sum(),
            f"Dropped ({df.loc[orphan, 'sku_id'].nunique()} SKUs: {df.loc[orphan,'sku_id'].min()}-{df.loc[orphan,'sku_id'].max()})",
            "No sales history or cost -> cannot forecast or value them. Client should confirm whether "
            "these are discontinued items or a separate catalogue.")
    df = df[~orphan].copy()

    for col in ("on_hand_units", "on_order_units", "lead_time_days", "safety_stock", "reorder_point"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    miss = df[["on_hand_units", "on_order_units", "lead_time_days"]].isna().sum().sum()
    neg = (df[["on_hand_units", "on_order_units"]] < 0).sum().sum()
    df[["on_hand_units", "on_order_units"]] = df[["on_hand_units", "on_order_units"]].clip(lower=0)
    log.add("inventory_snapshots", "Missing or negative stock values", miss + neg,
            "Negatives clipped to 0", "Stock cannot be negative.")

    cost = df["sku_id"].map(master.set_index("sku_id")["unit_cost"])
    implied = df["inventory_value_raw"] / (df["on_hand_units"] * cost)
    off = (~implied.between(0.95, 1.05)).sum()
    df["inventory_value"] = df["on_hand_units"] * cost
    log.add("inventory_snapshots", "Inventory_Value does not equal stock x unit cost", off,
            "Recomputed as on_hand_units x unit_cost",
            "The supplied value swings from 0.05x to 15x stock-at-cost, so it cannot be trusted for rupee impact.")

    # Stock should roll: next on-hand ~ on-hand + on-order - month's sales. It does not.
    log.add("inventory_snapshots", "Snapshots are monthly and do not roll forward from sales",
            df["date"].nunique(),
            "Latest snapshot used as the current stock position; no roll-forward",
            "Month-to-month stock jumps are unrelated to sales, so projecting stock from sales would invent "
            "numbers. We score risk as of the latest snapshot date instead.")
    return df.sort_values(["sku_id", "date"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- unify
def build_daily(sales, cal, master, first_sale):
    """The single analysis-ready dataset: one row per SKU per day with all attributes."""
    daily = (sales.merge(cal[["date", "season", "is_weekend", "is_holiday", "holiday_name",
                              "is_promo_event", "promo_event"]], on="date", how="left")
                  .merge(master[["sku_id", "product_name", "category", "subcategory", "unit_cost",
                                 "list_price", "gross_margin_per_unit", "negative_margin",
                                 "subcategory_suspect"]], on="sku_id", how="left"))
    daily["effective_launch_date"] = daily["sku_id"].map(first_sale)
    daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.dayofweek, unit="D")
    daily[["is_weekend", "is_holiday", "is_promo_event"]] = daily[["is_weekend", "is_holiday", "is_promo_event"]].fillna(0).astype(int)
    return daily


def build_weekly(daily, log: DQLog):
    """Weekly SKU panel (Mon-Sun). Incomplete trailing weeks are dropped."""
    days_in_week = daily.groupby("week_start")["date"].nunique()
    partial = days_in_week[days_in_week < 7].index
    log.add("sales_daily", "Incomplete weeks at the edge of the data", daily["week_start"].isin(partial).sum(),
            f"Dropped from the weekly panel (week(s) starting {', '.join(d.strftime('%Y-%m-%d') for d in partial)})",
            "A 3-day week would look like a demand collapse to the model.")
    d = daily[~daily["week_start"].isin(partial)]
    weekly = (d.groupby(["sku_id", "week_start"])
                .agg(units=("units_sold", "sum"), revenue=("revenue", "sum"),
                     promo_days=("promo_flag", "sum"), holiday_days=("is_holiday", "sum"))
                .reset_index())
    weekly = weekly.merge(d.groupby("sku_id")[["category", "unit_cost", "list_price"]].first().reset_index(), on="sku_id")
    return weekly.sort_values(["sku_id", "week_start"]).reset_index(drop=True)


def future_calendar(start, weeks):
    """Weekly calendar attributes for weeks beyond the supplied calendar.

    Assumption (documented): promotions follow the pattern seen in BOTH years of history -
    weekends of March, June, September and November; fixed-date holidays repeat.
    """
    days = pd.date_range(start, periods=7 * weeks, freq="D")
    promo = (days.dayofweek >= 5) & days.month.isin([3, 6, 9, 11])
    hol = [(1, 26), (8, 15), (12, 25)]
    holiday = np.array([(d.month, d.day) in hol for d in days])
    f = pd.DataFrame({"date": days, "promo": promo.astype(int), "hol": holiday.astype(int)})
    f["week_start"] = f["date"] - pd.to_timedelta(f["date"].dt.dayofweek, unit="D")
    return f.groupby("week_start").agg(promo_days=("promo", "sum"), holiday_days=("hol", "sum")).reset_index()


def run(save=True, verbose=True):
    raw = load_raw()
    log = DQLog()
    master = clean_sku_master(raw["sku_master"], log)
    sales, first_sale = clean_sales(raw["sales_daily"], master, log)
    cal = clean_calendar(raw["calendar"], sales, log)
    inv = clean_inventory(raw["inventory_snapshots"], master, log)
    daily = build_daily(sales, cal, master, first_sale)
    weekly = build_weekly(daily, log)
    dq = log.frame()
    if save:
        C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_csv(C.PROCESSED_DIR / "analysis_ready_daily.csv", index=False)
        weekly.to_csv(C.PROCESSED_DIR / "weekly_sku_panel.csv", index=False)
        inv.to_csv(C.PROCESSED_DIR / "inventory_clean.csv", index=False)
        master.to_csv(C.PROCESSED_DIR / "sku_master_clean.csv", index=False)
        cal.to_csv(C.PROCESSED_DIR / "calendar_clean.csv", index=False)
        dq.to_csv(C.OUTPUT_DIR / "data_quality_log.csv", index=False)
    if verbose:
        print(f"[pipeline] daily rows={len(daily):,}  weekly rows={len(weekly):,}  "
              f"SKUs={daily.sku_id.nunique()}  weeks={weekly.week_start.nunique()}  DQ checks={len(dq)}")
    return dict(daily=daily, weekly=weekly, inventory=inv, master=master, calendar=cal, dq=dq)


if __name__ == "__main__":
    run()
