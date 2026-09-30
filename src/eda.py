"""D2 - Exploratory analysis: the numbers and labelled charts behind the EDA memo and readout.

Writes reports/figures/*.png and outputs/eda_summary.json. Called by run_pipeline.py.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates
import numpy as np
import pandas as pd

from . import config as C

# Palette (validated reference palette: categorical slots 1-2, status colours for quadrant states)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, BLUE_LIGHT = "#2a78d6", "#eb6834", "#86b6ef"
QUAD_COLORS = {"Reorder now": "#d03b3b", "Markdown / clear": "#7a5cc7",
               "Watch / volatile": "#fab219", "Healthy": "#0ca30c"}
INR = "₹"


def _style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10, "font.family": "DejaVu Sans",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def _rupees(x):
    if abs(x) >= 1e7:
        return f"{INR}{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"{INR}{x / 1e5:.1f} L"
    return f"{INR}{x:,.0f}"


def make_figures(data, bt, fc, risk, metrics):
    _style()
    daily, weekly, inv, master = data["daily"], data["weekly"], data["inventory"], data["master"]
    summary = {}

    # 1. Weekly demand, 2024 vs 2025 by week of year -> trend + seasonality ------------------------
    tot = weekly.groupby("week_start")["units"].sum().reset_index()
    tot["year"] = tot.week_start.dt.year
    tot["woy"] = (tot.week_start.dt.dayofyear - 1) // 7 + 1
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for yr, col, lw in [(2024, BLUE_LIGHT, 1.8), (2025, BLUE, 2.2)]:
        g = tot[tot.year == yr]
        ax.plot(g.woy, g.units, color=col, lw=lw, label=str(yr))
        ax.annotate(str(yr), (g.woy.iloc[-1], g.units.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
    ax.set_title("Demand repeats the same shape every year - spring peak, autumn dip")
    ax.set_xlabel("Week of year"); ax.set_ylabel("Units sold per week (all SKUs)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="upper right")
    _save(fig, "01_weekly_demand_by_year.png")
    y24 = tot[tot.year == 2024].units.sum(); y25 = tot[(tot.year == 2025)].units.sum()
    n24 = (tot.year == 2024).sum(); n25 = (tot.year == 2025).sum()
    summary["yoy_weekly_avg_change_pct"] = 100 * ((y25 / n25) / (y24 / n24) - 1)
    summary["weekly_units_min_max"] = [float(tot.units.min()), float(tot.units.max())]

    # 2. Seasonality by month ---------------------------------------------------------------------
    mon = daily.groupby(daily.date.dt.month)["units_sold"].mean()
    idx = 100 * mon / mon.mean()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.bar(labels, idx.values - 100, bottom=100, color=[BLUE if v >= 100 else ORANGE for v in idx.values], width=0.7)
    ax.axhline(100, color=INK2, lw=0.8)
    for i, v in enumerate(idx.values):
        ax.text(i, v + (1.5 if v >= 100 else -1.5), f"{v:.0f}", ha="center", va="bottom" if v >= 100 else "top",
                fontsize=8, color=INK2)
    hi, lo = idx.idxmax(), idx.idxmin()
    ax.set_title(f"{labels[hi-1]} is the busiest month ({idx[hi]-100:+.0f}% vs average); "
                 f"{labels[lo-1]} the quietest ({idx[lo]-100:+.0f}%)")
    ax.set_ylabel("Demand index (average month = 100)")
    ax.set_ylim(70, 130)
    _save(fig, "02_monthly_seasonality.png")
    summary["month_index"] = {labels[i]: round(float(v), 1) for i, v in enumerate(idx.values)}

    # 3. Demand drivers: weekend, promotion, holiday ------------------------------------------------
    base_wd = daily[(daily.is_weekend == 0) & (daily.promo_flag == 0) & (daily.is_holiday == 0)].units_sold.mean()
    we = daily[(daily.is_weekend == 1) & (daily.promo_flag == 0)].units_sold.mean()
    pr = daily[(daily.is_weekend == 1) & (daily.promo_flag == 1)].units_sold.mean()
    ho = daily[(daily.is_holiday == 1)].units_sold.mean()
    all_mean = daily.units_sold.mean()
    lifts = {"Weekend\n(no promo)": 100 * (we / base_wd - 1),
             "Promotion weekend\nvs normal weekend": 100 * (pr / we - 1),
             "Public holiday\nvs average day": 100 * (ho / all_mean - 1)}
    fig, ax = plt.subplots(figsize=(8, 3.3))
    names, vals = list(lifts), list(lifts.values())
    ax.barh(names, vals, color=[BLUE if v >= 0 else ORANGE for v in vals], height=0.5)
    ax.axvline(0, color=INK2, lw=0.8)
    for i, v in enumerate(vals):
        ax.text(v + (1 if v >= 0 else -1), i, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=10)
    ax.set_xlim(min(vals) - 12, max(vals) + 12)
    ax.invert_yaxis()
    ax.set_title("What moves daily demand")
    ax.set_xlabel("Change in average units sold per SKU per day")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    _save(fig, "03_demand_drivers.png")
    summary["lift_pct"] = {k.replace("\n", " "): round(v, 1) for k, v in lifts.items()}

    # 4. Category mix ------------------------------------------------------------------------------
    cat = daily.groupby("category").agg(revenue=("revenue", "sum"), units=("units_sold", "sum")).sort_values("revenue")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(cat.index, cat.revenue / 1e7, color=BLUE, height=0.55)
    for i, v in enumerate(cat.revenue / 1e7):
        ax.text(v + 0.3, i, f"{INR}{v:.1f} Cr", va="center", fontsize=9, color=INK)
    ax.set_title("Revenue by category, Jan 2024 - Dec 2025")
    ax.set_xlabel(f"Revenue ({INR} crore)")
    ax.set_xlim(0, cat.revenue.max() / 1e7 * 1.18)
    _save(fig, "04_category_revenue.png")
    summary["category_revenue_cr"] = {k: round(v / 1e7, 2) for k, v in cat.revenue.items()}

    # 5. Top movers - YoY change, last 26 full weeks before as-of vs same weeks a year earlier ------
    as_of = pd.Timestamp(metrics["as_of"])
    last = weekly[weekly.week_start < as_of].week_start.drop_duplicates().sort_values().iloc[-26:]
    prev = last - pd.Timedelta(weeks=52)
    cur = weekly[weekly.week_start.isin(last)].groupby("sku_id").units.sum()
    old = weekly[weekly.week_start.isin(prev)].groupby("sku_id").units.sum()
    yoy = (100 * (cur / old - 1)).sort_values()
    movers = pd.concat([yoy.head(5), yoy.tail(5)])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(movers.index, movers.values, color=[BLUE if v >= 0 else ORANGE for v in movers.values], height=0.6)
    ax.axvline(0, color=INK2, lw=0.8)
    for i, v in enumerate(movers.values):
        ax.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    lim = max(abs(movers.values)) + 3
    ax.set_xlim(-lim, lim)
    ax.set_title(f"Top movers: even the biggest year-on-year changes are small "
                 f"({yoy.min():+.0f}% to {yoy.max():+.0f}%)")
    ax.set_xlabel("Units, last 26 weeks vs same 26 weeks a year earlier")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    _save(fig, "05_top_movers_yoy.png")
    summary["yoy_by_sku_pct"] = {"min": round(float(yoy.min()), 1), "max": round(float(yoy.max()), 1),
                                 "top_up": {k: round(v, 1) for k, v in yoy.tail(5)[::-1].items()},
                                 "top_down": {k: round(v, 1) for k, v in yoy.head(5).items()}}

    # 6. Revenue concentration (best-sellers) --------------------------------------------------------
    rev = daily.groupby("sku_id").revenue.sum().sort_values(ascending=False)
    share = rev.cumsum() / rev.sum()
    summary["top10_revenue_share_pct"] = round(100 * float(share.iloc[9]), 1)
    summary["top10_skus"] = list(rev.index[:10])
    summary["skus_for_80pct_revenue"] = int((share < 0.8).sum() + 1)

    # 7. Dead / slow stock: days of cover at the latest snapshot ------------------------------------
    recent = weekly[weekly.week_start < as_of].groupby("sku_id").apply(lambda g: g.sort_values("week_start").units.iloc[-8:].mean() / 7, include_groups=False)
    snap = inv[inv.date == as_of].set_index("sku_id")
    cover = (snap.on_hand_units / recent).sort_values(ascending=False)
    top = cover.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(top.index, top.values, color=[ORANGE if v > 90 else BLUE for v in top.values], height=0.6)
    ax.axvline(90, color=INK2, lw=0.8)
    ax.text(92, 0, "90 days", color=INK2, fontsize=8, va="bottom")
    for i, v in enumerate(top.values):
        ax.text(v + 4, i, f"{v:.0f} d", va="center", fontsize=8)
    ax.set_title(f"Slow stock: on-hand cover at {as_of:%d %b %Y}, 15 highest SKUs")
    ax.set_xlabel("Days of sales the on-hand stock would last (at the last 8 weeks' rate)")
    ax.set_xlim(0, top.max() * 1.12)
    _save(fig, "06_days_of_cover.png")
    cost = master.set_index("sku_id").unit_cost
    slow = cover[cover > 90]
    summary["slow_stock_skus_over_90d"] = {k: round(float(v)) for k, v in slow.items()}
    summary["slow_stock_value_inr"] = float((snap.loc[slow.index, "on_hand_units"] * cost[slow.index]).sum())
    summary["median_days_cover"] = round(float(cover.median()), 1)

    raw_inv = pd.read_csv(C.RAW_DIR / C.RAW_FILES["inventory_snapshots"])
    raw_inv["d"] = pd.to_datetime(raw_inv.Snapshot_Date, format="%d-%m-%Y")
    orphan = raw_inv[(raw_inv.d == as_of) & (~raw_inv.SKU.isin(master.sku_id))]
    summary["orphan_skus"] = int(orphan.SKU.nunique())
    summary["orphan_units_on_hand"] = int(orphan.Current_Stock.sum())
    summary["orphan_value_supplied_inr"] = float(orphan.Inventory_Value.sum())

    # 8. Stock does not roll forward from sales (why we don't project stock) -------------------------
    iv = inv.sort_values(["sku_id", "date"]).copy()
    iv["next_on_hand"] = iv.groupby("sku_id").on_hand_units.shift(-1)
    msales = daily.groupby(["sku_id", daily.date.dt.to_period("M").dt.to_timestamp()]).units_sold.sum()
    iv["month_sales"] = [msales.get((s, d), np.nan) for s, d in zip(iv.sku_id, iv.date)]
    iv["expected_next"] = iv.on_hand_units + iv.on_order_units - iv.month_sales
    ok = iv.dropna(subset=["next_on_hand", "month_sales"])
    err = (ok.next_on_hand - ok.expected_next).abs() / ok.next_on_hand
    summary["stock_rollforward_within_20pct_share"] = round(float((err < 0.2).mean()), 3)
    summary["stock_rollforward_negative_share"] = round(float((ok.expected_next < 0).mean()), 3)
    summary["stock_change_vs_sales_corr"] = round(float(np.corrcoef(ok.next_on_hand - ok.on_hand_units, -ok.month_sales)[0, 1]), 3)

    # 9. Negative-margin SKUs ----------------------------------------------------------------------
    neg = master[master.negative_margin].sku_id
    summary["negative_margin_skus"] = int(len(neg))
    summary["negative_margin_revenue_share_pct"] = round(100 * float(rev[neg].sum() / rev.sum()), 1)
    gm = (daily.units_sold * daily.gross_margin_per_unit)
    summary["negative_margin_gross_loss_inr"] = float(-gm[daily.negative_margin].sum())

    # 10. Price vs volume across SKUs (correlation check) -----------------------------------------
    pv = daily.groupby("sku_id").agg(price=("unit_price", "first"), units=("units_sold", "mean"))
    summary["price_volume_corr"] = round(float(pv.corr().iloc[0, 1]), 3)

    # 11. Illustrative forecast (Section 7.2): actual, baseline, forecast + 80% band ---------------
    sku = risk.iloc[0].sku_id
    h = weekly[(weekly.sku_id == sku)].sort_values("week_start")
    f = fc[fc.sku_id == sku].sort_values("week_start")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(h.week_start, h.units, color=INK, lw=1.6, label="Actual demand")
    ax.fill_between(f.week_start, f.lower_80, f.upper_80, color=BLUE, alpha=0.15, lw=0, label="80% range")
    ax.plot(f.week_start, f.forecast, color=BLUE, lw=2.2, label="FORESIGHT forecast")
    ax.plot(f.week_start, f.baseline, color=ORANGE, lw=1.6, ls=(0, (4, 3)), label="Same week last year (baseline)")
    ax.axvline(as_of, color=MUTED, lw=0.8)
    ax.text(as_of, ax.get_ylim()[1], "  forecast from stock date", color=INK2, fontsize=8, va="top")
    ax.set_xlim(pd.Timestamp("2025-03-01"), f.week_start.max() + pd.Timedelta(days=5))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.set_title(f"{sku}: weekly demand, baseline and 8-week forecast")
    ax.set_ylabel("Units per week")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    _save(fig, "07_forecast_example.png")
    summary["forecast_example_sku"] = sku

    # 12. Backtest: WAPE per fold -------------------------------------------------------------------
    pf = pd.read_csv(C.OUTPUT_DIR / "backtest_by_fold.csv", parse_dates=["week_from"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(pf)); w = 0.38
    ax.bar(x - w / 2 - 0.01, 100 * pf.baseline_wape, w, color=ORANGE, label="Seasonal-naive baseline")
    ax.bar(x + w / 2 + 0.01, 100 * pf.model_wape, w, color=BLUE, label="FORESIGHT model")
    ax.set_xticks(x, [d.strftime("%d %b\n%Y") for d in pf.week_from], fontsize=8)
    ax.set_ylabel("WAPE (lower is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    b = metrics["backtest"]
    ax.set_title(f"Model beats the baseline in all {len(pf)} backtest windows "
                 f"({100 * b['model_wape']:.1f}% vs {100 * b['baseline_wape']:.1f}% WAPE)")
    ax.set_xlabel("Start of each 8-week test window")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0, 100 * pf.baseline_wape.max() * 1.3)
    _save(fig, "08_backtest_wape_by_fold.png")

    # 13. Decisioning grid (Section 8.2) --------------------------------------------------------------
    from .risk import STOCKOUT_THRESHOLD, OVERSTOCK_THRESHOLD
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = 30 + 1400 * np.sqrt(risk.value_at_stake_inr / max(risk.value_at_stake_inr.max(), 1))
    for q, g in risk.groupby("quadrant"):
        ax.scatter(g.overstock_risk, g.stockout_risk, s=sizes[g.index], color=QUAD_COLORS[q], alpha=0.55,
                   edgecolor="white", linewidth=1.5, zorder=3)
    ax.axhline(STOCKOUT_THRESHOLD, color=INK2, lw=0.8); ax.axvline(OVERSTOCK_THRESHOLD, color=INK2, lw=0.8)
    ax.text(1.04, STOCKOUT_THRESHOLD + 0.01, f"stockout flag > {STOCKOUT_THRESHOLD:.0%}", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.text(OVERSTOCK_THRESHOLD + 0.01, 0.55, f"overstock flag > {OVERSTOCK_THRESHOLD:.0%}", rotation=90, fontsize=7, color=INK2)
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.13, 1.06)
    for (x0, y0, t) in [(0.25, 0.62, "REORDER NOW"), (0.75, 0.62, "WATCH / VOLATILE"),
                        (0.25, 0.04, "HEALTHY"), (0.75, 0.04, "MARKDOWN / CLEAR")]:
        ax.text(x0, y0, t, ha="center", fontsize=10, fontweight="bold", color=MUTED)
    # one label per cluster of (near-)identical positions, so overlapping SKUs stay readable
    flagged = risk[risk.quadrant != "Healthy"].copy()
    flagged["key"] = list(zip(flagged.overstock_risk.round(1), flagged.stockout_risk.round(1)))
    for key, g in flagged.groupby("key"):
        x0, y0 = g.overstock_risk.mean(), g.stockout_risk.mean()
        names = ", ".join(g.sort_values("value_at_stake_inr", ascending=False).sku_id)
        right = x0 < 0.5
        ax.annotate(names, (x0, y0), xytext=(14 if right else -14, -14 if y0 > 0.5 else (12 if y0 > 0.05 else -26)),
                    textcoords="offset points", fontsize=7.5, color=INK, ha="left" if right else "right")
    handles = [Line2D([], [], marker="o", ls="", markersize=8, color=QUAD_COLORS[q], alpha=0.8,
                      label=f"{q} ({int((risk.quadrant == q).sum())})") for q in QUAD_COLORS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, title="SKUs", title_fontsize=8)
    ax.set_xlabel("Overstock risk: chance of still holding more than safety stock after 8 weeks")
    ax.set_ylabel("Stockout risk: chance of dipping into\nsafety stock within the lead time")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_title(f"Every SKU on the decision grid (bubble size = {INR} at stake)")
    _save(fig, "09_decision_grid.png")

    with open(C.OUTPUT_DIR / "eda_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return summary
