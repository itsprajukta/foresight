"""D5 - FORESIGHT planning dashboard for the NorthBay operations team.

    streamlit run app/dashboard.py

Reads the seeded outputs written by `python run_pipeline.py` (outputs/*.csv, outputs/metrics.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
INR = "₹"
QUAD_COLORS = {"Reorder now": "#d03b3b", "Markdown / clear": "#7a5cc7",
               "Watch / volatile": "#fab219", "Healthy": "#0ca30c"}
QUAD_ICON = {"Reorder now": "🔴", "Markdown / clear": "🟣", "Watch / volatile": "🟡", "Healthy": "🟢"}
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#0b0b0b"

st.set_page_config(page_title="FORESIGHT - NorthBay stock planner", page_icon="📦", layout="wide")


def rupees(x: float) -> str:
    if abs(x) >= 1e7:
        return f"{INR}{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"{INR}{x / 1e5:.1f} L"
    return f"{INR}{x:,.0f}"


@st.cache_data(show_spinner="Loading the latest forecast and stock position...")
def load():
    need = ["risk_scores.csv", "forecast.csv", "weekly_history.csv", "backtest_predictions.csv",
            "backtest_by_fold.csv", "metrics.json"]
    missing = [f for f in need if not (OUT / f).exists()]
    if missing:
        return None, missing
    risk = pd.read_csv(OUT / "risk_scores.csv")
    fc = pd.read_csv(OUT / "forecast.csv", parse_dates=["week_start"])
    hist = pd.read_csv(OUT / "weekly_history.csv", parse_dates=["week_start"])
    bt = pd.read_csv(OUT / "backtest_predictions.csv", parse_dates=["week_start"])
    folds = pd.read_csv(OUT / "backtest_by_fold.csv", parse_dates=["week_from", "week_to"])
    metrics = json.loads((OUT / "metrics.json").read_text())
    eda = json.loads((OUT / "eda_summary.json").read_text()) if (OUT / "eda_summary.json").exists() else {}
    return dict(risk=risk, fc=fc, hist=hist, bt=bt, folds=folds, metrics=metrics, eda=eda), []


data, missing = load()
if data is None:
    st.title("FORESIGHT")
    st.error("No forecast has been generated yet, so there is nothing to show.")
    st.write("Missing files: " + ", ".join(missing))
    st.code("python run_pipeline.py", language="bash")
    st.stop()

risk, fc, hist, bt, folds, M, E = (data[k] for k in ("risk", "fc", "hist", "bt", "folds", "metrics", "eda"))
as_of = pd.Timestamp(M["as_of"])
b = M["backtest"]

# ------------------------------------------------------------------------------------ sidebar
st.sidebar.header("Filters")
cats = sorted(risk.category.unique())
sel_cats = st.sidebar.multiselect("Category", cats, default=cats, help="Leave all selected to see every product.")
quads = list(QUAD_COLORS)
sel_quads = st.sidebar.multiselect("Status", quads, default=quads)
pool = risk[risk.category.isin(sel_cats) & risk.quadrant.isin(sel_quads)]
sku_options = ["All SKUs"] + sorted(pool.sku_id)
sel_sku = st.sidebar.selectbox("SKU", sku_options)
view = pool if sel_sku == "All SKUs" else pool[pool.sku_id == sel_sku]
st.sidebar.markdown("---")
st.sidebar.caption(f"Stock position as of **{as_of:%d %b %Y}** (latest snapshot). "
                   f"Forecast covers {M['forecast_weeks'][0]} to {M['forecast_weeks'][1]} "
                   f"({M['horizon_weeks']} weeks).")

# ------------------------------------------------------------------------------------ header
st.title("📦 FORESIGHT - what to reorder, what to clear")
st.caption(f"NorthBay Living · {len(risk)} SKUs · stock as of {as_of:%d %b %Y} · next {M['horizon_weeks']} weeks")

if view.empty:
    st.info("No SKUs match these filters. Widen the Category or Status filter in the sidebar.")
    st.stop()

reorder = view[view.quadrant.isin(["Reorder now", "Watch / volatile"])]
clear = view[view.quadrant.isin(["Markdown / clear", "Watch / volatile"])]
k1, k2, k3, k4 = st.columns(4)
k1.metric("SKUs to reorder now", len(reorder))
k2.metric("Sales at risk if not reordered", rupees(reorder.sales_at_risk_inr.sum()),
          help="Revenue from forecast demand over the next 8 weeks that stock in hand plus stock on order cannot cover.")
k3.metric("SKUs to mark down / clear", len(clear))
k4.metric("Cash locked in excess stock", rupees(clear.capital_locked_inr.sum()),
          help="Stock (at cost) still above safety stock after 8 weeks of forecast sales.")

tab_act, tab_fc, tab_grid, tab_how = st.tabs(["✅ Action lists", "📈 Forecast vs actual", "🧭 Risk grid", "ℹ️ How to read this"])

# ------------------------------------------------------------------------------- action lists
with tab_act:
    st.subheader("Reorder now - highest sales at risk first")
    if reorder.empty:
        st.success("Nothing needs reordering in this selection.")
    else:
        t = reorder.sort_values("sales_at_risk_inr", ascending=False)
        st.dataframe(pd.DataFrame({
            "SKU": t.sku_id, "Product": t.product_name, f"Sales at risk ({INR})": t.sales_at_risk_inr.round(0),
            "Suggested order (units)": t.suggested_order_units, "Stockout risk": t.stockout_risk,
            "Days of stock left": t.days_of_cover, "Lead time (days)": t.lead_time_days,
            "On hand": t.on_hand_units, "On order": t.on_order_units,
            "Forecast next 8 wks": t.forecast_horizon_units.round(0), "Category": t.category,
        }), hide_index=True, width="stretch", column_config={
            "Stockout risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            f"Sales at risk ({INR})": st.column_config.NumberColumn(format="localized"),
        })
        st.caption("Suggested order = units needed to cover forecast demand for the next 8 weeks and restore safety stock. "
                   "It is a recommendation for the planner - FORESIGHT does not place orders.")

    st.subheader("Mark down / clear - most cash locked first")
    if clear.empty:
        st.success("No overstocked SKUs in this selection.")
    else:
        t = clear.sort_values("capital_locked_inr", ascending=False)
        excess = (t.on_hand_units - t.safety_stock - t.forecast_horizon_units).clip(lower=0).round(0)
        st.dataframe(pd.DataFrame({
            "SKU": t.sku_id, "Product": t.product_name, f"Cash locked ({INR}, at cost)": t.capital_locked_inr.round(0),
            "Excess units": excess, "Overstock risk": t.overstock_risk, "On hand": t.on_hand_units,
            "Forecast next 8 wks": t.forecast_horizon_units.round(0), "Days of stock left": t.days_of_cover,
            "Category": t.category,
        }), hide_index=True, width="stretch", column_config={
            "Overstock risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            f"Cash locked ({INR}, at cost)": st.column_config.NumberColumn(format="localized"),
        })

    with st.expander(f"All {len(view)} SKUs in this selection"):
        t = view.sort_values("value_at_stake_inr", ascending=False)
        st.dataframe(pd.DataFrame({
            "Status": [f"{QUAD_ICON[q]} {q}" for q in t.quadrant], "SKU": t.sku_id, "Category": t.category,
            "Action": t.recommended_action, "Stockout risk": t.stockout_risk, "Overstock risk": t.overstock_risk,
            "Days of stock left": t.days_of_cover, f"{INR} at stake": t.value_at_stake_inr.round(0),
            "Forecast reliability": ["Lower" if r > 1.5 else "Normal" for r in t.forecast_reliability],
        }), hide_index=True, width="stretch", column_config={
            "Stockout risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Overstock risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            f"{INR} at stake": st.column_config.NumberColumn(format="localized"),
        })

# ---------------------------------------------------------------------------- forecast vs actual
with tab_fc:
    options = sorted(view.sku_id)
    default = sel_sku if sel_sku != "All SKUs" else view.sort_values("value_at_stake_inr", ascending=False).sku_id.iloc[0]
    sku = st.selectbox("Product", options, index=options.index(default), key="fc_sku")
    r = risk[risk.sku_id == sku].iloc[0]
    h = hist[hist.sku_id == sku].sort_values("week_start")
    f = fc[fc.sku_id == sku].sort_values("week_start")
    # Past forecasts: for each week, the backtest forecast made closest to it (shortest lead)
    past = bt[bt.sku_id == sku].sort_values("h").drop_duplicates("week_start").sort_values("week_start")

    st.markdown(f"**{sku} · {r.product_name} · {r.category}** — {QUAD_ICON[r.quadrant]} **{r.quadrant}**: {r.recommended_action}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(f.week_start) + list(f.week_start[::-1]), y=list(f.upper_80) + list(f.lower_80[::-1]),
                             fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), hoverinfo="skip", mode="lines",
                             name="Likely range (80%)"))
    fig.add_trace(go.Scatter(x=h.week_start, y=h.actual, mode="lines", name="Actual units sold", line=dict(color=INK, width=2)))
    fig.add_trace(go.Scatter(x=past.week_start, y=past.model.round(0), mode="lines", name="Past forecasts (backtest)",
                             line=dict(color=BLUE, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=f.week_start, y=f.forecast.round(0), mode="lines+markers", name="Forecast", line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=f.week_start, y=f.baseline, mode="lines", name="Same week last year", line=dict(color=ORANGE, width=1.5, dash="dash")))
    fig.add_vline(x=as_of, line_width=1, line_color="#8a8984")
    fig.update_xaxes(range=[as_of - pd.Timedelta(weeks=48), f.week_start.max() + pd.Timedelta(days=4)])
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis_title="Units per week", legend=dict(orientation="h", y=1.1), plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#e6e5e1"); fig.update_yaxes(showgrid=True, gridcolor="#e6e5e1")
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast, next 8 weeks", f"{f.forecast.sum():,.0f} units")
    c2.metric("Stock on hand + on order", f"{r.stock_position_units:,.0f} units")
    c3.metric("Days of stock left", f"{r.days_of_cover:,.0f}")
    past_ok = past.dropna(subset=["actual"])
    if len(past_ok):
        err = (past_ok.actual - past_ok.model).abs().sum() / past_ok.actual.sum()
        c4.metric("Past forecast error (this SKU)", f"{err:.0%}", help="Average miss vs what actually sold, over the backtest.")
    else:
        c4.metric("Past forecast error (this SKU)", "n/a")

    st.markdown("**Week-by-week forecast**")
    st.dataframe(pd.DataFrame({
        "Week starting": f.week_start.dt.strftime("%d %b %Y"), "Forecast (units)": f.forecast.round(0),
        "Likely low": f.lower_80.round(0), "Likely high": f.upper_80.round(0),
        "Same week last year": f.baseline.round(0),
        "Planned promo days": f.planned_promo_days.astype(int),
        "Actual so far": f.actual.map(lambda v: "" if pd.isna(v) else f"{v:,.0f}"),
    }), hide_index=True, width="stretch")
    st.caption("'Actual so far' fills in for weeks that have already happened after the stock date - a live check on the forecast.")

# ----------------------------------------------------------------------------------- risk grid
with tab_grid:
    st.markdown("Each bubble is a SKU. **Up** = more likely to run short before new stock can arrive. "
                "**Right** = more likely to be sitting on excess stock in 8 weeks. Bigger bubble = more rupees at stake.")
    so_t, ov_t = M["risk"]["stockout_threshold"], M["risk"]["overstock_threshold"]
    fig = go.Figure()
    mx = max(view.value_at_stake_inr.max(), 1)
    for q, g in view.groupby("quadrant"):
        fig.add_trace(go.Scatter(
            x=g.overstock_risk, y=g.stockout_risk, mode="markers", name=f"{q} ({len(g)})",
            marker=dict(size=10 + 50 * (g.value_at_stake_inr / mx) ** 0.5, color=QUAD_COLORS[q], opacity=0.65,
                        line=dict(color="white", width=2)),
            customdata=g[["sku_id", "category", "value_at_stake_inr", "days_of_cover"]].values,
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Stockout risk %{y:.0%}<br>"
                          "Overstock risk %{x:.0%}<br>At stake " + INR + "%{customdata[2]:,.0f}<br>"
                          "Days of stock %{customdata[3]:.0f}<extra></extra>"))
    fig.add_hline(y=so_t, line_width=1, line_color="#52514e")
    fig.add_vline(x=ov_t, line_width=1, line_color="#52514e")
    for x0, y0, t in [(0.25, 0.6, "REORDER NOW"), (0.75, 0.6, "WATCH / VOLATILE"),
                      (0.25, 0.05, "HEALTHY"), (0.75, 0.05, "MARKDOWN / CLEAR")]:
        fig.add_annotation(x=x0, y=y0, text=t, showarrow=False, font=dict(color="#8a8984", size=13))
    fig.update_layout(height=520, xaxis=dict(title="Overstock risk", range=[-0.05, 1.05], tickformat=".0%", gridcolor="#e6e5e1"),
                      yaxis=dict(title="Stockout risk", range=[-0.05, 1.05], tickformat=".0%", gridcolor="#e6e5e1"),
                      plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    counts = view.quadrant.value_counts()
    st.caption(" · ".join(f"{QUAD_ICON[q]} {q}: {counts.get(q, 0)}" for q in QUAD_COLORS) +
               f" · Stockout flagged above {so_t:.0%} (a {1 - so_t:.0%} service level); overstock above {ov_t:.0%}.")

# ------------------------------------------------------------------------------------ how to read
with tab_how:
    st.subheader("How much can I trust the forecast?")
    st.markdown(
        f"We tested the forecast the honest way: pretend it is an earlier date, forecast the next 8 weeks using only "
        f"data up to then, and compare with what really sold. Repeated {b['n_folds']} times across 2025.\n\n"
        f"- **FORESIGHT forecast misses by {b['model_wape']:.1%} on average** (WAPE).\n"
        f"- The simple rule 'sell the same as the same week last year' misses by **{b['baseline_wape']:.1%}**.\n"
        f"- So FORESIGHT is **{b['wape_improvement_pct']:.0f}% more accurate**, and it won in "
        f"{M['folds_won_by_model']} of {b['n_folds']} test windows.\n"
        f"- It is not systematically high or low (bias {b['model_bias']:+.1%}).")
    ho = M.get("holdout_after_as_of", {})
    if ho.get("weeks"):
        st.markdown(f"- **Live check:** for the {ho['weeks']} weeks after the stock date that have already happened, "
                    f"the forecast missed by {ho['model_wape']:.1%} vs {ho['baseline_wape']:.1%} for last-year's-sales.")
    fig = go.Figure()
    fig.add_bar(x=folds.week_from.dt.strftime("%d %b"), y=folds.baseline_wape, name="Same week last year", marker_color=ORANGE)
    fig.add_bar(x=folds.week_from.dt.strftime("%d %b"), y=folds.model_wape, name="FORESIGHT", marker_color=BLUE)
    fig.update_layout(barmode="group", height=300, yaxis=dict(title="Forecast error (lower is better)", tickformat=".0%"),
                      xaxis_title="Start of 8-week test window (2025)", plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")

    st.subheader("How the flags are decided")
    st.markdown(
        f"- **Stockout risk** - the chance that demand during the supplier lead time eats into safety stock, "
        f"counting stock on hand **and** already on order. Flagged above {so_t:.0%}.\n"
        f"- **Overstock risk** - the chance that, after 8 weeks of forecast sales, on-hand stock is still above "
        f"safety stock. Flagged above {ov_t:.0%}.\n"
        "- **Rupees at stake** - for reorders, the revenue of forecast demand the current stock cannot cover in 8 weeks; "
        "for markdowns, the cost value of stock left over the buffer after 8 weeks.\n"
        f"- **Watch / volatile** catches SKUs flagged for both. "
        f"{'None qualify this month.' if (risk.quadrant == 'Watch / volatile').sum() == 0 else str((risk.quadrant == 'Watch / volatile').sum()) + ' qualify this month.'}")
    st.subheader("Things to know")
    st.markdown(
        f"- Stock figures come from the monthly snapshot of {as_of:%d %b %Y}. Refresh after the next snapshot.\n"
        f"- {E.get('orphan_skus', 'Some')} SKUs in the stock file have stock but no sales or product record, so they are not scored.\n"
        f"- {E.get('negative_margin_skus', 'Some')} SKUs are priced below their unit cost in the product master - worth confirming with Finance.\n"
        "- Promotions are assumed to follow last year's pattern (weekends in Mar, Jun, Sep, Nov).")
