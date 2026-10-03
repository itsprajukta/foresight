"""D5 - FORESIGHT planning dashboard for the NorthBay operations team.

    streamlit run app/dashboard.py

Reads the seeded outputs written by `python run_pipeline.py` (outputs/*.csv, outputs/metrics.json).
The what-if panel re-scores risk with the same rule the pipeline uses (src/risk.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import risk as R  # noqa: E402

OUT = ROOT / "outputs"
INR = "₹"
INK, INK2, MUTED, LINE = "#1b2430", "#4f5a68", "#8a94a3", "#e3e7ec"
BLUE, ORANGE = "#1f5fad", "#e0662f"
QUAD_COLORS = {"Reorder now": "#d03b3b", "Markdown / clear": "#7a5cc7",
               "Watch / volatile": "#d99a00", "Healthy": "#1a8f3c"}
QUAD_TINT = {"Reorder now": "#fdecec", "Markdown / clear": "#f1edfb",
             "Watch / volatile": "#fdf4dc", "Healthy": "#e8f5ec"}
VIEWS = ["Action lists", "Product detail", "Risk map", "Forecast accuracy"]

st.set_page_config(page_title="FORESIGHT stock planner", page_icon="📦", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select, [data-testid="stMetricValue"] {{
  font-family: 'Public Sans', system-ui, sans-serif; }}
.stApp {{ background: #ffffff; color: {INK}; }}
[data-testid="stHeader"] {{ display: none; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.lede {{ font-size: 2.55rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.025em; margin: 0 0 .45rem; color: {INK}; }}
.lede .r {{ color: {QUAD_COLORS['Reorder now']}; }} .lede .c {{ color: {QUAD_COLORS['Markdown / clear']}; }}
.sub {{ color: {INK2}; font-size: 1.02rem; max-width: 70ch; margin-bottom: 1.3rem; }}
.figs {{ display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid {LINE};
        border-bottom: 1px solid {LINE}; margin: 0 0 1.4rem; }}
.fig {{ padding: .85rem 1rem .9rem; border-left: 1px solid {LINE}; }}
.fig:first-child {{ border-left: none; padding-left: 0; }}
.fig .v {{ font-size: 1.65rem; font-weight: 700; font-variant-numeric: tabular-nums; color: {INK}; }}
.fig .l {{ font-size: .86rem; color: {INK2}; margin-top: .15rem; line-height: 1.35; }}
.badge {{ display: inline-flex; align-items: center; gap: .4rem; padding: .18rem .65rem; border-radius: 999px;
          font-weight: 600; font-size: .9rem; }}
.badge i {{ width: .55rem; height: .55rem; border-radius: 50%; display: inline-block; }}
.hint {{ color: {MUTED}; font-size: .88rem; }}
.whatif-out {{ border-left: 3px solid var(--q); padding: .2rem 0 .2rem .9rem; margin: .3rem 0 .6rem; }}
@media (max-width: 760px) {{ .figs {{ grid-template-columns: repeat(2, 1fr); }} .fig:nth-child(3) {{ border-left: none; padding-left: 0; }}
  .lede {{ font-size: 1.9rem; }} }}
</style>""", unsafe_allow_html=True)


def rupees(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e7:
        return f"{INR}{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"{INR}{x / 1e5:.1f} L"
    return f"{INR}{x:,.0f}"


def badge(q: str) -> str:
    return (f"<span class='badge' style='background:{QUAD_TINT[q]};color:{QUAD_COLORS[q]}'>"
            f"<i style='background:{QUAD_COLORS[q]}'></i>{q}</span>")


@st.cache_data(show_spinner="Loading the latest forecast and stock position...")
def load():
    need = ["risk_scores.csv", "forecast.csv", "weekly_history.csv", "backtest_predictions.csv",
            "backtest_by_fold.csv", "metrics.json"]
    missing = [f for f in need if not (OUT / f).exists()]
    if missing:
        return None, missing
    d = dict(
        risk=pd.read_csv(OUT / "risk_scores.csv"),
        fc=pd.read_csv(OUT / "forecast.csv", parse_dates=["week_start"]),
        hist=pd.read_csv(OUT / "weekly_history.csv", parse_dates=["week_start"]),
        bt=pd.read_csv(OUT / "backtest_predictions.csv", parse_dates=["week_start"]),
        folds=pd.read_csv(OUT / "backtest_by_fold.csv", parse_dates=["week_from", "week_to"]),
        metrics=json.loads((OUT / "metrics.json").read_text()),
        eda=json.loads((OUT / "eda_summary.json").read_text()) if (OUT / "eda_summary.json").exists() else {},
    )
    return d, []


data, missing = load()
if data is None:
    st.markdown("<div class='lede'>No forecast yet</div>", unsafe_allow_html=True)
    st.error("The dashboard needs the pipeline outputs. Missing: " + ", ".join(missing))
    st.markdown("Run this from the project folder, then refresh the page:")
    st.code("python run_pipeline.py", language="bash")
    st.stop()

risk, fc, hist, bt, folds, M, E = (data[k] for k in ("risk", "fc", "hist", "bt", "folds", "metrics", "eda"))
as_of = pd.Timestamp(M["as_of"])
b = M["backtest"]
REL_SD = {int(k): v for k, v in M["window_rel_sd"].items()}
ALL_SKUS = sorted(risk.sku_id)

ss = st.session_state
ss.setdefault("view", VIEWS[0])
ss.setdefault("cur_sku", risk.sort_values("value_at_stake_inr", ascending=False).sku_id.iloc[0])


def open_sku(sku: str):
    if sku in ALL_SKUS:
        ss.cur_sku = sku
        ss.view = "Product detail"


def _from_table(key, frame_key):
    sel = ss[key]["selection"]
    rows = list(sel.get("rows", [])) or [c[0] for c in sel.get("cells", [])]
    if rows:
        open_sku(ss[frame_key].iloc[int(rows[0])]["SKU"])


def _from_grid():
    pts = ss["grid"]["selection"]["points"]
    if pts:
        cd = pts[0].get("customdata")
        open_sku(cd[0] if isinstance(cd, (list, tuple)) else cd)


# ------------------------------------------------------------------------------------ filters
f1, f2 = st.columns([3, 2])
cats = sorted(risk.category.unique())
sel_cats = f1.pills("Category", cats, selection_mode="multi", default=cats, key="cats") or []
sel_status = f2.segmented_control("Status", ["All", "Needs action", "Healthy"], default="All", key="status") or "All"
pool = risk[risk.category.isin(sel_cats)]
if sel_status == "Needs action":
    pool = pool[pool.quadrant != "Healthy"]
elif sel_status == "Healthy":
    pool = pool[pool.quadrant == "Healthy"]

if pool.empty:
    st.info("No products match these filters. Select at least one category, or set Status to All.")
    st.stop()

reorder = pool[pool.quadrant.isin(["Reorder now", "Watch / volatile"])].sort_values("sales_at_risk_inr", ascending=False)
clear = pool[pool.quadrant.isin(["Markdown / clear", "Watch / volatile"])].sort_values("capital_locked_inr", ascending=False)
expedite = reorder[reorder.forecast_lead_time_units > reorder.stock_position_units]

# ------------------------------------------------------------------------------------- lede
n_r, n_c = len(reorder), len(clear)
if n_r or n_c:
    parts = []
    if n_r:
        parts.append(f"<span class='r'>Reorder {n_r} product{'s' if n_r != 1 else ''}</span>")
    if n_c:
        parts.append(f"<span class='c'>clear {n_c}</span>" if n_r else f"<span class='c'>Clear {n_c} product{'s' if n_c != 1 else ''}</span>")
    lede = " and ".join(parts) + " this month."
else:
    lede = "Nothing in this selection needs action."
st.markdown(f"<div class='lede'>{lede}</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sub'>Based on stock counted on {as_of:%d %b %Y} and the forecast for the next "
    f"{M['horizon_weeks']} weeks ({pd.Timestamp(M['forecast_weeks'][0]):%d %b} to "
    f"{pd.Timestamp(M['forecast_weeks'][1]) + pd.Timedelta(days=6):%d %b %Y}). "
    f"Showing {len(pool)} of {len(risk)} products.</div>", unsafe_allow_html=True)
st.markdown(f"""<div class='figs'>
<div class='fig'><div class='v'>{rupees(reorder.sales_at_risk_inr.sum())}</div><div class='l'>sales at risk if the reorder list isn't ordered</div></div>
<div class='fig'><div class='v'>{rupees(clear.capital_locked_inr.sum())}</div><div class='l'>cash tied up in stock that won't sell in 8 weeks</div></div>
<div class='fig'><div class='v'>{len(expedite)}</div><div class='l'>products that run out before a normal order arrives</div></div>
<div class='fig'><div class='v'>{b['model_wape']:.1%}</div><div class='l'>average forecast miss, vs {b['baseline_wape']:.1%} for "same as last year"</div></div>
</div>""", unsafe_allow_html=True)

view = st.segmented_control("View", VIEWS, key="view", label_visibility="collapsed") or VIEWS[0]

# ------------------------------------------------------------------------------- action lists
if view == "Action lists":
    st.markdown("<p class='hint'>Click any row to open that product's forecast and what-if panel.</p>", unsafe_allow_html=True)
    st.subheader("Reorder now")
    if reorder.empty:
        st.success("Nothing in this selection needs reordering.")
    else:
        t = pd.DataFrame({
            "SKU": reorder.sku_id, "Category": reorder.category,
            f"Sales at risk ({INR})": reorder.sales_at_risk_inr.round(0),
            "Suggested order": reorder.suggested_order_units, "Stockout risk": reorder.stockout_risk,
            "Days of stock left": reorder.days_of_cover, "Lead time (days)": reorder.lead_time_days,
            "Expedite": ["Yes" if s in set(expedite.sku_id) else "" for s in reorder.sku_id],
            "Below cost": ["Yes" if n else "" for n in reorder.negative_margin],
        }).reset_index(drop=True)
        ss["reorder_frame"] = t
        st.dataframe(t, hide_index=True, width="stretch", key="t_reorder", on_select=lambda: _from_table("t_reorder", "reorder_frame"),
                     selection_mode="single-cell", column_config={
                         "Stockout risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                         f"Sales at risk ({INR})": st.column_config.NumberColumn(format="localized"),
                         "Suggested order": st.column_config.NumberColumn(help="Units to cover 8 weeks of forecast demand plus safety stock, minus stock in hand and on order. A recommendation; nothing is ordered automatically."),
                     })
        st.download_button("Download reorder list (CSV)", t.to_csv(index=False), "foresight_reorder_list.csv", "text/csv")

    st.subheader("Mark down or clear")
    if clear.empty:
        st.success("No overstocked products in this selection.")
    else:
        t = pd.DataFrame({
            "SKU": clear.sku_id, "Category": clear.category,
            f"Cash locked ({INR}, at cost)": clear.capital_locked_inr.round(0),
            "Excess units": (clear.on_hand_units - clear.safety_stock - clear.forecast_horizon_units).clip(lower=0).round(0),
            "Overstock risk": clear.overstock_risk, "On hand": clear.on_hand_units,
            "Forecast next 8 wks": clear.forecast_horizon_units.round(0), "Days of stock left": clear.days_of_cover,
            "Below cost": ["Yes" if n else "" for n in clear.negative_margin],
        }).reset_index(drop=True)
        ss["clear_frame"] = t
        st.dataframe(t, hide_index=True, width="stretch", key="t_clear", on_select=lambda: _from_table("t_clear", "clear_frame"),
                     selection_mode="single-cell", column_config={
                         "Overstock risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                         f"Cash locked ({INR}, at cost)": st.column_config.NumberColumn(format="localized"),
                     })
        st.download_button("Download markdown list (CSV)", t.to_csv(index=False), "foresight_markdown_list.csv", "text/csv")

    with st.expander(f"All {len(pool)} products in this selection"):
        t = pool.sort_values("value_at_stake_inr", ascending=False)
        t = pd.DataFrame({
            "SKU": t.sku_id, "Category": t.category, "Status": t.quadrant, "Action": t.recommended_action,
            "Stockout risk": t.stockout_risk, "Overstock risk": t.overstock_risk, "Days of stock left": t.days_of_cover,
            f"{INR} at stake": t.value_at_stake_inr.round(0),
        }).reset_index(drop=True)
        ss["all_frame"] = t
        st.dataframe(t, hide_index=True, width="stretch", key="t_all", on_select=lambda: _from_table("t_all", "all_frame"),
                     selection_mode="single-cell", column_config={
                         "Stockout risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                         "Overstock risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                         f"{INR} at stake": st.column_config.NumberColumn(format="localized"),
                     })

# ----------------------------------------------------------------------------- product detail
elif view == "Product detail":
    sku = st.selectbox("Product", ALL_SKUS, index=ALL_SKUS.index(ss.cur_sku))
    ss.cur_sku = sku
    r = risk[risk.sku_id == sku].iloc[0]
    h = hist[hist.sku_id == sku].sort_values("week_start")
    f = fc[fc.sku_id == sku].sort_values("week_start")
    past = bt[bt.sku_id == sku].sort_values("h").drop_duplicates("week_start").sort_values("week_start")

    st.markdown(f"<div style='margin:.2rem 0 .6rem'><b style='font-size:1.25rem'>{sku}</b> "
                f"<span class='hint'>{r.product_name}, {r.category}</span>&nbsp;&nbsp;{badge(r.quadrant)}"
                f"<div class='hint' style='margin-top:.3rem'>{r.recommended_action}</div></div>", unsafe_allow_html=True)

    c_a, c_b, c_c = st.columns([1, 1, 2])
    show_past = c_a.toggle("Past forecasts", value=True, help="What the model predicted for earlier weeks during the backtest.")
    show_np = c_b.toggle("Without promotions", value=False, help="The same forecast with planned promotion days removed.")
    span = c_c.select_slider("History shown", ["3 months", "6 months", "1 year", "All"], value="1 year")
    weeks_back = {"3 months": 13, "6 months": 26, "1 year": 52, "All": 200}[span]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(f.week_start) + list(f.week_start[::-1]), y=list(f.upper_80) + list(f.lower_80[::-1]),
                             fill="toself", fillcolor="rgba(31,95,173,0.13)", line=dict(width=0), hoverinfo="skip",
                             mode="lines", name="Likely range (8 in 10 weeks)"))
    fig.add_trace(go.Scatter(x=h.week_start, y=h.actual, mode="lines", name="Actual units sold", line=dict(color=INK, width=2)))
    if show_past:
        fig.add_trace(go.Scatter(x=past.week_start, y=past.model.round(0), mode="lines", name="Past forecasts",
                                 line=dict(color=BLUE, width=1.4, dash="dot")))
    fig.add_trace(go.Scatter(x=f.week_start, y=f.forecast.round(0), mode="lines+markers", name="Forecast",
                             line=dict(color=BLUE, width=3), marker=dict(size=7)))
    if show_np:
        fig.add_trace(go.Scatter(x=f.week_start, y=f.forecast_no_promo.round(0), mode="lines", name="Forecast without promotions",
                                 line=dict(color=BLUE, width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=f.week_start, y=f.baseline, mode="lines", name="Same week last year",
                             line=dict(color=ORANGE, width=1.5, dash="dash")))
    fig.add_vline(x=as_of, line_width=1, line_color=MUTED)
    fig.add_annotation(x=as_of, y=1.02, yref="paper", text="stock counted", showarrow=False, font=dict(color=MUTED, size=11), xanchor="left")
    fig.update_xaxes(range=[as_of - pd.Timedelta(weeks=weeks_back), f.week_start.max() + pd.Timedelta(days=4)],
                     showgrid=True, gridcolor=LINE, rangeslider=dict(visible=True, thickness=0.06, bgcolor="#f6f8fa"))
    fig.update_yaxes(showgrid=True, gridcolor=LINE, title="Units per week")
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified", plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(family="Public Sans, sans-serif", color=INK),
                      legend=dict(orientation="h", y=1.14, x=0))
    st.plotly_chart(fig, width="stretch", key="fc_chart")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forecast, next 8 weeks", f"{f.forecast.sum():,.0f} units")
    m2.metric("On hand + on order", f"{r.stock_position_units:,.0f} units")
    m3.metric("Days of stock left", f"{r.days_of_cover:,.0f}", help=f"Supplier lead time is {r.lead_time_days} days.")
    po = past.dropna(subset=["actual"])
    m4.metric("Past forecast miss (this product)", f"{(po.actual - po.model).abs().sum() / po.actual.sum():.0%}" if len(po) else "n/a")

    # ------------------------------------------------------------------ what-if
    st.subheader("What if the stock position changes?")
    st.markdown("<p class='hint'>Move the sliders to try a fresh stock count, an incoming delivery or a different supplier lead time. "
                "The forecast stays the same; the risk and recommended action are recalculated with the same rule as the plan.</p>",
                unsafe_allow_html=True)
    keys = {k: f"wi_{k}_{sku}" for k in ("oh", "oo", "lt", "ss")}
    if st.button("Reset to counted stock", key=f"reset_{sku}"):
        for k in keys.values():
            ss.pop(k, None)
        st.rerun()
    cap = int(max(r.forecast_horizon_units * 2, r.on_hand_units * 1.5, 100))
    w1, w2 = st.columns([3, 2], gap="large")
    with w1:
        oh = st.slider("Units on hand", 0, cap, int(r.on_hand_units), key=keys["oh"])
        oo = st.slider("Units on order", 0, cap, int(r.on_order_units), key=keys["oo"])
        lt = st.slider("Supplier lead time (days)", 1, 30, int(r.lead_time_days), key=keys["lt"])
        sst = st.slider("Safety stock (units)", 0, int(max(r.safety_stock * 3, 50)), int(r.safety_stock), key=keys["ss"])
    new = R.score_sku(f.forecast.tolist(), oh, oo, lt, sst, float(r.unit_cost), float(r.list_price), REL_SD, float(r.forecast_reliability))
    with w2:
        changed = new["quadrant"] != r.quadrant
        st.markdown(f"<div class='whatif-out' style='--q:{QUAD_COLORS[new['quadrant']]}'>"
                    f"<div class='hint'>Result</div><div style='margin:.25rem 0 .35rem'>{badge(new['quadrant'])}"
                    f"{'&nbsp;<span class=hint>changed from ' + r.quadrant + '</span>' if changed else ''}</div>"
                    f"<div>{new['recommended_action']}</div></div>", unsafe_allow_html=True)
        a, bb = st.columns(2)
        def _delta(now, before):
            d = (now - before) * 100
            return None if abs(d) < 0.5 else f"{d:+.0f} pts vs counted stock"
        a.metric("Stockout risk", f"{new['stockout_risk']:.0%}", delta=_delta(new["stockout_risk"], r.stockout_risk),
                 delta_color="inverse")
        bb.metric("Overstock risk", f"{new['overstock_risk']:.0%}", delta=_delta(new["overstock_risk"], r.overstock_risk),
                  delta_color="inverse")
        a.metric("Sales at risk", rupees(new["sales_at_risk_inr"]))
        bb.metric("Cash locked", rupees(new["capital_locked_inr"]))
        if new["suggested_order_units"]:
            st.markdown(f"Suggested order: **{new['suggested_order_units']:,} units**")

    st.markdown("**Week by week**")
    st.dataframe(pd.DataFrame({
        "Week starting": f.week_start.dt.strftime("%d %b %Y"), "Forecast": f.forecast.round(0),
        "Likely low": f.lower_80.round(0), "Likely high": f.upper_80.round(0),
        "Without promotions": f.forecast_no_promo.round(0), "Same week last year": f.baseline.round(0),
        "Promo days": f.planned_promo_days.astype(int),
        "Actual so far": f.actual.map(lambda v: "" if pd.isna(v) else f"{v:,.0f}"),
    }), hide_index=True, width="stretch")
    st.caption("'Actual so far' fills in for weeks that have already happened since the stock count, a live check on the forecast.")

# ----------------------------------------------------------------------------------- risk map
elif view == "Risk map":
    st.markdown("<p class='hint'>Each bubble is a product. Higher means more likely to run short before new stock can arrive; "
                "further right means more likely to be left with excess stock in 8 weeks. Bubble size is the rupee value at stake. "
                "Click a bubble to open that product.</p>", unsafe_allow_html=True)
    so_t, ov_t = M["risk"]["stockout_threshold"], M["risk"]["overstock_threshold"]
    fig = go.Figure()
    mx = max(pool.value_at_stake_inr.max(), 1)
    # Products with (near-)identical scores would sit on top of each other and only one could be clicked:
    # nudge them apart horizontally (display only; hover shows the true values).
    pts = pool.copy()
    pts["key"] = list(zip(pts.overstock_risk.round(2), pts.stockout_risk.round(2)))
    pts["rank"] = pts.sort_values("value_at_stake_inr", ascending=False).groupby("key").cumcount()
    pts.loc[pts.quadrant == "Healthy", "rank"] = 0  # healthy points stay where they are
    side = (pts.overstock_risk > 0.5).map({True: -1, False: 1})
    pts["xp"] = pts.overstock_risk + side * pts["rank"] * 0.045
    for q in QUAD_COLORS:
        g = pts[pts.quadrant == q]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g.xp, y=g.stockout_risk, mode="markers", name=f"{q} ({len(g)})",
            marker=dict(size=12 + 50 * (g.value_at_stake_inr / mx) ** 0.5, color=QUAD_COLORS[q], opacity=0.6,
                        line=dict(color="white", width=2)),
            customdata=g[["sku_id", "category", "value_at_stake_inr", "days_of_cover", "overstock_risk"]].values,
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Stockout risk %{y:.0%}<br>"
                          "Overstock risk %{customdata[4]:.0%}<br>At stake " + INR + "%{customdata[2]:,.0f}<br>"
                          "Days of stock %{customdata[3]:.0f}<extra>Click to open</extra>"))
    fig.add_hline(y=so_t, line_width=1, line_color=INK2)
    fig.add_vline(x=ov_t, line_width=1, line_color=INK2)
    for x0, y0, t in [(0.25, 0.62, "Reorder now"), (0.75, 0.62, "Watch / volatile"),
                      (0.25, 0.04, "Healthy"), (0.75, 0.04, "Markdown / clear")]:
        fig.add_annotation(x=x0, y=y0, text=t, showarrow=False, font=dict(color=MUTED, size=14))
    fig.update_layout(height=540, plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(family="Public Sans, sans-serif", color=INK), clickmode="event+select",
                      xaxis=dict(title="Overstock risk", range=[-0.06, 1.06], tickformat=".0%", gridcolor=LINE),
                      yaxis=dict(title="Stockout risk", range=[-0.06, 1.08], tickformat=".0%", gridcolor=LINE),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch", key="grid", on_select=_from_grid, selection_mode="points")
    st.caption(f"Stockout is flagged above {so_t:.0%} (a {1 - so_t:.0%} service level) and overstock above {ov_t:.0%}. "
               "Products with identical scores are spaced slightly apart so each can be clicked. "
               f"{'No product is flagged for both this month.' if (risk.quadrant == 'Watch / volatile').sum() == 0 else ''}")

# ---------------------------------------------------------------------------- forecast accuracy
else:
    st.subheader("How much can the forecast be trusted?")
    ho = M.get("holdout_after_as_of", {})
    st.markdown(
        f"We tested it the honest way: pretend it's an earlier date, forecast the next 8 weeks with only the data up to then, "
        f"and compare with what really sold. We did this {b['n_folds']} times across 2025.\n\n"
        f"- FORESIGHT misses by **{b['model_wape']:.1%}** on average. \"Sell the same as the same week last year\" misses by "
        f"**{b['baseline_wape']:.1%}**. That's {b['wape_improvement_pct']:.0f}% more accurate, and better in all "
        f"{M['folds_won_by_model']} test windows.\n"
        f"- It isn't systematically high or low (bias {b['model_bias']:+.1%}).\n"
        + (f"- Live check: over the {ho['weeks']} weeks since the stock count, it missed by {ho['model_wape']:.1%} "
           f"against {ho['baseline_wape']:.1%}." if ho.get("weeks") else ""))
    metric = st.segmented_control("Compare by", ["Test window", "Weeks ahead"], default="Test window", key="acc_by") or "Test window"
    fig = go.Figure()
    if metric == "Test window":
        x = folds.week_from.dt.strftime("%d %b")
        fig.add_bar(x=x, y=folds.baseline_wape, name="Same week last year", marker_color=ORANGE)
        fig.add_bar(x=x, y=folds.model_wape, name="FORESIGHT", marker_color=BLUE)
        xt = "Start of 8-week test window (2025)"
    else:
        by_h = bt.groupby("h").apply(lambda g: pd.Series({
            "model": (g.actual - g.model).abs().sum() / g.actual.sum(),
            "base": (g.actual - g.baseline).abs().sum() / g.actual.sum()}), include_groups=False).reset_index()
        fig.add_bar(x=by_h.h, y=by_h.base, name="Same week last year", marker_color=ORANGE)
        fig.add_bar(x=by_h.h, y=by_h.model, name="FORESIGHT", marker_color=BLUE)
        xt = "Weeks ahead"
    fig.update_layout(barmode="group", height=340, plot_bgcolor="white", paper_bgcolor="white", bargap=0.25,
                      yaxis=dict(title="Average miss (lower is better)", tickformat=".0%", gridcolor=LINE),
                      xaxis_title=xt, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12),
                      font=dict(family="Public Sans, sans-serif", color=INK))
    st.plotly_chart(fig, width="stretch", key="acc_chart")

    st.subheader("How the flags are decided")
    st.markdown(
        f"- **Stockout risk:** the chance that demand during the supplier lead time eats into safety stock, counting stock on hand "
        f"and stock already on order. Flagged above {M['risk']['stockout_threshold']:.0%}.\n"
        f"- **Overstock risk:** the chance that after 8 weeks of forecast sales, stock on hand is still above safety stock. "
        f"Flagged above {M['risk']['overstock_threshold']:.0%}.\n"
        "- **Rupees at stake:** for reorders, the revenue from forecast demand that current stock can't cover in 8 weeks; "
        "for markdowns, the cost value of stock left above the buffer after 8 weeks.")
    st.subheader("Things to know")
    st.markdown(
        f"- Stock figures come from the monthly count on {as_of:%d %b %Y}. Use the what-if panel for a fresher count.\n"
        f"- {E.get('orphan_skus', 'Some')} products in the stock file have no sales or product record, so they aren't scored.\n"
        f"- {E.get('negative_margin_skus', 'Some')} products are priced below their recorded unit cost. Worth confirming with Finance.\n"
        "- Promotions are assumed to follow last year's pattern (weekends in March, June, September and November).")
