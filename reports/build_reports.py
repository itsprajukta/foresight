"""Builds the written deliverables from the pipeline outputs, so every number in them is reproducible.

    python reports/build_reports.py      (after python run_pipeline.py; needs playwright + Chromium)

Writes:
  reports/EDA_Insight_Memo.pdf     D2 - data-quality & EDA insight memo
  reports/Executive_Readout.pdf    D7 - 8-slide readout for the Head of Operations and Finance
  reports/Project_Report.pdf       project documentation (overview, stack, architecture, screenshots, conclusion)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, REP, FIG = ROOT / "outputs", ROOT / "reports", ROOT / "reports" / "figures"
HTML_DIR = REP / "_html"
INR = "₹"

M = json.loads((OUT / "metrics.json").read_text())
E = json.loads((OUT / "eda_summary.json").read_text())
RISK = pd.read_csv(OUT / "risk_scores.csv")
DQ = pd.read_csv(OUT / "data_quality_log.csv")
FC = pd.read_csv(OUT / "forecast.csv", parse_dates=["week_start"])
B = M["backtest"]
AS_OF = pd.Timestamp(M["as_of"])


def rs(x):
    x = float(x)
    if abs(x) >= 1e7:
        return f"{INR}{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"{INR}{x / 1e5:.1f} L"
    return f"{INR}{x:,.0f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def img(name, cls=""):
    return f'<img class="{cls}" src="../figures/{name}">'


reorder = RISK[RISK.quadrant.isin(["Reorder now", "Watch / volatile"])].sort_values("sales_at_risk_inr", ascending=False)
clear = RISK[RISK.quadrant.isin(["Markdown / clear", "Watch / volatile"])].sort_values("capital_locked_inr", ascending=False)
reorder = reorder.assign(lt_gap=(reorder.forecast_lead_time_units - reorder.stock_position_units).clip(lower=0))
expedite = reorder[reorder.lt_gap > 0]
neg_reorder = reorder[reorder.negative_margin]
q = RISK.quadrant.value_counts()
SALES_AT_RISK = reorder.sales_at_risk_inr.sum()
CAPITAL = clear.capital_locked_inr.sum()
excess_units = (clear.on_hand_units - clear.safety_stock - clear.forecast_horizon_units).clip(lower=0).sum()
fc_rev = (FC.forecast * FC.sku_id.map(RISK.set_index("sku_id").list_price)).sum()
month_idx = E["month_index"]
hi_m = max(month_idx, key=month_idx.get); lo_m = min(month_idx, key=month_idx.get)
lift = E["lift_pct"]
lift_we, lift_pr, lift_ho = lift["Weekend (no promo)"], lift["Promotion weekend vs normal weekend"], lift["Public holiday vs average day"]

BASE_CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --line:#e6e5e1; --soft:#f4f4f2; --blue:#2a78d6;
        --blue-soft:#eaf2fc; --red:#d03b3b; --violet:#7a5cc7; --green:#0ca30c; --amber:#b77900; }
* { box-sizing:border-box; }
body { margin:0; font-family:"Carlito","DejaVu Sans",sans-serif; color:var(--ink); background:#fff; }
h1,h2,h3 { margin:0; line-height:1.15; }
p { margin:0 0 8px; line-height:1.45; }
table { border-collapse:collapse; width:100%; }
th { text-align:left; font-weight:700; color:var(--ink2); border-bottom:1.5px solid var(--ink2); padding:5px 8px; }
td { border-bottom:1px solid var(--line); padding:5px 8px; vertical-align:top; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
.tag { display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.85em; font-weight:700; }
.t-red { background:#fbe9e9; color:var(--red); } .t-vio { background:#efeafa; color:var(--violet); }
.t-grn { background:#e5f5e5; color:#087a08; } .t-amb { background:#fdf3dc; color:var(--amber); }
.muted { color:var(--muted); } .ink2 { color:var(--ink2); }
"""

# =============================================================================== D2 EDA memo
def memo_html():
    issues = DQ[DQ.rows_affected > 0]
    issues = issues[~issues.check.isin(["Rows in / rows out", "Blank holiday / promotion_event cells"])]
    clean_checks = DQ[(DQ.rows_affected == 0)].check.tolist()
    dq_rows = "".join(f"<tr><td>{r.table}</td><td><b>{r.check}</b><br><span class='ink2'>{r.rationale}</span></td>"
                      f"<td class='n'>{r.rows_affected:,}</td><td>{r.action}</td></tr>" for r in issues.itertuples())
    slow = E["slow_stock_skus_over_90d"]
    slow_txt = ", ".join(f"{k} ({v} days)" for k, v in slow.items())
    up = ", ".join(f"{k} {v:+.1f}%" for k, v in list(E["yoy_by_sku_pct"]["top_up"].items())[:3])
    down = ", ".join(f"{k} {v:+.1f}%" for k, v in list(E["yoy_by_sku_pct"]["top_down"].items())[:3])
    cat = E["category_revenue_cr"]; top_cat = max(cat, key=cat.get)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>EDA Insight Memo</title><style>{BASE_CSS}
@page {{ size:A4; }}
body {{ font-size:10.5pt; }}
.head {{ border-bottom:3px solid var(--blue); padding-bottom:10px; margin-bottom:14px; }}
.kicker {{ font-size:8.5pt; letter-spacing:.14em; color:var(--blue); font-weight:700; text-transform:uppercase; }}
h1 {{ font-size:21pt; margin:4px 0 8px; }}
.meta {{ display:grid; grid-template-columns:70px 1fr; gap:2px 10px; font-size:9.5pt; color:var(--ink2); }}
h2 {{ font-size:13.5pt; margin:18px 0 8px; padding-top:4px; }}
h2 .num {{ color:var(--blue); margin-right:6px; }}
.box {{ background:var(--blue-soft); border-left:4px solid var(--blue); padding:10px 14px; margin:8px 0 10px; }}
.box ol {{ margin:4px 0 0 18px; padding:0; }} .box li {{ margin:4px 0; line-height:1.4; }}
img {{ width:100%; margin:4px 0 2px; }} img.half {{ width:49%; }}
.cap {{ font-size:8.5pt; color:var(--muted); margin-bottom:10px; }}
table {{ font-size:9pt; }}
.ins {{ border:1px solid var(--line); border-radius:6px; padding:10px 14px; margin:8px 0; break-inside:avoid; }}
.ins h3 {{ font-size:11pt; margin-bottom:4px; }} .ins .so {{ color:var(--ink2); }}
.pb {{ break-before:page; }}
figure {{ margin:0 0 6px; break-inside:avoid; }} figure img {{ width:92%; display:block; margin:4px auto 2px; }}
h2, h3 {{ break-after:avoid; }}
</style></head><body>
<div class="head"><div class="kicker">Project FORESIGHT · Deliverable D2</div>
<h1>Data-quality &amp; EDA insight memo</h1>
<div class="meta"><b>To</b><span>Head of Operations, Merchandiser, Finance lead - NorthBay Living</span>
<b>From</b><span>FORESIGHT data scientist (Zidio Development)</span>
<b>Data</b><span>Daily sales 1 Jan 2024 - 31 Dec 2025 · 50 SKUs · monthly stock snapshots to {AS_OF:%d %b %Y}</span></div></div>

<div class="box"><b>In one minute</b><ol>
<li><b>Demand is seasonal and steady.</b> Every year peaks in {hi_m} ({month_idx[hi_m] - 100:+.0f}% vs an average month) and bottoms out in {lo_m} ({month_idx[lo_m] - 100:+.0f}%). Year-on-year volume is flat ({E['yoy_weekly_avg_change_pct']:+.1f}%).</li>
<li><b>Stock is not where demand is.</b> {q.get('Reorder now', 0)} SKUs are about to run short while {len(slow)} SKUs hold more than 90 days of stock ({rs(E['slow_stock_value_inr'])} at cost).</li>
<li><b>Promotions and weekends are big, predictable levers.</b> Promotion weekends sell {lift_pr:.0f}% more than normal weekends, which already sell {lift_we:.0f}% more than weekdays.</li>
<li><b>{E['negative_margin_skus']} SKUs are priced below their recorded cost</b> ({E['negative_margin_revenue_share_pct']:.0f}% of revenue). Finance should confirm the cost data.</li>
<li><b>The extracts need fixes at source.</b> There are 150 stock-only SKUs, launch dates that come after first sales, mismatched subcategories, and stock values that don't match cost. All are handled in code and listed in section 1.</li>
</ol></div>

<h2><span class="num">1</span>Data quality: what we found and how we handled it</h2>
<p>All four extracts are ingested and cleaned by <code>src/pipeline.py</code>. No step is done by hand, and a re-run gives the same result. The pipeline checked for the missing values, duplicates and inconsistent labels the brief warned about. These checks found nothing: <span class="ink2">{'; '.join(clean_checks)}</span>. The problems it did find:</p>
<table><tr><th>Table</th><th>Issue and why it matters</th><th class="n">Rows</th><th>What we did</th></tr>{dq_rows}</table>
<p class="cap">Full log with rationale: outputs/data_quality_log.csv. Two raw files arrived as Excel despite a .csv name. The loader reads them by content.</p>

<h2><span class="num">2</span>Demand patterns</h2>
<h3>Seasonality and trend</h3>
<figure>{img('01_weekly_demand_by_year.png')}<p class="cap">Total units per week, all 50 SKUs. The 2025 line sits almost on top of 2024. Weekly totals range from {E['weekly_units_min_max'][0]:,.0f} to {E['weekly_units_min_max'][1]:,.0f} units.</p></figure>
<figure>{img('02_monthly_seasonality.png')}<p class="cap">Average daily units per SKU in each month, indexed to 100. Spring (Feb-Jun) runs above average and Aug-Dec below.</p></figure>
<h3>What moves demand day to day</h3>
<figure>{img('03_demand_drivers.png')}<p class="cap">Promotions run store-wide on weekends in March, June, September and November. They are compared with normal weekends so the weekend effect isn't counted twice. Holidays (Republic Day, Independence Day, Diwali, Christmas) are compared with an average day.</p></figure>
<h3>Categories, best-sellers and top movers</h3>
<figure>{img('04_category_revenue.png')}<p class="cap">{top_cat} is the largest category ({INR}{cat[top_cat]:.1f} Cr over two years). Revenue is concentrated: the top 10 SKUs bring in {E['top10_revenue_share_pct']:.0f}% of revenue, and {E['skus_for_80pct_revenue']} SKUs make up 80%.</p></figure>
<figure>{img('05_top_movers_yoy.png')}<p class="cap">Biggest risers: {up}. Biggest fallers: {down}. No SKU moved more than {max(abs(E['yoy_by_sku_pct']['min']), abs(E['yoy_by_sku_pct']['max'])):.0f}%. Price and volume are barely related across SKUs (correlation {E['price_volume_corr']:.2f}).</p></figure>
<h3>Dead and slow stock</h3>
<figure>{img('06_days_of_cover.png')}<p class="cap">Median SKU holds {E['median_days_cover']:.0f} days of stock. Over 90 days: {slow_txt}. Beyond these, {E['orphan_skus']} SKUs (SKU051-SKU200) hold {E['orphan_units_on_hand']:,} units but have no sales at all. They may be dead stock or a separate catalogue.</p></figure>

<h2 class="pb"><span class="num">3</span>Business insights</h2>
<div class="ins"><h3>1 · Plan by season, not by last month</h3>
<p>Demand repeats the same yearly shape: {hi_m} is {month_idx[hi_m] - month_idx[lo_m]:.0f} index points above {lo_m}, and year-on-year growth is flat. Ordering by "what we sold last month" leaves the team short going into spring and overstocked going into autumn.</p>
<p class="so"><b>So what:</b> build stock from February for the Mar-Jun peak, and let it run down from July.</p></div>
<div class="ins"><h3>2 · {len(reorder)} SKUs need stock now while {len(slow)} hold more than 90 days of it</h3>
<p>At the {AS_OF:%d %b} snapshot, {len(reorder)} SKUs are flagged to reorder. {len(expedite)} of them ({', '.join(expedite.sku_id)}) have less stock, on hand and on order, than they will sell before a new order can arrive. At the other end, {', '.join(list(slow)[:2])} hold {list(slow.values())[0]} and {list(slow.values())[1]} days of stock. The problem is stock allocation, not total stock.</p>
<p class="so"><b>So what:</b> use the FORESIGHT reorder and markdown lists each month rather than a blanket reorder.</p></div>
<div class="ins"><h3>3 · Promotions are worth planning stock around</h3>
<p>A promotion weekend adds {lift_pr:.0f}% on top of an already-strong weekend ({lift_we:+.0f}% vs weekdays). Holidays depress sales ({lift_ho:+.0f}%). Promotions fall on the same weekends every year.</p>
<p class="so"><b>So what:</b> stock up best-sellers before the March, June, September and November promotion weekends. The forecast already includes planned promotions.</p></div>
<div class="ins"><h3>4 · {E['negative_margin_skus']} SKUs lose money on every sale, if the cost data is right</h3>
<p>For these SKUs the product master records a unit cost above the selling price. They bring in {E['negative_margin_revenue_share_pct']:.0f}% of revenue. At the recorded costs they would have lost {rs(E['negative_margin_gross_loss_inr'])} in gross margin over two years. {len(neg_reorder)} of them ({', '.join(neg_reorder.sku_id)}) are on this month's reorder list.</p>
<p class="so"><b>So what:</b> Finance should confirm these costs before those SKUs are reordered. Pricing decisions are outside this engagement.</p></div>
<div class="ins"><h3>5 · The stock file needs a clean-up at source</h3>
<p>Month-to-month stock levels don't follow from sales and deliveries: only {pct(E['stock_rollforward_within_20pct_share'], 0)} of monthly moves are within 20% of what sales and on-order stock would predict. The stock value column doesn't equal stock × cost, and 150 SKUs appear only in the stock file.</p>
<p class="so"><b>So what:</b> FORESIGHT scores risk from the latest snapshot only and recalculates values from unit cost. A reliable weekly stock export would make the risk flags sharper.</p></div>

<h2><span class="num">4</span>What this means for the forecast</h2>
<p>Strong yearly seasonality makes "same week last year" a demanding baseline to beat. Weekends, promotions, holidays and recent level are the obvious features to add. The forecast is weekly by SKU, 8 weeks ahead, and is scored on WAPE (average miss as a share of actual demand). It beats that baseline in every backtest window: {pct(B['model_wape'])} vs {pct(B['baseline_wape'])} (see the executive readout and README).</p>
</body></html>"""


# ============================================================================ D7 exec readout
def readout_html():
    def rrow(r):
        flag = " <span class='tag t-amb'>below cost</span>" if r.negative_margin else ""
        exp = " <span class='tag t-red'>expedite</span>" if r.lt_gap > 0 else ""
        return (f"<tr><td><b>{r.sku_id}</b>{exp}{flag}<br><span class='muted'>{r.category}</span></td>"
                f"<td class='n'>{r.days_of_cover:.0f}</td><td class='n'>{r.lead_time_days}</td>"
                f"<td class='n'>{r.suggested_order_units:,}</td><td class='n'><b>{rs(r.sales_at_risk_inr)}</b></td></tr>")
    rrows = "".join(rrow(r) for r in reorder.itertuples())
    BC = " <span class='tag t-amb'>below cost</span>"
    crow = "".join(f"<tr><td><b>{r.sku_id}</b>{BC if r.negative_margin else ''}<br><span class='muted'>{r.category}</span></td>"
                   f"<td class='n'>{r.on_hand_units:,}</td><td class='n'>{r.forecast_horizon_units:,.0f}</td>"
                   f"<td class='n'>{r.days_of_cover:,.0f}</td><td class='n'><b>{rs(r.capital_locked_inr)}</b></td></tr>" for r in clear.itertuples())
    ho = M["holdout_after_as_of"]
    slides = []
    slides.append(f"""<section class="s title"><div class="kicker">Project FORESIGHT · Executive readout · stock as of {AS_OF:%d %b %Y}</div>
<h1>Reorder {len(reorder)} products now to protect <span class="hl">{rs(SALES_AT_RISK)}</span> of sales.<br>
Clear {len(clear)} to free <span class="hl2">{rs(CAPITAL)}</span> of cash.</h1>
<div class="tiles">
<div class="tile"><div class="v">{rs(SALES_AT_RISK)}</div><div class="l">sales at risk over the next 8 weeks if these {len(reorder)} SKUs are not reordered</div></div>
<div class="tile"><div class="v">{rs(CAPITAL)}</div><div class="l">cash locked in stock (at cost) that {len(clear)} SKUs won't sell in 8 weeks</div></div>
<div class="tile"><div class="v">{len(expedite)}</div><div class="l">SKUs that run out before a normal order could arrive. Expedite these.</div></div>
<div class="tile"><div class="v">{pct(B['model_wape'])}</div><div class="l">average forecast miss, vs {pct(B['baseline_wape'])} for "same as last year"</div></div></div>
<p class="foot">For: Head of Operations · Finance lead · Merchandiser &nbsp;|&nbsp; 50 active SKUs · forecast {pd.Timestamp(M['forecast_weeks'][0]):%d %b} to {pd.Timestamp(M['forecast_weeks'][1]) + pd.Timedelta(days=6):%d %b %Y}</p></section>""")

    slides.append(f"""<section class="s"><h2>Action 1 · Reorder these {len(reorder)} SKUs this week</h2>
<p class="sub">Highest sales at risk first. <span class='tag t-red'>expedite</span> means stock runs out before a normal-lead-time order lands.</p>
<table class="big"><tr><th>SKU</th><th class="n">Days of stock left</th><th class="n">Supplier lead time (days)</th><th class="n">Suggested order (units)</th><th class="n">Sales at risk</th></tr>{rrows}</table>
<p class="note">Suggested order = forecast demand for the next 8 weeks plus safety stock, minus stock in hand and on order. It is a recommendation; FORESIGHT does not place orders.
{f"<b>{', '.join(neg_reorder.sku_id)}</b> are priced below recorded cost. Confirm the cost with Finance before ordering, because each extra unit sold may lose money." if len(neg_reorder) else ''}</p></section>""")

    slides.append(f"""<section class="s"><h2>Action 2 · Mark down or pause buying on these {len(clear)} SKUs</h2>
<p class="sub">Their stock on hand is well above what they will sell in 8 weeks plus safety stock.</p>
<table class="big"><tr><th>SKU</th><th class="n">On hand (units)</th><th class="n">Forecast sales, 8 weeks</th><th class="n">Days of stock</th><th class="n">Cash locked (at cost)</th></tr>{crow}</table>
<p class="note">{excess_units:,.0f} excess units in total. Suggested options: a targeted promotion (promotion weekends lift sales {lift_pr:.0f}%), pausing open orders, or clearance. The other {q.get('Healthy', 0)} SKUs need no action today. {('No SKU is' if q.get('Watch / volatile', 0) == 0 else str(q.get('Watch / volatile')) + ' SKUs are')} flagged for both stockout and overstock ("watch / volatile") this month.</p></section>""")

    slides.append(f"""<section class="s two"><div><h2>Where the money is</h2>
<table class="big kv">
<tr><td>Forecast revenue, next 8 weeks (all SKUs)</td><td class="n"><b>{rs(fc_rev)}</b></td></tr>
<tr><td>Sales at risk: the {len(reorder)} SKUs to reorder</td><td class="n"><b style="color:var(--red)">{rs(SALES_AT_RISK)}</b></td></tr>
<tr><td>&nbsp;&nbsp;share of 8-week revenue</td><td class="n">{pct(SALES_AT_RISK / fc_rev)}</td></tr>
<tr><td>Cash locked in excess stock: {len(clear)} SKUs</td><td class="n"><b style="color:var(--violet)">{rs(CAPITAL)}</b></td></tr>
<tr><td>Total stock on hand at cost (50 SKUs)</td><td class="n">{rs(M['risk']['total_stock_value_inr'])}</td></tr>
<tr><td>Revenue from SKUs priced below cost (2 yrs)</td><td class="n">{E['negative_margin_revenue_share_pct']:.0f}%</td></tr>
</table>
<p class="note">Sales at risk is revenue at selling price. Cash locked is stock at unit cost. Both come from the forecast and the {AS_OF:%d %b} stock snapshot.</p></div>
<div>{img('09_decision_grid.png')}<p class="cap">Each bubble is a SKU, placed by its two risk scores. Bubble size is the rupee value at stake.</p></div></section>""")

    slides.append(f"""<section class="s two"><div><h2>How demand behaves</h2>
<ul class="pts"><li><b>The same shape every year.</b> {hi_m} is the peak ({month_idx[hi_m] - 100:+.0f}%) and {lo_m} the low ({month_idx[lo_m] - 100:+.0f}%). Year-on-year volume is flat.</li>
<li><b>Weekends sell {lift_we:.0f}% more</b>, and promotion weekends a further {lift_pr:.0f}%. Promotions run in Mar, Jun, Sep and Nov.</li>
<li><b>Holidays dip {abs(lift_ho):.0f}%.</b></li>
<li><b>The top 10 SKUs bring in {E['top10_revenue_share_pct']:.0f}% of revenue.</b></li></ul></div>
<div>{img('01_weekly_demand_by_year.png')}{img('03_demand_drivers.png')}</div></section>""")

    slides.append(f"""<section class="s two"><div><h2>Can we trust the forecast?</h2>
<ul class="pts"><li>We tested it the honest way. We pretended it was an earlier date, forecast 8 weeks using only data up to then, and compared with what actually sold. We did this {B['n_folds']} times across 2025.</li>
<li><b>It misses by {pct(B['model_wape'])} on average</b> (the chart calls this WAPE: total units missed ÷ units sold). The simple rule "sell the same as the same week last year" (the seasonal-naive baseline) misses by {pct(B['baseline_wape'])}. That is <b>{B['wape_improvement_pct']:.0f}% more accurate</b>, and it won all {M['folds_won_by_model']} test windows.</li>
<li><b>Live check:</b> over the {ho['weeks']} weeks after the stock date, it missed by {pct(ho['model_wape'])} against {pct(ho['baseline_wape'])}.</li>
<li>It is not systematically high or low ({B['model_bias']:+.1%}). Each forecast comes with a likely range: 8 weeks in 10, actual sales should fall inside it.</li></ul></div>
<div>{img('08_backtest_wape_by_fold.png')}{img('07_forecast_example.png')}</div></section>""")

    slides.append(f"""<section class="s"><h2>Limits to know before acting</h2>
<div class="cols3">
<div class="card"><h3>Stock data is monthly and unreliable</h3><p>Stock levels don't follow from sales month to month, and the supplied stock values don't match cost. We use the latest snapshot ({AS_OF:%d %b}) as-is and recalculate values from unit cost. Check stock counts for flagged SKUs before ordering.</p></div>
<div class="card"><h3>Costs look wrong for {E['negative_margin_skus']} SKUs</h3><p>The recorded cost is above the selling price. If that's true, these SKUs lose money on every sale. If it's a data error, the "cash locked" figure for them is overstated. Finance to confirm.</p></div>
<div class="card"><h3>150 SKUs could not be scored</h3><p>SKU051-SKU200 hold {E['orphan_units_on_hand']:,} units but have no sales or product record. They may be dead stock, or they may belong to another catalogue.</p></div>
<div class="card"><h3>Forecast accuracy has a ceiling</h3><p>A {pct(B['model_wape'])} average miss suits planning, not exact counts. Individual SKU-weeks can miss by more, which is why the risk flags include a safety margin.</p></div>
<div class="card"><h3>Promotions are assumed to repeat</h3><p>The Jan 2026 calendar was not supplied. We assume promotions keep last year's pattern. The forecast is also available without promotions.</p></div>
<div class="card"><h3>This is a planning aid</h3><p>FORESIGHT recommends; it doesn't place orders or set prices. Suggested order sizes don't include supplier minimums or pack sizes.</p></div>
</div></section>""")

    slides.append(f"""<section class="s"><h2>Recommended next steps</h2>
<ol class="steps">
<li><b>This week:</b> raise orders for the {len(reorder)} reorder SKUs and ask suppliers to expedite {', '.join(expedite.sku_id)}. Hold the below-cost SKUs until Finance confirms their cost.</li>
<li><b>This month:</b> plan markdowns or pause buying for {', '.join(clear.sku_id)}, freeing up to {rs(CAPITAL)}.</li>
<li><b>Monthly:</b> drop the new sales and stock extracts into <code>data/raw</code> and run the pipeline (one command). The dashboard and scoring service update from the new outputs.</li>
<li><b>Data fixes at source:</b> correct unit costs, launch dates and subcategories; explain the 150 stock-only SKUs; send stock weekly if possible.</li>
<li><b>Before spring:</b> build stock for the Mar-Jun peak, starting with the top-10 revenue SKUs.</li>
</ol>
<p class="foot">Dashboard: filter by category or SKU for forecast vs actual, risk flags and the prioritised reorder and markdown lists. Scoring service: forecast and risk for any SKU or batch.</p></section>""")

    css = BASE_CSS + """
@page { size:13.333in 7.5in; margin:0; }
body { font-size:15pt; }
.s { width:13.333in; height:7.5in; padding:0.55in 0.7in; position:relative; break-after:page; overflow:hidden; }
.s h2 { font-size:28pt; margin-bottom:10px; }
.sub { color:var(--ink2); font-size:14pt; margin-bottom:14px; }
.title { background:#10142a; color:#fff; }
.title .kicker { color:#9db8f0; letter-spacing:.12em; font-size:11pt; text-transform:uppercase; font-weight:700; }
.title h1 { font-size:36pt; margin:22px 0 34px; line-height:1.2; }
.hl { color:#ff8a80; } .hl2 { color:#c7b8ff; }
.tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
.tile { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14); border-radius:10px; padding:16px 18px; }
.tile .v { font-size:28pt; font-weight:700; } .tile .l { font-size:12pt; color:#cfd6ea; margin-top:6px; line-height:1.35; }
.foot { position:absolute; bottom:0.45in; left:0.7in; right:0.7in; font-size:11pt; color:var(--muted); }
.title .foot { color:#9aa3bf; }
table.big { font-size:13.5pt; } table.big td, table.big th { padding:7px 10px; }
table.kv td { padding:9px 6px; }
.note { font-size:12pt; color:var(--ink2); margin-top:14px; line-height:1.45; }
.two { display:grid; grid-template-columns:1fr 1.15fr; gap:34px; align-items:start; }
.two img { width:100%; margin-bottom:6px; }
.cap { font-size:10.5pt; color:var(--muted); }
.pts { padding-left:20px; margin:8px 0; } .pts li { margin:10px 0; line-height:1.4; font-size:14.5pt; }
.cols3 { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:12px; }
.card { border:1px solid var(--line); border-radius:8px; padding:14px 16px; background:var(--soft); }
.card h3 { font-size:14pt; margin-bottom:6px; } .card p { font-size:12pt; color:var(--ink2); margin:0; }
.steps { padding-left:26px; } .steps li { margin:13px 0; font-size:15.5pt; line-height:1.4; }
code { font-size:.9em; background:var(--soft); padding:1px 5px; border-radius:4px; }
"""
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Executive Readout</title><style>{css}</style></head><body>{''.join(slides)}</body></html>"


# ============================================================================ project report
def report_html():
    ho = M["holdout_after_as_of"]
    stack = [("Language & analysis", "Python 3.11, pandas, NumPy, Jupyter"), ("Modelling", "LightGBM (via its scikit-learn interface)"),
             ("Visualisation", "matplotlib (reports), Plotly (dashboard)"), ("Dashboard", "Streamlit"),
             ("Scoring service", "FastAPI + Uvicorn, Pydantic validation"), ("Reports", "HTML rendered to PDF with Chromium (Playwright)"),
             ("Testing", "pytest (leakage, risk rules, API error handling)"),
             ("Deployment", "Streamlit Community Cloud (dashboard), Render (API, render.yaml blueprint)"), ("Version control", "Git")]
    stack_rows = "".join(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in stack)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Project Report</title><style>{BASE_CSS}
@page {{ size:A4; }}
body {{ font-size:10.5pt; }}
.cover {{ height:245mm; display:flex; flex-direction:column; justify-content:center; break-after:page; }}
.cover .k {{ color:var(--blue); letter-spacing:.14em; font-weight:700; font-size:10pt; text-transform:uppercase; }}
.cover h1 {{ font-size:34pt; margin:10px 0; }} .cover h2 {{ font-size:16pt; color:var(--ink2); font-weight:400; }}
.cover .m {{ margin-top:40px; font-size:11pt; color:var(--ink2); line-height:1.7; }}
h2.sec {{ font-size:15pt; margin:18px 0 8px; border-bottom:2px solid var(--blue); padding-bottom:4px; }}
h3 {{ font-size:11.5pt; margin:12px 0 6px; }}
img {{ width:100%; margin:4px 0; border:1px solid var(--line); }} img.nb {{ border:none; }}
.cap {{ font-size:8.5pt; color:var(--muted); margin-bottom:10px; }}
ul {{ margin:4px 0 8px 18px; padding:0; }} li {{ margin:3px 0; line-height:1.4; }}
.arch {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:10px 0; }}
.layer {{ border:1px solid var(--line); border-radius:6px; padding:8px 10px; background:var(--soft); font-size:9pt; }}
.layer b {{ display:block; font-size:10pt; margin-bottom:4px; color:var(--blue); }}
.arrow {{ text-align:center; color:var(--muted); font-size:9pt; }}
table {{ font-size:9.5pt; margin:6px 0 10px; }}
.pb {{ break-before:page; }}
figure {{ margin:0; break-inside:avoid; }} figure img.nb {{ width:88%; display:block; margin:4px auto; }}
h2, h3 {{ break-after:avoid; }}
code {{ font-size:9pt; background:var(--soft); padding:1px 4px; border-radius:3px; }}
</style></head><body>
<div class="cover"><div class="k">Zidio Development · Data Science &amp; Analytics</div>
<h1>Project FORESIGHT</h1><h2>AI-powered demand &amp; inventory intelligence for NorthBay Living</h2>
<div class="m"><b>Project report</b><br>Client: NorthBay Living (D2C home &amp; lifestyle)<br>Role: Data Scientist<br>
Stack: Python · pandas · LightGBM · Streamlit · FastAPI<br>Data: Jan 2024 - Dec 2025, 50 SKUs, stock as of {AS_OF:%d %b %Y}</div></div>

<h2 class="sec">1. Overview</h2>
<p>NorthBay Living plans inventory on gut feel and spreadsheets. It loses money both ways: best-sellers run out and slow movers pile up. The client asked for four things: a demand forecast, a stockout early warning, an overstock flag, and something a non-technical team can use. FORESIGHT delivers all four.</p>
<ul><li>A weekly, SKU-level demand forecast for the next <b>8 weeks</b> with an 80% likely range. It beats a seasonal-naive baseline on a rolling-origin backtest.</li>
<li>Stockout and overstock risk for every SKU, with a recommended action and the rupee value at stake.</li>
<li>A Streamlit planning dashboard and a FastAPI scoring service, both fed by one reproducible pipeline.</li>
<li>A data-quality &amp; EDA memo and an executive readout for the Head of Operations and Finance.</li></ul>
<table><tr><th>Headline result</th><th class="n">Value</th></tr>
<tr><td>Backtest WAPE: FORESIGHT model</td><td class="n"><b>{pct(B['model_wape'])}</b></td></tr>
<tr><td>Backtest WAPE: seasonal-naive baseline</td><td class="n">{pct(B['baseline_wape'])}</td></tr>
<tr><td>Improvement over baseline · folds won</td><td class="n">{B['wape_improvement_pct']:.1f}% · {M['folds_won_by_model']}/{B['n_folds']}</td></tr>
<tr><td>Hold-out after stock date ({ho['weeks']} weeks): model vs baseline</td><td class="n">{pct(ho['model_wape'])} vs {pct(ho['baseline_wape'])}</td></tr>
<tr><td>Bias (model)</td><td class="n">{B['model_bias']:+.1%}</td></tr>
<tr><td>SKUs to reorder now · sales at risk</td><td class="n">{len(reorder)} · {rs(SALES_AT_RISK)}</td></tr>
<tr><td>SKUs to mark down / clear · cash locked</td><td class="n">{len(clear)} · {rs(CAPITAL)}</td></tr></table>

<h2 class="sec">2. Data</h2>
<p>Four extracts from the client: <b>sales_daily</b> (36,550 rows, one per SKU per day), <b>sku_master</b> (50 SKUs), <b>calendar</b> (731 days, holidays and promotion events) and <b>inventory_snapshots</b> (monthly, 200 SKUs). The main data issues, each handled in code and logged: an unlabelled cost column (<code>wfr</code>, verified as unit cost), 150 stock-only SKUs, launch dates after first sales, mismatched subcategories, stock values that don't equal stock × cost, snapshots that don't reconcile with sales, and 16 SKUs priced below cost. Full detail is in the EDA memo and <code>outputs/data_quality_log.csv</code>.</p>

<h2 class="sec">3. Tech stack</h2>
<table><tr><th>Area</th><th>Tools</th></tr>{stack_rows}</table>

<h2 class="sec pb">4. Architecture</h2>
<div class="arch">
<div class="layer"><b>1 · Data sources</b>sales_daily.csv<br>inventory_snapshots.csv<br>sku_master (.xlsx)<br>calendar (.xlsx)</div>
<div class="layer"><b>2 · Pipeline</b><code>src/pipeline.py</code><br>ingest → validate → clean → unify → weekly panel<br><code>src/features.py</code> leakage-safe features</div>
<div class="layer"><b>3 · Models</b><code>src/forecast.py</code> seasonal-naive + LightGBM, rolling-origin backtest, 80% range<br><code>src/risk.py</code> stockout / overstock rule</div>
<div class="layer"><b>4 · Serving</b><code>app/dashboard.py</code> Streamlit<br><code>service/main.py</code> FastAPI<br><code>reports/</code> memo, readout</div></div>
<p class="arrow">run_pipeline.py runs layers 1 → 3 with one command and writes <code>outputs/</code>, which layer 4 reads</p>
<h3>Repository layout</h3>
<table><tr><td><code>data/raw</code> · <code>data/processed</code></td><td>client extracts as delivered · cleaned, analysis-ready tables</td></tr>
<tr><td><code>src/</code></td><td>config, pipeline, features, forecast, risk, eda</td></tr>
<tr><td><code>notebooks/</code></td><td>01_eda, 02_baseline, 03_model (executed)</td></tr>
<tr><td><code>app/</code> · <code>service/</code></td><td>Streamlit dashboard · FastAPI scoring service</td></tr>
<tr><td><code>outputs/</code></td><td>forecast, risk scores, backtest results, metrics, data-quality log</td></tr>
<tr><td><code>reports/</code> · <code>tests/</code></td><td>PDF deliverables and figures · pytest checks</td></tr></table>

<h2 class="sec">5. Methodology</h2>
<h3>Frame and baseline</h3>
<p>The target is weekly units per SKU (Monday-Sunday) over an 8-week horizon. The primary metric is WAPE, with bias as a secondary check. The baseline is seasonal-naive: each week is forecast as the same week last year.</p>
<h3>Features and model</h3>
<p>One global LightGBM model covers all SKUs and predicts each horizon directly (one row per SKU × origin × horizon). Features: recent lags, 4/8/13/26-week rolling means, volatility, same-week-last-year (raw, smoothed and adjusted for level), planned promotion and holiday days in the target week, week of year, month, category and price. Demand features are scaled by each SKU's recent level. The model is trained with an absolute-error loss weighted by that level, so it minimises the same quantity WAPE measures.</p>
<h3>Backtest and leakage control</h3>
<p>The backtest uses rolling origins: {B['n_folds']} origins, four weeks apart, from Jan to Sep 2025. Each fold trains only on weeks up to its origin and forecasts the next 8. Features use only data up to the origin; only calendar facts known in advance describe the target week. A unit test scrambles all future demand and confirms that no feature changes. As an extra check, the 4 weeks after the stock date were never used for training or model choice.</p>
<h3>Risk scoring</h3>
<p><b>Stockout risk</b> is the probability that lead-time demand exceeds on-hand + on-order stock minus safety stock. It is flagged above 10%, which corresponds to a 90% service level. <b>Overstock risk</b> is the probability that 8-week demand is below on-hand stock minus safety stock, flagged above 50%. Demand uncertainty comes from the empirical backtest error for each window length, scaled by the SKU's own forecast reliability. The two flags map to the four quadrants of the Section 08 grid. Each SKU also gets a value at stake: sales at risk (revenue the stock cannot serve) or cash locked (excess stock at cost).</p>

<h2 class="sec pb">6. Results</h2>
<figure><img class="nb" src="../figures/08_backtest_wape_by_fold.png"><p class="cap">WAPE for each backtest window: the model beats the baseline in every one.</p></figure>
<figure><img class="nb" src="../figures/07_forecast_example.png"><p class="cap">Forecast output for the highest-value SKU: actual history, baseline, forecast and 80% range.</p></figure>
<figure><img class="nb" src="../figures/09_decision_grid.png" style="width:78%"><p class="cap">Decision grid: {q.get('Reorder now', 0)} reorder now, {q.get('Markdown / clear', 0)} markdown / clear, {q.get('Watch / volatile', 0)} watch / volatile, {q.get('Healthy', 0)} healthy.</p></figure>

<h2 class="sec">7. Dashboard and scoring service</h2>
<p>The dashboard gives the operations team four views: <b>action lists</b> (prioritised reorder and markdown tables with rupee values), <b>forecast vs actual</b> per SKU (history, past backtest forecasts, the 8-week forecast with its range, and a live check against weeks that have already happened), the <b>risk grid</b>, and a plain-language <b>how to read this</b> page. Filters for category, status and SKU sit in the sidebar. Empty filters show a clear message, and a missing pipeline run shows setup instructions.</p>
<figure><img src="../figures/dash_1_actions.png"><p class="cap">Action lists with headline KPIs.</p></figure>
<figure><img src="../figures/dash_2_forecast.png"><p class="cap">Forecast vs actual for one SKU.</p></figure>
<figure><img src="../figures/dash_3_grid.png"><p class="cap">Risk grid (interactive, with a tooltip per SKU).</p></figure>
<h3>Scoring service (FastAPI)</h3>
<table><tr><th>Endpoint</th><th>Returns</th></tr>
<tr><td><code>GET /health</code></td><td>status and number of SKUs loaded</td></tr>
<tr><td><code>GET /skus</code></td><td>valid SKU ids</td></tr>
<tr><td><code>GET /score/{{sku_id}}</code></td><td>8-week forecast with range, plus risk, action and rupees at stake</td></tr>
<tr><td><code>POST /score</code></td><td>the same for a batch of up to 200; optional fresher stock inputs per SKU are re-scored with the same rule</td></tr></table>
<p>Bad input is handled without crashing: an unknown SKU returns 404 (or a per-item error inside a batch), and a negative or non-numeric value returns 422 with a readable message. Interactive documentation is served at <code>/docs</code>.</p>

<h2 class="sec">8. Reproducibility and deployment</h2>
<ul><li><code>pip install -r requirements.txt</code> then <code>python run_pipeline.py</code> rebuilds every output from the raw files in about a minute, with fixed seeds and single-threaded training so the numbers are identical on every run.</li>
<li>Dashboard: Streamlit Community Cloud, entry point <code>app/dashboard.py</code>. Scoring service: Render, using <code>render.yaml</code>. Both read the seeded outputs committed in <code>outputs/</code>.</li></ul>

<h2 class="sec">9. Limitations</h2>
<ul><li>Stock snapshots are monthly and don't reconcile with sales, so risk is scored as of the latest snapshot.</li>
<li>Recorded unit costs exceed price for 16 SKUs, which affects cash-locked values until Finance confirms them.</li>
<li>150 stock-only SKUs cannot be forecast.</li>
<li>Promotion days for January 2026 are assumed to follow the historical pattern.</li>
<li>Suggested order sizes ignore supplier minimums and pack sizes.</li></ul>

<h2 class="sec">10. Conclusion</h2>
<p>FORESIGHT turns NorthBay's own extracts into a forecast that is measurably better than the "same as last year" rule ({pct(B['model_wape'])} vs {pct(B['baseline_wape'])} average miss, better in every test window). It also produces a transparent monthly action list. This month that means reordering {len(reorder)} SKUs to protect {rs(SALES_AT_RISK)} of sales and clearing {len(clear)} to free {rs(CAPITAL)} of cash. The pipeline re-runs from raw data with one command, so the team can refresh it every month without a data scientist in the room.</p>
</body></html>"""


async def render(html_name, html, pdf_name, landscape_slides=False):
    from playwright.async_api import async_playwright
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    p = HTML_DIR / html_name
    p.write_text(html, encoding="utf-8")
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        page = await b.new_page()
        await page.goto(p.as_uri())
        await page.wait_for_load_state("networkidle")
        if landscape_slides:
            await page.pdf(path=str(REP / pdf_name), width="13.333in", height="7.5in", print_background=True,
                           prefer_css_page_size=True)
        else:
            await page.pdf(path=str(REP / pdf_name), format="A4", print_background=True, prefer_css_page_size=True,
                           display_header_footer=True, header_template="<span></span>",
                           footer_template="<div style='font-size:7pt;color:#8a8984;width:100%;text-align:center'>"
                                           "Project FORESIGHT · <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
                           margin=dict(top="14mm", bottom="16mm", left="14mm", right="14mm"))
        await b.close()
    print("wrote", REP / pdf_name)


async def main():
    await render("eda_memo.html", memo_html(), "EDA_Insight_Memo.pdf")
    await render("executive_readout.html", readout_html(), "Executive_Readout.pdf", landscape_slides=True)
    await render("project_report.html", report_html(), "Project_Report.pdf")


if __name__ == "__main__":
    asyncio.run(main())
