"""D4 - Stockout / overstock risk scoring and decisioning (Section 08).

Transparent rule, no black box. For every SKU, using the latest stock position:

  Stockout risk  = P( demand during the replenishment lead time  >  on-hand + on-order - safety stock )
                   "chance we eat into safety stock before a new order could arrive"
  Overstock risk = P( demand over the next 8 weeks  <  on-hand - safety stock )
                   "chance we are still sitting on more than our buffer after two months of selling"

Demand totals come from the forecast. Their uncertainty comes from the backtest: the spread of
forecast errors for a window of that length, scaled by how reliable the forecast has been for that SKU.
Probabilities use a normal approximation of that error spread.

Thresholds: stockout is flagged when its risk exceeds 1 - SERVICE_LEVEL (the brief ties the stockout
threshold to the service level); overstock is flagged when it is more likely than not (> 0.5).
Quadrants follow the Section 08 grid.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C

SERVICE_LEVEL = 0.90                      # target probability of NOT eating into safety stock over the lead time
STOCKOUT_THRESHOLD = 1 - SERVICE_LEVEL    # 0.10
OVERSTOCK_THRESHOLD = 0.50

ACTIONS = {
    "Reorder now": "Raise a replenishment order before stock runs out.",
    "Markdown / clear": "Promote or discount to free up capital.",
    "Watch / volatile": "Investigate - demand is erratic; review manually.",
    "Healthy": "No action needed; leave as is.",
}


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def lead_time_demand(weekly_fc, lead_time_days):
    """Sum of forecast daily demand over the lead time (week w's demand spread evenly over 7 days)."""
    days = int(round(lead_time_days))
    total = 0.0
    for d in range(days):
        w = min(d // 7, len(weekly_fc) - 1)
        total += weekly_fc[w] / 7.0
    return total


def score_sku(weekly_fc, on_hand, on_order, lead_time_days, safety_stock, unit_cost, price,
              rel_sd: dict, reliability: float = 1.0):
    """Score one SKU. weekly_fc = forecast units for weeks 1..H. reliability = SKU WAPE / overall WAPE."""
    weekly_fc = [max(float(x), 0.0) for x in weekly_fc]
    H = len(weekly_fc)
    d_lt = lead_time_demand(weekly_fc, lead_time_days)
    d_h = float(sum(weekly_fc))
    k_lt = min(max(1, math.ceil(lead_time_days / 7)), H)
    sd_lt = max(rel_sd[k_lt] * reliability * d_lt, 1e-6)
    sd_h = max(rel_sd[H] * reliability * d_h, 1e-6)

    position = on_hand + on_order
    stockout_risk = _phi((d_lt - (position - safety_stock)) / sd_lt)
    overstock_risk = _phi(((on_hand - safety_stock) - d_h) / sd_h)

    so = stockout_risk > STOCKOUT_THRESHOLD
    ov = overstock_risk > OVERSTOCK_THRESHOLD
    quadrant = ("Watch / volatile" if so and ov else "Reorder now" if so
                else "Markdown / clear" if ov else "Healthy")

    # Rupee value at stake (what the team is protecting / freeing by acting)
    lost_units = max(0.0, d_h - position)                            # 8-week demand that stock in hand + on order cannot serve
    excess_units = max(0.0, on_hand - safety_stock - d_h)            # stock still above buffer after 8 weeks
    sales_at_risk = lost_units * price
    capital_locked = excess_units * unit_cost
    # Units to cover forecast 8-week demand and restore the safety buffer (a recommendation, not an order)
    suggested_order = (math.ceil(max(0.0, d_h + safety_stock - position))
                       if quadrant in ("Reorder now", "Watch / volatile") else 0)
    daily = d_h / (7 * H) if d_h > 0 else 0.0
    days_of_cover = position / daily if daily > 0 else float("inf")

    value_at_stake = {"Reorder now": sales_at_risk, "Markdown / clear": capital_locked,
                      "Watch / volatile": sales_at_risk + capital_locked, "Healthy": 0.0}[quadrant]
    return dict(
        forecast_lead_time_units=round(d_lt, 1), forecast_horizon_units=round(d_h, 1),
        stock_position_units=position, days_of_cover=round(days_of_cover, 1),
        stockout_risk=round(stockout_risk, 4), overstock_risk=round(overstock_risk, 4),
        quadrant=quadrant, recommended_action=ACTIONS[quadrant],
        suggested_order_units=suggested_order,
        sales_at_risk_inr=round(sales_at_risk, 2), capital_locked_inr=round(capital_locked, 2),
        value_at_stake_inr=round(value_at_stake, 2),
    )


def score_all(fc: pd.DataFrame, inv_latest: pd.DataFrame, master: pd.DataFrame, rel_sd: dict,
              overall_wape: float):
    """fc: final forecast (sku_id, h, forecast, sku_backtest_wape). inv_latest: one row per SKU."""
    m = master.set_index("sku_id")
    inv = inv_latest.set_index("sku_id")
    rows = []
    for sku, g in fc.sort_values("h").groupby("sku_id"):
        rel = float(np.clip(g["sku_backtest_wape"].iloc[0] / overall_wape, 0.5, 3.0))
        i = inv.loc[sku]
        r = score_sku(g["forecast"].tolist(), i.on_hand_units, i.on_order_units, i.lead_time_days,
                      i.safety_stock, m.loc[sku, "unit_cost"], m.loc[sku, "list_price"], rel_sd, rel)
        r.update(sku_id=sku, product_name=m.loc[sku, "product_name"], category=m.loc[sku, "category"],
                 on_hand_units=int(i.on_hand_units), on_order_units=int(i.on_order_units),
                 lead_time_days=int(i.lead_time_days), safety_stock=int(i.safety_stock),
                 reorder_point_supplied=int(i.reorder_point), unit_cost=m.loc[sku, "unit_cost"],
                 list_price=m.loc[sku, "list_price"], negative_margin=bool(m.loc[sku, "negative_margin"]),
                 forecast_reliability=rel, stock_as_of=i.date)
        rows.append(r)
    out = pd.DataFrame(rows)
    lead = ["sku_id", "product_name", "category", "quadrant", "recommended_action", "value_at_stake_inr",
            "stockout_risk", "overstock_risk", "suggested_order_units"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    return out.sort_values("value_at_stake_inr", ascending=False).reset_index(drop=True)
