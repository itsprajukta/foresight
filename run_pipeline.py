"""One command, raw extracts -> every output the dashboard, service and reports use.

    python run_pipeline.py

Steps: ingest + clean (D1) -> weekly panel + features -> seasonal-naive baseline and LightGBM model,
rolling-origin backtest (D3) -> final 8-week forecast -> stockout / overstock scoring (D4)
-> EDA figures (D2). Results are deterministic (fixed seeds, single-threaded training).
"""
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from src import config as C
from src import pipeline, eda
from src import forecast as F
from src import risk as R
from src.features import Panel


def main():
    t0 = time.time()
    np.random.seed(C.SEED)
    data = pipeline.run()
    weekly, inv, master = data["weekly"], data["inventory"], data["master"]

    # As-of date = latest stock snapshot. The forecast starts that week so stock and demand line up.
    as_of = inv["date"].max()
    P0 = Panel(weekly)
    last_known = int(np.searchsorted(P0.weeks, as_of) - 1)          # last full week before the snapshot
    future = pipeline.future_calendar(P0.weeks[-1] + pd.Timedelta(weeks=1), C.HORIZON_WEEKS)
    P = Panel(weekly, extra_weeks=future)
    print(f"[forecast] as-of {as_of.date()} | training weeks {P.weeks[0].date()} .. {P.weeks[last_known].date()} "
          f"| horizon {C.HORIZON_WEEKS} weeks")

    print("[backtest] rolling-origin CV")
    bt = F.backtest(P, last_known)
    overall, per_fold, per_h, per_sku = F.summarise_backtest(bt)
    band, rel_sd = F.error_profile(bt)

    fc, _ = F.final_forecast(P, last_known, band, per_sku)
    fc = fc.merge(master[["sku_id", "product_name", "category"]], on="sku_id")

    # Held-out check: the weeks after the as-of date that already have actual sales.
    live = fc.dropna(subset=["actual"])
    holdout = dict(weeks=int(live.week_start.nunique()),
                   model_wape=F.wape(live.actual, live.forecast) if len(live) else None,
                   baseline_wape=F.wape(live.actual, live.baseline) if len(live) else None)

    inv_latest = inv[inv["date"] == as_of]
    risk = R.score_all(fc, inv_latest, master, rel_sd, overall["model_wape"])

    # History for the dashboard's forecast-vs-actual view (backtest forecasts per week, averaged over folds at h)
    hist = weekly[["sku_id", "week_start", "units"]].rename(columns={"units": "actual"})

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bt.to_csv(C.OUTPUT_DIR / "backtest_predictions.csv", index=False)
    per_fold.to_csv(C.OUTPUT_DIR / "backtest_by_fold.csv", index=False)
    per_h.to_csv(C.OUTPUT_DIR / "backtest_by_horizon.csv", index=False)
    per_sku.to_csv(C.OUTPUT_DIR / "backtest_by_sku.csv", index=False)
    fc.to_csv(C.OUTPUT_DIR / "forecast.csv", index=False)
    risk.to_csv(C.OUTPUT_DIR / "risk_scores.csv", index=False)
    hist.to_csv(C.OUTPUT_DIR / "weekly_history.csv", index=False)

    q = risk["quadrant"].value_counts().to_dict()
    metrics = dict(
        as_of=str(as_of.date()), horizon_weeks=C.HORIZON_WEEKS,
        forecast_weeks=[str(fc.week_start.min().date()), str(fc.week_start.max().date())],
        backtest={k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in overall.items()},
        folds_won_by_model=int((per_fold.model_wape < per_fold.baseline_wape).sum()),
        holdout_after_as_of={k: (float(v) if v is not None and not isinstance(v, int) else v) for k, v in holdout.items()},
        interval_ratios=band.to_dict(orient="list"), window_rel_sd={str(k): v for k, v in rel_sd.items()},
        risk=dict(service_level=R.SERVICE_LEVEL, stockout_threshold=R.STOCKOUT_THRESHOLD,
                  overstock_threshold=R.OVERSTOCK_THRESHOLD, quadrant_counts=q,
                  sales_at_risk_inr=float(risk.loc[risk.quadrant.isin(["Reorder now", "Watch / volatile"]), "sales_at_risk_inr"].sum()),
                  capital_locked_inr=float(risk.loc[risk.quadrant.isin(["Markdown / clear", "Watch / volatile"]), "capital_locked_inr"].sum()),
                  total_stock_value_inr=float((inv_latest.on_hand_units * inv_latest.sku_id.map(master.set_index("sku_id").unit_cost)).sum())),
    )
    with open(C.OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    eda.make_figures(data, bt, fc, risk, metrics)

    b = metrics["backtest"]
    print("\n================ HEADLINE ================")
    print(f"Backtest WAPE  model {b['model_wape']:.1%}  vs  seasonal-naive {b['baseline_wape']:.1%}  "
          f"({b['wape_improvement_pct']:.1f}% better; model won {metrics['folds_won_by_model']}/{b['n_folds']} folds)")
    print(f"Bias           model {b['model_bias']:+.1%}  vs  seasonal-naive {b['baseline_bias']:+.1%}")
    if holdout["weeks"]:
        print(f"Hold-out ({holdout['weeks']} wks after as-of)  model {holdout['model_wape']:.1%}  vs  baseline {holdout['baseline_wape']:.1%}")
    print(f"Quadrants      {q}")
    print(f"Sales at risk  Rs {metrics['risk']['sales_at_risk_inr']:,.0f}   Capital locked  Rs {metrics['risk']['capital_locked_inr']:,.0f}")
    print(f"Done in {time.time() - t0:.0f}s -> outputs/, data/processed/, reports/figures/")


if __name__ == "__main__":
    main()
