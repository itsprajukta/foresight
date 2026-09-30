"""D3 - Demand forecast: seasonal-naive baseline, LightGBM model, rolling-origin backtest,
and the final 8-week forecast with an 80% interval.

Run on its own:  python -m src.forecast   (needs src.pipeline outputs)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from .features import FEATURES, Panel, rows_for_origin, training_rows


# ----------------------------------------------------------------------------- metrics
def wape(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return np.abs(actual - forecast).sum() / np.abs(actual).sum()


def bias(actual, forecast):
    """Signed: positive = over-forecasting, as a share of actual demand."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return (forecast - actual).sum() / np.abs(actual).sum()


def mape(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    m = actual > 0
    return np.mean(np.abs(actual[m] - forecast[m]) / actual[m])


# ---------------------------------------------------------------------------- models
def seasonal_naive(P: Panel, o: int, horizons=range(1, C.HORIZON_WEEKS + 1)):
    """Forecast for week T = demand in the same week one year earlier (T - 52)."""
    return {h: P.Y[:, o + h - C.SEASON_LENGTH] for h in horizons}


def fit_model(train: pd.DataFrame):
    X = train[FEATURES].copy()
    for c in ["lag0", "lag1", "lag2", "lag3", "rm4", "rm8", "rm13", "rm26", "rstd8", "exp_mean", "sn", "sn3", "sn_adj"]:
        X[c] = X[c] / train["scale"]
    y = train["y"] / train["scale"]
    model = lgb.LGBMRegressor(**C.LGBM_PARAMS)
    # weight = scale turns an L1 loss on the ratio into an L1 loss on units, i.e. what WAPE measures
    model.fit(X, y, sample_weight=train["scale"], categorical_feature=["category_code"])
    return model


def predict(model, rows: pd.DataFrame):
    X = rows[FEATURES].copy()
    for c in ["lag0", "lag1", "lag2", "lag3", "rm4", "rm8", "rm13", "rm26", "rstd8", "exp_mean", "sn", "sn3", "sn_adj"]:
        X[c] = X[c] / rows["scale"]
    return np.clip(model.predict(X) * rows["scale"].to_numpy(), 0, None)


# -------------------------------------------------------------------------- backtest
def backtest_origins(last_known: int):
    """Origins whose full 8-week test window ends on or before last_known."""
    return list(range(C.BACKTEST_FIRST_ORIGIN, last_known - C.HORIZON_WEEKS + 1, C.BACKTEST_STEP))


def backtest(P: Panel, last_known: int, verbose=True):
    """Rolling-origin CV: for each origin, train only on weeks <= origin, forecast the next 8 weeks."""
    out = []
    for o in backtest_origins(last_known):
        model = fit_model(training_rows(P, o))
        test = rows_for_origin(P, o)
        test["model"] = predict(model, test)
        sn = seasonal_naive(P, o)
        test["baseline"] = np.concatenate([sn[h] for h in range(1, C.HORIZON_WEEKS + 1)])
        test["fold"] = o
        out.append(test[["fold", "sku_id", "h", "week_start", "y", "baseline", "model"]])
        if verbose:
            print(f"  fold origin={P.week_date(o).date()}  model WAPE={wape(test.y, test.model):.3f}  "
                  f"baseline WAPE={wape(test.y, test.baseline):.3f}")
    return pd.concat(out, ignore_index=True).rename(columns={"y": "actual"})


def summarise_backtest(bt: pd.DataFrame):
    overall = dict(
        model_wape=wape(bt.actual, bt.model), baseline_wape=wape(bt.actual, bt.baseline),
        model_bias=bias(bt.actual, bt.model), baseline_bias=bias(bt.actual, bt.baseline),
        model_mape=mape(bt.actual, bt.model), baseline_mape=mape(bt.actual, bt.baseline),
        n_folds=int(bt.fold.nunique()), n_forecasts=int(len(bt)),
    )
    overall["wape_improvement_pct"] = 100 * (1 - overall["model_wape"] / overall["baseline_wape"])
    per_fold = bt.groupby("fold").apply(lambda g: pd.Series(dict(
        week_from=g.week_start.min(), week_to=g.week_start.max(),
        model_wape=wape(g.actual, g.model), baseline_wape=wape(g.actual, g.baseline))), include_groups=False).reset_index()
    per_h = bt.groupby("h").apply(lambda g: pd.Series(dict(
        model_wape=wape(g.actual, g.model), baseline_wape=wape(g.actual, g.baseline))), include_groups=False).reset_index()
    per_sku = bt.groupby("sku_id").apply(lambda g: pd.Series(dict(
        model_wape=wape(g.actual, g.model), baseline_wape=wape(g.actual, g.baseline))), include_groups=False).reset_index()
    return overall, per_fold, per_h, per_sku


def error_profile(bt: pd.DataFrame):
    """Empirical uncertainty from the backtest (no distributional assumption).

    - interval ratios: 10th / 90th percentile of actual / forecast, per horizon -> 80% band
    - window relative sd: spread of (actual total - forecast total) / forecast total for the first k weeks,
      used by the risk layer to turn a forecast total into a probability.
    """
    b = bt[bt.model > 0].copy()
    b["ratio"] = b.actual / b.model
    band = b.groupby("h")["ratio"].quantile([C.INTERVAL_LOWER_Q, C.INTERVAL_UPPER_Q]).unstack()
    band.columns = ["lo_ratio", "hi_ratio"]
    rel_sd = {}
    for k in range(1, C.HORIZON_WEEKS + 1):
        w = bt[bt.h <= k].groupby(["fold", "sku_id"])[["actual", "model"]].sum()
        rel_sd[k] = float(((w.actual - w.model) / w.model).std())
    return band.reset_index(), rel_sd


# ----------------------------------------------------------------------- final forecast
def final_forecast(P: Panel, last_known: int, band: pd.DataFrame, per_sku: pd.DataFrame):
    """Train on every week up to last_known, forecast the next HORIZON_WEEKS weeks."""
    model = fit_model(training_rows(P, last_known))
    rows = rows_for_origin(P, last_known, with_target=True)
    rows["forecast"] = predict(model, rows)
    # Promotion sensitivity (brief 16.2): the same forecast with promotion days removed.
    no_promo = rows.copy()
    no_promo["promo_days_T"] = 0.0
    rows["forecast_no_promo"] = predict(model, no_promo)
    rows["baseline"] = np.concatenate([seasonal_naive(P, last_known)[h] for h in range(1, C.HORIZON_WEEKS + 1)])
    rows = rows.merge(band, on="h", how="left")
    rows["lower_80"] = rows.forecast * rows.lo_ratio
    rows["upper_80"] = rows.forecast * rows.hi_ratio
    rel = per_sku.set_index("sku_id")
    rows["sku_backtest_wape"] = rows.sku_id.map(rel["model_wape"])
    rows["actual"] = rows.pop("y")
    cols = ["sku_id", "h", "week_start", "forecast", "lower_80", "upper_80", "forecast_no_promo",
            "baseline", "actual", "promo_days_T", "sku_backtest_wape"]
    return rows[cols].rename(columns={"promo_days_T": "planned_promo_days"}), model
