"""Leakage-safe feature engineering for the direct multi-horizon forecast.

Each training/scoring row is (SKU, origin week o, horizon h) and predicts demand in week T = o + h.
Rule: every demand-derived feature uses weeks <= o only. The only facts about week T that are used
are calendar facts known in advance (planned promotion days, holidays, week of year) and
demand from a year earlier (T - 52 <= o because h <= 8).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

FEATURES = [
    "h", "lag0", "lag1", "lag2", "lag3", "rm4", "rm8", "rm13", "rm26", "rstd8", "exp_mean",
    "sn", "sn3", "sn_adj", "yoy_level", "promo_days_T", "holiday_days_T", "promo_days_T_ly",
    "woy_T", "month_T", "category_code", "log_price",
]


class Panel:
    """Weekly demand as a SKU x week matrix plus week-level calendar arrays."""

    def __init__(self, weekly: pd.DataFrame, extra_weeks: pd.DataFrame | None = None):
        piv = weekly.pivot(index="sku_id", columns="week_start", values="units").sort_index()
        self.skus = piv.index.to_numpy()
        self.weeks = pd.to_datetime(piv.columns)
        self.Y = piv.to_numpy(dtype=float)
        cal = weekly.groupby("week_start")[["promo_days", "holiday_days"]].max().reindex(piv.columns)
        cal.index = pd.to_datetime(cal.index)
        if extra_weeks is not None and len(extra_weeks):
            ex = extra_weeks.set_index(pd.to_datetime(extra_weeks["week_start"]))[["promo_days", "holiday_days"]]
            cal = pd.concat([cal, ex[~ex.index.isin(cal.index)]])
        self.cal_weeks = cal.index
        self.promo = cal["promo_days"].to_numpy(dtype=float)
        self.hol = cal["holiday_days"].to_numpy(dtype=float)
        static = weekly.groupby("sku_id")[["category", "list_price"]].first().reindex(self.skus)
        self.category = static["category"].to_numpy()
        self.cat_codes = pd.Categorical(static["category"]).codes.astype(float)
        self.log_price = np.log(static["list_price"].to_numpy(dtype=float))

    def week_date(self, idx):
        if idx < len(self.cal_weeks):
            return self.cal_weeks[idx]
        return self.cal_weeks[-1] + pd.Timedelta(weeks=idx - len(self.cal_weeks) + 1)


def _mean(Y, a, b):
    a = max(a, 0)
    return Y[:, a:b + 1].mean(axis=1) if b >= a else np.full(Y.shape[0], np.nan)


def rows_for_origin(P: Panel, o: int, horizons=range(1, C.HORIZON_WEEKS + 1), with_target=True):
    """Feature rows for every SKU at origin o (last observed week index) for each horizon."""
    Y = P.Y
    n = Y.shape[0]
    base = dict(
        lag0=Y[:, o], lag1=Y[:, o - 1], lag2=Y[:, o - 2], lag3=Y[:, o - 3],
        rm4=_mean(Y, o - 3, o), rm8=_mean(Y, o - 7, o), rm13=_mean(Y, o - 12, o), rm26=_mean(Y, o - 25, o),
        rstd8=Y[:, max(o - 7, 0):o + 1].std(axis=1), exp_mean=_mean(Y, 0, o),
    )
    ly_rm8 = _mean(Y, o - 59, o - 52) if o - 59 >= 0 else np.full(n, np.nan)
    yoy = base["rm8"] / ly_rm8
    out = []
    for h in horizons:
        T = o + h
        sn = Y[:, T - C.SEASON_LENGTH] if T - C.SEASON_LENGTH >= 0 else np.full(n, np.nan)
        sn3 = _mean(Y, T - 53, T - 51) if T - 53 >= 0 else np.full(n, np.nan)
        wd = P.week_date(T)
        r = dict(base)
        r.update(
            h=np.full(n, h), sn=sn, sn3=sn3, sn_adj=sn * yoy, yoy_level=yoy,
            promo_days_T=np.full(n, P.promo[T] if T < len(P.promo) else 0.0),
            holiday_days_T=np.full(n, P.hol[T] if T < len(P.hol) else 0.0),
            promo_days_T_ly=np.full(n, P.promo[T - 52] if T - 52 >= 0 else np.nan),
            woy_T=np.full(n, (wd.dayofyear - 1) // 7 + 1), month_T=np.full(n, wd.month),
            category_code=P.cat_codes, log_price=P.log_price,
        )
        df = pd.DataFrame(r)
        df.insert(0, "sku_id", P.skus)
        df.insert(1, "origin", o)
        df.insert(2, "target_idx", T)
        df.insert(3, "week_start", wd)
        df["scale"] = base["rm13"] + 1.0
        if with_target:
            df["y"] = Y[:, T] if T < Y.shape[1] else np.nan
        out.append(df)
    return pd.concat(out, ignore_index=True)


def training_rows(P: Panel, last_known: int, min_origin: int = 3):
    """All rows whose TARGET week is <= last_known - nothing after the cut-off is ever used."""
    frames = []
    for o in range(min_origin, last_known):
        hs = [h for h in range(1, C.HORIZON_WEEKS + 1) if o + h <= last_known]
        if hs:
            frames.append(rows_for_origin(P, o, hs))
    return pd.concat(frames, ignore_index=True)
