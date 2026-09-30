"""D6 - FORESIGHT scoring service (FastAPI).

    uvicorn service.main:app --host 0.0.0.0 --port 8000

Returns the 8-week forecast and stockout / overstock risk for one SKU or a batch. The batch endpoint
optionally accepts a fresher stock position per SKU and re-scores risk with exactly the same rule
the pipeline uses (src/risk.py). Interactive docs at /docs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import risk as R  # noqa: E402

OUT = ROOT / "outputs"
MAX_BATCH = 200

app = FastAPI(
    title="FORESIGHT scoring service",
    version="1.0",
    description="Weekly SKU demand forecast (next 8 weeks, with an 80% range) and stockout / overstock "
                "risk with a recommended action for NorthBay Living. Built by the FORESIGHT pipeline.",
)


def _load():
    fc = pd.read_csv(OUT / "forecast.csv", parse_dates=["week_start"])
    risk = pd.read_csv(OUT / "risk_scores.csv")
    metrics = json.loads((OUT / "metrics.json").read_text())
    return fc, risk.set_index("sku_id"), metrics


try:
    FC, RISK, METRICS = _load()
    LOAD_ERROR = None
except Exception as e:  # service still starts and reports the problem instead of crashing
    FC, RISK, METRICS, LOAD_ERROR = None, None, None, str(e)


# --------------------------------------------------------------------------------- schemas
class StockOverride(BaseModel):
    sku_id: str = Field(..., examples=["SKU012"], description="Product id, e.g. SKU012 (case-insensitive).")
    on_hand_units: Optional[float] = Field(None, ge=0, description="Units physically in stock. Default: latest snapshot.")
    on_order_units: Optional[float] = Field(None, ge=0, description="Units ordered, not yet received. Default: latest snapshot.")
    lead_time_days: Optional[float] = Field(None, gt=0, le=365, description="Supplier lead time in days. Default: latest snapshot.")
    safety_stock: Optional[float] = Field(None, ge=0, description="Safety stock in units. Default: latest snapshot.")


class BatchRequest(BaseModel):
    items: List[StockOverride] = Field(..., min_length=1, max_length=MAX_BATCH)


# --------------------------------------------------------------------------------- helpers
def _ready():
    if LOAD_ERROR:
        raise HTTPException(503, detail=f"Model outputs not available - run `python run_pipeline.py` first. ({LOAD_ERROR})")


def _norm(sku: str) -> str:
    return str(sku).strip().upper()


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _forecast_block(sku: str):
    f = FC[FC.sku_id == sku].sort_values("h")
    return [dict(week_start=d.week_start.strftime("%Y-%m-%d"), forecast_units=round(d.forecast, 1),
                 lower_80=round(d.lower_80, 1), upper_80=round(d.upper_80, 1),
                 forecast_units_without_promo=round(d.forecast_no_promo, 1),
                 same_week_last_year=_clean(float(d.baseline))) for d in f.itertuples()]


def _score(sku: str, ov: Optional[StockOverride] = None):
    if sku not in RISK.index:
        return None
    base = RISK.loc[sku]
    inputs = dict(on_hand_units=float(base.on_hand_units), on_order_units=float(base.on_order_units),
                  lead_time_days=float(base.lead_time_days), safety_stock=float(base.safety_stock))
    source = f"latest snapshot ({base.stock_as_of})"
    if ov is not None:
        given = {k: v for k, v in ov.model_dump().items() if k != "sku_id" and v is not None}
        if given:
            inputs.update(given)
            source = "request (missing fields filled from latest snapshot)"
    weekly = FC[FC.sku_id == sku].sort_values("h").forecast.tolist()
    rel_sd = {int(k): v for k, v in METRICS["window_rel_sd"].items()}
    scored = R.score_sku(weekly, inputs["on_hand_units"], inputs["on_order_units"], inputs["lead_time_days"],
                         inputs["safety_stock"], float(base.unit_cost), float(base.list_price), rel_sd,
                         float(base.forecast_reliability))
    return dict(
        sku_id=sku, product_name=base.product_name, category=base.category,
        stock_inputs=dict(**inputs, source=source),
        risk={k: _clean(v) for k, v in scored.items()},
        forecast=_forecast_block(sku),
    )


# -------------------------------------------------------------------------- error handling
@app.exception_handler(RequestValidationError)
async def _bad_input(request: Request, exc: RequestValidationError):
    problems = [f"{'.'.join(str(x) for x in e['loc'] if x != 'body')}: {e['msg']}" for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "Invalid input", "problems": problems,
                                                  "hint": "See /docs for the expected request format."})


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Unexpected server error", "detail": str(exc)})


# ------------------------------------------------------------------------------- endpoints
@app.get("/", summary="Service info")
def root():
    info = dict(service="FORESIGHT scoring service", docs="/docs",
                endpoints={"GET /health": "status", "GET /skus": "list of SKU ids",
                           "GET /score/{sku_id}": "forecast + risk for one SKU",
                           "POST /score": "forecast + risk for a batch, optional fresher stock inputs"})
    if METRICS:
        info.update(as_of=METRICS["as_of"], horizon_weeks=METRICS["horizon_weeks"],
                    backtest_wape=round(METRICS["backtest"]["model_wape"], 4),
                    baseline_wape=round(METRICS["backtest"]["baseline_wape"], 4))
    return info


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok" if not LOAD_ERROR else "degraded", "skus_loaded": 0 if RISK is None else len(RISK),
            "detail": LOAD_ERROR}


@app.get("/skus", summary="List SKUs the service can score")
def skus():
    _ready()
    return {"count": len(RISK), "skus": sorted(RISK.index.tolist())}


@app.get("/score/{sku_id}", summary="Forecast + risk for one SKU (latest stock snapshot)")
def score_one(sku_id: str):
    _ready()
    sku = _norm(sku_id)
    out = _score(sku)
    if out is None:
        raise HTTPException(404, detail=f"Unknown SKU '{sku_id}'. Call GET /skus for valid ids.")
    return out


@app.post("/score", summary="Forecast + risk for a batch of SKUs")
def score_batch(req: BatchRequest):
    """Each item needs `sku_id`; stock fields are optional overrides of the latest snapshot.
    Unknown SKUs are reported per item in `errors` instead of failing the whole batch."""
    _ready()
    results, errors = [], []
    for item in req.items:
        sku = _norm(item.sku_id)
        out = _score(sku, item)
        if out is None:
            errors.append({"sku_id": item.sku_id, "error": "Unknown SKU"})
        else:
            results.append(out)
    return {"as_of": METRICS["as_of"], "count": len(results), "results": results, "errors": errors}
