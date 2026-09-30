"""Checks behind the acceptance criteria: no leakage (D3.4), transparent risk rule (D4), graceful API (D6.4).

    python -m pytest -q tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline, risk as R  # noqa: E402
from src.features import FEATURES, Panel, rows_for_origin, training_rows  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return Panel(pipeline.run(save=False, verbose=False)["weekly"])


def test_features_do_not_see_the_future(panel):
    """Scramble every week after the origin: features must not change (only the target may)."""
    o = 80
    before = rows_for_origin(panel, o)
    tampered = Panel.__new__(Panel)
    tampered.__dict__.update(panel.__dict__)
    tampered.Y = panel.Y.copy()
    tampered.Y[:, o + 1:] = np.random.default_rng(0).integers(0, 10_000, tampered.Y[:, o + 1:].shape)
    after = rows_for_origin(tampered, o)
    np.testing.assert_allclose(before[FEATURES].to_numpy(float), after[FEATURES].to_numpy(float), equal_nan=True)
    assert not np.allclose(before["y"], after["y"])


def test_training_rows_stop_at_cutoff(panel):
    tr = training_rows(panel, 70)
    assert tr["target_idx"].max() <= 70
    assert (tr["origin"] < tr["target_idx"]).all()


def test_weekly_panel_complete(panel):
    assert panel.Y.shape == (50, 104)
    assert not np.isnan(panel.Y).any()


REL_SD = {k: 0.08 for k in range(1, 9)}


def test_low_stock_is_reorder():
    r = R.score_sku([100] * 8, on_hand=20, on_order=0, lead_time_days=10, safety_stock=20,
                    unit_cost=100, price=200, rel_sd=REL_SD)
    assert r["quadrant"] == "Reorder now" and r["stockout_risk"] > 0.9
    assert r["sales_at_risk_inr"] == pytest.approx((800 - 20) * 200, rel=1e-6)


def test_excess_stock_is_markdown():
    r = R.score_sku([100] * 8, on_hand=3000, on_order=0, lead_time_days=10, safety_stock=50,
                    unit_cost=100, price=200, rel_sd=REL_SD)
    assert r["quadrant"] == "Markdown / clear"
    assert r["capital_locked_inr"] == pytest.approx((3000 - 50 - 800) * 100, rel=1e-6)


def test_balanced_stock_is_healthy():
    r = R.score_sku([100] * 8, on_hand=500, on_order=0, lead_time_days=7, safety_stock=50,
                    unit_cost=100, price=200, rel_sd=REL_SD)
    assert r["quadrant"] == "Healthy" and r["value_at_stake_inr"] == 0


def test_api_handles_bad_input():
    from fastapi.testclient import TestClient
    from service.main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/score/NOT-A-SKU").status_code == 404
    assert c.post("/score", json={"items": [{"sku_id": "SKU001", "on_hand_units": -1}]}).status_code == 422
    assert c.post("/score", json={"items": []}).status_code == 422
    r = c.post("/score", json={"items": [{"sku_id": "sku001"}, {"sku_id": "bogus"}]}).json()
    assert r["count"] == 1 and r["errors"][0]["sku_id"] == "bogus"
