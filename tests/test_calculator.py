"""app.calc: cost-floor net metrics and the deploy verdict."""
import pandas as pd
import pytest

from app import calc


def _daily(vals):
    return pd.Series(vals, index=pd.date_range("2026-01-01", periods=len(vals), freq="1D", tz="UTC"))


def test_verdict_thresholds():
    assert calc.verdict(-0.5) == "do not deploy"
    assert calc.verdict(0.0) == "do not deploy"            # zero is not deployable
    assert calc.verdict(1.0).startswith("marginal")        # below the retail anchor
    assert calc.verdict(1.8) == "deployable"               # at or above the retail anchor
    assert calc.verdict(3.0) == "deployable"


def test_net_metrics_applies_the_cost_drags():
    daily = _daily([0.001] * 100)                          # 10 bp per day gross, flat
    zero = calc.net_metrics(daily, fee_round_trip=0.0, borrow_bps_day=0.0, drift_bps_day=0.0)
    assert zero["gross_apr"] == pytest.approx(0.001 * 365)
    assert zero["net_apr"] == pytest.approx(0.001 * 365)   # no drag, net equals gross
    withdrag = calc.net_metrics(daily, fee_round_trip=0.0, borrow_bps_day=1.0, drift_bps_day=0.0)
    assert withdrag["net_apr"] == pytest.approx((0.001 - 0.0001) * 365)   # 1 bp per day borrow


def test_heavy_cost_drag_flips_the_verdict_to_do_not_deploy():
    daily = _daily([0.0005, 0.0015] * 50)                  # 10 bp per day mean carry, with variance
    cheap = calc.net_metrics(daily, fee_round_trip=0.0, borrow_bps_day=0.0, drift_bps_day=0.0)
    assert cheap["net_sharpe"] > 0                         # a positive carry at zero cost
    heavy = calc.net_metrics(daily, fee_round_trip=0.0, borrow_bps_day=10.0, drift_bps_day=10.0)  # 20 bp per day
    assert heavy["net_apr"] < 0                            # the drag overwhelms the 10 bp per day carry
    assert heavy["verdict"] == "do not deploy"
