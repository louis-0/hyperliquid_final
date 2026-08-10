"""hlq.signals: funding-carry and basis-drift hourly PnL."""
import pandas as pd
import pytest

from hlq import signals


def _frame(hours, **cols):
    return pd.DataFrame({"ts": pd.to_datetime(hours, utc=True), **cols})


def test_funding_carry_pnl_is_the_funding_rate():
    rates = [0.001, -0.0005, 0.0003]
    funding = _frame(["2026-06-01 00:00", "2026-06-01 01:00", "2026-06-01 02:00"], funding_rate=rates)
    assert signals.funding_carry_pnl(funding)["pnl"].tolist() == pytest.approx(rates)


def test_basis_drift_pnl_combines_drift_and_funding():
    hours = ["2026-06-01 00:00", "2026-06-01 01:00"]
    perp = _frame(hours, close=[100.0, 102.0])
    spot = _frame(hours, close=[100.0, 101.0])
    funding = _frame(hours, funding_rate=[0.0002, 0.0003])
    out = signals.basis_drift_pnl(funding, spot, perp)
    d_spot, d_perp, prior_perp = 101.0 - 100.0, 102.0 - 100.0, 100.0
    drift = (d_spot - d_perp) / prior_perp
    assert len(out) == 1                                   # first bar has no prior, so it is dropped
    assert out["drift_pnl"].iloc[0] == pytest.approx(drift)
    assert out["funding_pnl"].iloc[0] == pytest.approx(0.0003)
    assert out["pnl"].iloc[0] == pytest.approx(drift + 0.0003)


def test_basis_drift_normalises_both_legs_by_prior_perp():
    # real basis: prior spot 110 != prior perp 100; both legs divide by prior perp
    hours = ["2026-06-01 00:00", "2026-06-01 01:00"]
    perp = _frame(hours, close=[100.0, 102.0])
    spot = _frame(hours, close=[110.0, 111.0])
    funding = _frame(hours, funding_rate=[0.0, 0.0])
    out = signals.basis_drift_pnl(funding, spot, perp)
    assert out["drift_pnl"].iloc[0] == pytest.approx((1.0 - 2.0) / 100.0)


def test_basis_drift_perfect_hedge_leaves_only_funding():
    hours = ["2026-06-01 00:00", "2026-06-01 01:00", "2026-06-01 02:00"]
    px = [100.0, 105.0, 110.0]
    perp = _frame(hours, close=px)
    spot = _frame(hours, close=px)                          # spot tracks perp exactly, so drift is zero
    funding = _frame(hours, funding_rate=[0.0001, 0.0001, 0.0001])
    out = signals.basis_drift_pnl(funding, spot, perp)
    assert (out["drift_pnl"].abs() < 1e-12).all()
    assert out["pnl"].tolist() == pytest.approx(out["funding_pnl"].tolist())


def test_basis_drift_aligns_on_the_hour():
    # sub-hour timestamps on different offsets still join on the hour bucket
    perp = _frame(["2026-06-01 00:59:59.999", "2026-06-01 01:59:59.999"], close=[100.0, 102.0])
    spot = _frame(["2026-06-01 00:00:00.100", "2026-06-01 01:00:00.100"], close=[100.0, 101.0])
    funding = _frame(["2026-06-01 00:00:00", "2026-06-01 01:00:00"], funding_rate=[0.0, 0.0003])
    out = signals.basis_drift_pnl(funding, spot, perp)
    assert len(out) == 1
    assert out["pnl"].iloc[0] == pytest.approx((1.0 - 2.0) / 100.0 + 0.0003)


def test_regime_label_classifies_bull_bear_calm():
    # rolling_days=1: each day's label depends on the move since the prior day
    hours = ["2026-06-01 12:00", "2026-06-02 12:00", "2026-06-03 12:00", "2026-06-04 12:00"]
    perp = _frame(hours, close=[100.0, 112.0, 100.0, 105.0])
    labels = signals.regime_label(perp, rolling_days=1)
    #  d0: no prior -> calm | d1: +12% -> bull | d2: -10.7% -> bear | d3: +5% -> calm
    assert labels.tolist() == ["calm", "bull", "bear", "calm"]


def test_regime_label_threshold_direction():
    # +9% stays calm (below the +10% line); larger moves cross into bull / bear
    calm_up = _frame(["2026-06-01", "2026-06-02"], close=[100.0, 109.0])
    bull = _frame(["2026-06-01", "2026-06-02"], close=[100.0, 120.0])
    bear = _frame(["2026-06-01", "2026-06-02"], close=[100.0, 80.0])
    assert signals.regime_label(calm_up, rolling_days=1).iloc[-1] == "calm"
    assert signals.regime_label(bull, rolling_days=1).iloc[-1] == "bull"
    assert signals.regime_label(bear, rolling_days=1).iloc[-1] == "bear"


def test_regime_label_uses_last_mark_of_each_day():
    # two ticks on the second day; the later close (112) sets that day's value
    perp = _frame(["2026-06-01 12:00", "2026-06-02 09:00", "2026-06-02 20:00"],
                  close=[100.0, 999.0, 112.0])
    assert signals.regime_label(perp, rolling_days=1).tolist() == ["calm", "bull"]


def test_regime_label_is_calm_until_enough_history():
    # a rolling window longer than the data leaves the return undefined -> calm everywhere
    perp = _frame(["2026-06-01", "2026-06-02", "2026-06-03"], close=[100.0, 50.0, 200.0])
    assert (signals.regime_label(perp, rolling_days=5) == "calm").all()
