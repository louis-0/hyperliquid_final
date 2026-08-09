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
