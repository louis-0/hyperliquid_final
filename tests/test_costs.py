"""hlq.costs: taker/maker/slippage round-trip and per-turnover amortisation."""
import pytest

from hlq.costs import CostModel


def test_taker_round_trip_is_11bps():
    # defaults: 2 * (4.5 + 1) bps = 0.11%
    assert CostModel().round_trip() == pytest.approx(0.0011)


def test_maker_is_cheaper_than_taker():
    c = CostModel()
    assert c.round_trip(maker=True) < c.round_trip()


def test_maker_rebate_is_negative():
    assert CostModel(maker_fee=-0.00005).round_trip(maker=True) < 0.0


def test_amortised_daily_spreads_round_trip():
    c = CostModel()
    assert c.amortised_daily(100) == pytest.approx(c.round_trip() / 100)


def test_amortised_daily_handles_zero_days():
    assert CostModel().amortised_daily(0) == 0.0
