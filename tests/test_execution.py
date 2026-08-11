"""hlq.execution: order-book walking for volume-weighted fill price and slippage."""
import pytest

from hlq import execution


def _asks(*levels):
    return [{"px": px, "sz": sz} for px, sz in levels]


def test_single_level_fill_slippage():
    # buy $10,000 into one ask at 101 against a mid of 100 -> 1% slippage
    slip = execution.walk_book(_asks((101.0, 100.0)), target_notional=10_000, mid=100.0)
    assert slip == pytest.approx(0.01)


def test_multi_level_vwap():
    # 50 units @100 ($5,000) + 50 units @110 ($5,500) -> vwap 105 -> 5% slippage
    slip = execution.walk_book(_asks((100.0, 50.0), (110.0, 50.0)), target_notional=10_500, mid=100.0)
    assert slip == pytest.approx(0.05)


def test_vwap_is_cost_per_unit_not_notional_weighted():
    # $1,000 buys 1 unit @100 then 0.9 units @1000 -> 1.9 units, vwap = 1000/1.9.
    # A price weighted by notional would land far higher and overstate the slippage.
    asks = _asks((100.0, 1.0), (1000.0, 100.0))
    slip = execution.walk_book(asks, target_notional=1_000, mid=100.0)
    expected = (1_000 / (1.0 + 0.9) - 100.0) / 100.0
    assert slip == pytest.approx(expected)


def test_insufficient_depth_returns_none():
    # only $101 of depth cannot fill a $10,000 order
    assert execution.walk_book(_asks((101.0, 1.0)), target_notional=10_000, mid=100.0) is None


def test_parse_l2_snapshot_returns_mid_and_asks():
    lj = '[[{"px": 99.0, "sz": 5.0}], [{"px": 101.0, "sz": 5.0}, {"px": 102.0, "sz": 3.0}]]'
    mid, asks = execution.parse_l2_snapshot(99.0, 101.0, lj)
    assert mid == pytest.approx(100.0)
    assert asks == [{"px": 101.0, "sz": 5.0}, {"px": 102.0, "sz": 3.0}]


def test_parse_l2_snapshot_none_on_malformed_json():
    assert execution.parse_l2_snapshot(99.0, 101.0, "not json") is None


def test_parse_l2_snapshot_none_when_ask_side_missing():
    assert execution.parse_l2_snapshot(99.0, 101.0, '[[{"px": 99.0, "sz": 5.0}]]') is None  # bids only
    assert execution.parse_l2_snapshot(99.0, 101.0, '[[], []]') is None                      # empty asks