"""hlq.flow: signed flows, bar aggregation, and rank statistics."""
import pandas as pd
import pytest

from hlq import flow


def _trades():
    return pd.DataFrame({
        "trade_ms": [0, 10_000, 40_000],
        "side": ["B", "A", "B"],
        "px": [100.0, 101.0, 102.0],
        "notional": [50.0, 30.0, 20.0],
        "users_buyer": ["0xaaa", "0xccc", "0xaaa"],
        "users_seller": ["0xbbb", "0xaaa", "0xddd"],
    })


def test_signed_flows_identity_and_taker():
    f = flow.signed_flows(_trades(), smart={"0xaaa"})
    # 0xaaa buys 50, sells 30, buys 20 -> +50, -30, +20; taker signs by aggressor side
    assert f["inf_net"].tolist() == [50.0, -30.0, 20.0]
    assert f["taker"].tolist() == [50.0, -30.0, 20.0]


def test_signed_flows_cancel_when_cohort_on_both_sides():
    f = flow.signed_flows(_trades(), smart={"0xaaa", "0xbbb"})
    # first trade has cohort wallets on both legs: +50 - 50 = 0
    assert f["inf_net"].iloc[0] == pytest.approx(0.0)


def test_bar_aggregate_splits_bars_and_takes_last_px():
    f = flow.signed_flows(_trades(), smart=set())
    bars = flow.bar_aggregate(f, bar_ms=30_000)
    # trades at 0s and 10s share bar 0 (px last 101); trade at 40s is bar 30000
    assert bars.index.tolist() == [0, 30_000]
    assert bars["px"].tolist() == [101.0, 102.0]
    assert bars["taker"].iloc[0] == pytest.approx(50.0 - 30.0)


def test_fwd_ret_bps_known_value():
    px = pd.Series([100.0, 101.0, 102.0])
    # one-bar forward return from 100 to 101 is 100 bps
    assert flow.fwd_ret_bps(px, 1).iloc[0] == pytest.approx(100.0)


def test_spearman_ic_monotone_is_one():
    s = pd.Series(range(200), dtype=float)
    assert flow.spearman_ic(s, s * 2, min_n=100) == pytest.approx(1.0)


def test_top_decile_move_hand_value():
    sig = pd.Series(range(100), dtype=float)
    fwd = pd.Series([0.0] * 90 + [10.0] * 10)
    # top decile of the signal is exactly the ten bars with fwd = 10
    assert flow.top_decile_move(sig, fwd) == pytest.approx(10.0)
