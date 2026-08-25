"""hlq.microfeatures: hourly aggregates from the captured tape."""
import pandas as pd
import pytest

from hlq import microfeatures

H0 = int(pd.Timestamp("2026-06-10 00:00", tz="UTC").timestamp() * 1000)
H1 = int(pd.Timestamp("2026-06-10 01:00", tz="UTC").timestamp() * 1000)


@pytest.fixture
def ws_root(tmp_path):
    """A one-coin ws tree with two hours of synthetic bbo/trades/ctx rows."""
    for ch in ("bbo", "trades", "ctx"):
        (tmp_path / "ws" / "BTC" / ch / "2026-06-10").mkdir(parents=True)

    bbo = pd.DataFrame({
        "bbo_ms": [H0 + 1000, H0 + 2000, H1 + 1000],
        "bid_px": [99.0, 99.0, 99.0], "ask_px": [101.0, 101.0, 101.0],
        "bid_sz": [2.0, 1.0, 5.0], "ask_sz": [1.0, 3.0, 5.0],
    })
    bbo.to_parquet(tmp_path / "ws" / "BTC" / "bbo" / "2026-06-10" / "00.parquet", index=False)

    trades = pd.DataFrame({
        "trade_ms": [H0 + 1000, H0 + 2000],
        "side": ["B", "A"], "notional": [100.0, 50.0],
        "users_buyer": ["0xaaa", "0xccc"], "users_seller": ["0xbbb", "0xaaa"],
    })
    trades.to_parquet(tmp_path / "ws" / "BTC" / "trades" / "2026-06-10" / "00.parquet", index=False)

    ctx = pd.DataFrame({
        "captured_ms": [H0 + 1000, H0 + 2000],
        "premium": [0.001, 0.003], "funding": [0.00001, 0.00002],
    })
    ctx.to_parquet(tmp_path / "ws" / "BTC" / "ctx" / "2026-06-10" / "00.parquet", index=False)
    return tmp_path


def test_hourly_book_imbalance_and_spread(ws_root):
    out = microfeatures.hourly("BTC", root=ws_root).set_index("hour")
    h0 = out.iloc[0]
    # row imbalances (2-1)/3 and (1-3)/4 average to 1/24; spread is 2/100 = 200 bps
    assert h0["obi_mean"] == pytest.approx((1 / 3 - 1 / 2) / 2)
    assert h0["spread_bps_mean"] == pytest.approx(200.0)


def test_hourly_buckets_are_separate(ws_root):
    out = microfeatures.hourly("BTC", root=ws_root)
    assert len(out) == 2
    # hour 1 has one balanced bbo row and no trades or ctx
    h1 = out.iloc[1]
    assert h1["obi_mean"] == pytest.approx(0.0)
    assert pd.isna(h1["taker_imbalance"])


def test_hourly_taker_imbalance(ws_root):
    h0 = microfeatures.hourly("BTC", root=ws_root).iloc[0]
    # buy 100 against sell 50: (100 - 50) / 150
    assert h0["taker_imbalance"] == pytest.approx(1 / 3)


def test_hourly_premium_and_funding_are_last_and_mean(ws_root):
    h0 = microfeatures.hourly("BTC", root=ws_root).iloc[0]
    assert h0["premium_mean"] == pytest.approx(0.002)
    assert h0["premium_last"] == pytest.approx(0.003)
    assert h0["ctx_funding_last"] == pytest.approx(0.00002)


def test_hourly_cohort_net_share(ws_root):
    # 0xaaa buys 100 and sells 50: signed +50 over 300 of leg notional
    h0 = microfeatures.hourly("BTC", root=ws_root, smart={"0xaaa"}).iloc[0]
    assert h0["cohort_net_share"] == pytest.approx(50 / 300)
