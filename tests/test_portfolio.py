"""hlq.portfolio: equal-weight basket construction over the coins' common window."""
import pandas as pd
import pytest

from hlq import portfolio


def _daily(dates, vals):
    return pd.Series(vals, index=pd.to_datetime(dates, utc=True))


def test_equal_weight_is_the_mean_across_coins():
    dates = ["2026-06-01", "2026-06-02", "2026-06-03"]
    a = _daily(dates, [0.01, 0.02, 0.03])
    b = _daily(dates, [0.03, 0.02, 0.01])
    basket = portfolio.equal_weight_basket({"A": a, "B": b})
    assert basket.tolist() == pytest.approx([0.02, 0.02, 0.02])


def test_basket_uses_the_common_window_intersection_not_union():
    # B lacks the last day; the basket must drop that day, not average B alone on it.
    # Intersection, not union: a longer-history coin must not distort the days it alone covers.
    a = _daily(["2026-06-01", "2026-06-02", "2026-06-03"], [0.01, 0.02, 0.03])
    b = _daily(["2026-06-01", "2026-06-02"], [0.03, 0.02])
    basket = portfolio.equal_weight_basket({"A": a, "B": b})
    assert len(basket) == 2                                       # intersection of the two windows
    assert basket.index.max() == pd.Timestamp("2026-06-02", tz="UTC")
    assert basket.tolist() == pytest.approx([0.02, 0.02])


def test_exclude_builds_the_named_sub_basket():
    dates = ["2026-06-01", "2026-06-02"]
    coins = {"BTC": _daily(dates, [0.01, 0.01]),
             "SOL": _daily(dates, [0.05, 0.05]),
             "ETH": _daily(dates, [0.03, 0.03])}
    full = portfolio.equal_weight_basket(coins)
    drop_sol = portfolio.equal_weight_basket(coins, exclude=["SOL"])
    assert full.iloc[0] == pytest.approx((0.01 + 0.05 + 0.03) / 3)
    assert drop_sol.iloc[0] == pytest.approx((0.01 + 0.03) / 2)


def test_single_coin_basket_is_that_coin():
    a = _daily(["2026-06-01", "2026-06-02"], [0.01, 0.02])
    basket = portfolio.equal_weight_basket({"A": a})
    assert basket.tolist() == pytest.approx([0.01, 0.02])


def test_excluding_every_coin_gives_an_empty_basket():
    a = _daily(["2026-06-01"], [0.01])
    assert portfolio.equal_weight_basket({"A": a}, exclude=["A"]).empty
