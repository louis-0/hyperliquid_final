"""hlq.baselines: naive prediction baselines and the buy-and-hold return series."""
import pandas as pd
import pytest

from hlq import baselines


def test_majority_class_accuracy_predicts_training_majority():
    # training is 3/4 positive -> majority is 1; test has 2 of 3 positive -> accuracy 2/3
    assert baselines.majority_class_accuracy([1, 1, 1, 0], [1, 0, 1]) == pytest.approx(2 / 3)


def test_majority_class_accuracy_flips_when_training_mostly_negative():
    # training mostly 0 -> majority is 0; test all 0 -> accuracy 1.0
    assert baselines.majority_class_accuracy([0, 0, 0, 1], [0, 0]) == pytest.approx(1.0)


def test_buy_and_hold_daily_simple_return():
    # last close moves 100 -> 110 across two days: one daily return of +10%
    idx = pd.date_range("2026-06-01", periods=48, freq="h", tz="UTC")
    closes = [100.0] * 24 + [105.0] * 23 + [110.0]
    df = pd.DataFrame({"ts": idx, "close": closes})
    ret = baselines.buy_and_hold_daily(df)
    assert len(ret) == 1
    assert ret.iloc[0] == pytest.approx(0.10)


def test_buy_and_hold_daily_uses_last_close_of_day():
    # intraday spike then close back at 100: day-over-day return is zero
    idx = pd.date_range("2026-06-01", periods=48, freq="h", tz="UTC")
    closes = [100.0] * 24 + [140.0] * 23 + [100.0]
    df = pd.DataFrame({"ts": idx, "close": closes})
    ret = baselines.buy_and_hold_daily(df)
    assert ret.iloc[0] == pytest.approx(0.0)


def test_persistence_rmse_matches_hand_value():
    # errors of 0, +2, -2 -> sqrt(mean(0, 4, 4)) = sqrt(8/3)
    assert baselines.persistence_rmse([10.0, 12.0, 8.0], [10.0, 10.0, 10.0]) == pytest.approx((8 / 3) ** 0.5)