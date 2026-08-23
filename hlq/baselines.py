"""Naive prediction baselines for the funding-timing feasibility check, and the
buy-and-hold return series used by the strategy-baseline comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd


def majority_class_accuracy(y_train, y_test) -> float:
    """Accuracy of always predicting the training majority class, evaluated on y_test.

    The naive direction baseline: if funding is positive more often than not in training,
    predict positive for every test hour and measure how often that is correct.
    """
    majority = int(np.asarray(y_train).mean() > 0.5)
    return float((np.asarray(y_test) == majority).mean())


def buy_and_hold_daily(candles: pd.DataFrame) -> pd.Series:
    """Daily buy-and-hold return series from candle closes.

    Each UTC day is represented by its last close; the series is that daily price's
    simple return. Holding the asset earns exactly this series, so it is the
    passive benchmark the timed strategy is compared against.
    """
    ts = pd.to_datetime(candles["ts"], utc=True)
    daily = pd.Series(candles["close"].to_numpy(), index=ts).resample("1D").last().dropna()
    return daily.pct_change(fill_method=None).dropna()


def persistence_rmse(actual, last_value) -> float:
    """RMSE of the last-value forecast: predict each hour's funding as the previous hour's.
    `last_value` is the aligned previous-hour funding series."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(last_value, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2)))