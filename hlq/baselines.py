"""Naive prediction baselines for the funding-timing feasibility check."""
from __future__ import annotations

import numpy as np


def majority_class_accuracy(y_train, y_test) -> float:
    """Accuracy of always predicting the training majority class, evaluated on y_test.

    The naive direction baseline: if funding is positive more often than not in training,
    predict positive for every test hour and measure how often that is correct.
    """
    majority = int(np.asarray(y_train).mean() > 0.5)
    return float((np.asarray(y_test) == majority).mean())


def persistence_rmse(actual, last_value) -> float:
    """RMSE of the last-value forecast: predict each hour's funding as the previous hour's.
    `last_value` is the aligned previous-hour funding series."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(last_value, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2)))