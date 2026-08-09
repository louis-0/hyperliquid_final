"""hlq.stats: Sharpe, daily aggregation, block bootstrap, Probabilistic/Deflated Sharpe."""
import numpy as np
import pandas as pd
import pytest

from hlq import stats


def test_annualised_sharpe_known_value():
    # mean/std = 2, so sharpe = 2*sqrt(365)
    s = pd.Series([0.001, 0.002, 0.003])
    sharpe, ann_ret, ann_vol = stats.annualised_sharpe(s)
    assert sharpe == pytest.approx(2.0 * np.sqrt(365), rel=1e-9)
    assert ann_ret == pytest.approx(0.002 * 365, rel=1e-9)
    assert ann_vol == pytest.approx(0.001 * np.sqrt(365), rel=1e-9)


def test_annualised_sharpe_flat_series_is_zero():
    sharpe, _, ann_vol = stats.annualised_sharpe(pd.Series([0.005, 0.005, 0.005]))
    assert sharpe == 0.0
    assert ann_vol == 0.0


def test_to_daily_sums_within_utc_day():
    idx = pd.date_range("2026-06-01", periods=48, freq="h", tz="UTC")
    daily = stats.to_daily(pd.Series(0.0001, index=idx))
    assert len(daily) == 2
    assert daily.iloc[0] == pytest.approx(24 * 0.0001, rel=1e-9)


def test_to_daily_localises_tz_naive_index():
    idx = pd.date_range("2026-06-01", periods=24, freq="h")   # tz-naive index, treated as UTC
    daily = stats.to_daily(pd.Series(0.0002, index=idx))
    assert len(daily) == 1
    assert daily.iloc[0] == pytest.approx(24 * 0.0002, rel=1e-9)


def test_psr_is_half_when_sharpe_equals_benchmark():
    # z is 0 when sr equals the benchmark, so PSR = 0.5 for any T/skew/kurt
    assert stats.psr(0.1, 200, -0.3, 5.0, sr_star=0.1) == pytest.approx(0.5)


def test_psr_penalises_fat_tails():
    # higher kurtosis widens the denominator, lowering PSR for the same Sharpe
    assert stats.psr(0.12, 200, 0.0, 8.0) < stats.psr(0.12, 200, 0.0, 3.0)


def test_psr_strong_sharpe_near_one():
    # annual SR 2.5 over ~5y, per trading day (Bailey-Lopez de Prado worked-example scale)
    assert stats.psr(2.5 / np.sqrt(252), 1260, -0.3, 5.0, 0.0) > 0.99


def test_expected_max_sharpe_increases_with_trials():
    var_sr = 0.04
    assert stats.expected_max_sharpe(var_sr, 100) > stats.expected_max_sharpe(var_sr, 10) > 0.0


def test_expected_max_sharpe_degenerate():
    assert stats.expected_max_sharpe(0.0, 100) == 0.0
    assert stats.expected_max_sharpe(0.04, 1) == 0.0


def test_deflated_sharpe_falls_with_more_trials():
    sr, T, var_sr = 2.5 / np.sqrt(252), 1260, (0.6 / np.sqrt(252)) ** 2
    assert stats.deflated_sharpe(sr, T, -0.3, 5.0, 10, var_sr) > stats.deflated_sharpe(sr, T, -0.3, 5.0, 1000, var_sr)


def test_deflation_lowers_significance():
    # deflating for N>1 trials raises the benchmark above 0, so DSR < the raw PSR
    sr, T, var_sr = 0.15, 200, 0.02
    assert stats.deflated_sharpe(sr, T, 0.0, 3.0, 50, var_sr) < stats.psr(sr, T, 0.0, 3.0, 0.0)


def test_block_bootstrap_deterministic_and_ordered():
    s = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 200))
    a = stats.block_bootstrap_ci(s, seed=42)
    b = stats.block_bootstrap_ci(s, seed=42)
    assert a == b
    lo, hi = a
    assert lo < hi


def test_block_length_widens_ci_for_autocorrelated_series():
    # AR(1) with strong persistence: block>1 must give a wider CI than block=1
    rng = np.random.default_rng(1)
    n = 300
    e = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.85 * r[t - 1] + e[t]
    s = pd.Series(r + 0.002)
    lo1, hi1 = stats.block_bootstrap_ci(s, block=1, seed=42)
    lo20, hi20 = stats.block_bootstrap_ci(s, block=20, seed=42)
    assert (hi20 - lo20) > (hi1 - lo1)
