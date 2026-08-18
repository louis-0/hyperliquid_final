"""Statistics: Sharpe ratio, daily aggregation, block bootstrap, Probabilistic/Deflated Sharpe.

Sources are cited on each function: Sharpe (1994); Künsch (1989); Bailey and Lopez de Prado (2012, 2014).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 365          # crypto trades 7 days a week
GAMMA = np.euler_gamma      # Euler-Mascheroni constant, used in the DSR expected-max term


def annualised_sharpe(daily: pd.Series) -> tuple[float, float, float]:
    """Sharpe ratio (Sharpe 1994): return (sharpe, ann_return, ann_vol); 0 when the series is flat."""
    mu = daily.mean() * TRADING_DAYS
    sigma = daily.std() * np.sqrt(TRADING_DAYS)      # ddof=1
    sharpe = mu / sigma if sigma > 0 else 0.0
    return float(sharpe), float(mu), float(sigma)


def psr_inputs(daily: pd.Series) -> tuple[float, int, float, float]:
    """Per-observation (Sharpe, T, skew, non-excess kurtosis) for psr/deflated_sharpe, from a return series."""
    d = pd.Series(daily).dropna()
    T = len(d)
    sd = d.std()                                    # ddof=1
    if not sd > 0:                                  # flat (or degenerate) series: no dispersion
        return 0.0, T, 0.0, 3.0
    sr = float(d.mean() / sd)
    skew = float(d.skew()) if T > 2 else 0.0
    kurt = float(d.kurt() + 3.0) if T > 3 else 3.0  # pandas kurt is excess; psr wants non-excess
    return sr, T, skew, kurt


def to_daily(returns: pd.Series) -> pd.Series:
    """Sum a datetime-indexed return series into daily UTC buckets."""
    idx = pd.DatetimeIndex(returns.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return pd.Series(returns.to_numpy(), index=idx).resample("1D").sum().dropna()


def block_bootstrap_ci(daily: pd.Series, n_resamples: int = 1000, block: int = 10,
                       seed: int = 42, ci: float = 0.95) -> tuple[float, float]:
    """Moving-block bootstrap CI on the annualised Sharpe (Künsch 1989). block=1 is an IID
    resample; block>1 keeps serial correlation, giving a wider CI for autocorrelated series."""
    rng = np.random.default_rng(seed=seed)
    arr = np.asarray(daily.to_numpy(), dtype=float)
    n = len(arr)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    sharpes = []
    for _ in range(n_resamples):
        n_blocks = max(1, n // block)
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        sample = np.concatenate([arr[s:s + block] for s in starts])[:n]
        mu = sample.mean() * TRADING_DAYS
        sigma = sample.std() * np.sqrt(TRADING_DAYS)     # ddof=0 (numpy)
        sharpes.append(mu / sigma if sigma > 0 else 0.0)
    return float(np.percentile(sharpes, lo_q)), float(np.percentile(sharpes, hi_q))


def psr(sr: float, T: int, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey and Lopez de Prado 2012). sr/sr_star per-observation; kurt non-excess."""
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))
    z = (sr - sr_star) * np.sqrt(max(1, T - 1)) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """Expected max of n_trials iid Sharpes ~ N(0, var_sr), the DSR deflation benchmark (Bailey and Lopez de Prado 2014)."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1.0 - GAMMA) * z1 + GAMMA * z2))


def deflated_sharpe(sr: float, T: int, skew: float, kurt: float,
                    n_trials: int, var_sr: float) -> float:
    """Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014): PSR against the expected-max-Sharpe benchmark."""
    return psr(sr, T, skew, kurt, expected_max_sharpe(var_sr, n_trials))
