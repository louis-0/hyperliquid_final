"""Trade-flow signals: informed-cohort net flow against anonymous taker flow.

Builds fixed-length bars from the captured trade legs and exposes the two flow series the
informed-flow study compares: the frozen cohort's net signed notional (identity) and the raw
aggressor flow (anonymous). Rank correlation and top-decile forward moves quantify whether
either predicts the next bars' return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_MS = 30_000       # 30-second bars
FLOW_WIN = 4          # rolling flow window, 4 bars = 2 minutes


def signed_flows(trades: pd.DataFrame, smart: set[str]) -> pd.DataFrame:
    """Add the two signed flow columns to a raw trades frame.

    `inf_net` signs each trade's notional by the cohort's side of it (buyer in the set adds,
    seller in the set subtracts; both cancel). `taker` signs by the aggressor: side B is a
    taker buy, anything else a taker sell.
    """
    out = trades.copy()
    buy_inf = out["users_buyer"].isin(smart)
    sell_inf = out["users_seller"].isin(smart)
    out["inf_net"] = np.where(buy_inf, out["notional"], 0.0) - np.where(sell_inf, out["notional"], 0.0)
    out["taker"] = np.where(out["side"] == "B", out["notional"], -out["notional"])
    return out


def bar_aggregate(flows: pd.DataFrame, bar_ms: int = BAR_MS) -> pd.DataFrame:
    """Sum the flow columns into fixed-length bars keyed by bar-start epoch ms.

    Price is the last trade price in the bar, so a forward return over k bars is the move
    from one bar close to another.
    """
    g = flows.sort_values("trade_ms")
    g = g.assign(bar=(g["trade_ms"] // bar_ms) * bar_ms)
    return (g.groupby("bar")
             .agg(inf_net=("inf_net", "sum"), taker=("taker", "sum"),
                  px=("px", "last"), ntl=("notional", "sum"))
             .sort_index())


def fwd_ret_bps(px: pd.Series, h: int) -> pd.Series:
    """Forward return over h bars, in basis points."""
    return (px.shift(-h) / px - 1) * 1e4


def spearman_ic(signal: pd.Series, fwd: pd.Series, min_n: int = 100) -> float:
    """Rank correlation between a signal and the forward return, NaN below min_n pairs."""
    m = signal.notna() & fwd.notna()
    if int(m.sum()) < min_n:
        return float("nan")
    return float(signal[m].rank().corr(fwd[m].rank()))


def top_decile_move(signal: pd.Series, fwd: pd.Series, q: float = 0.9) -> float:
    """Mean forward move (bps) over the bars where the signal is in its top decile."""
    m = signal.notna() & fwd.notna()
    thr = signal[m].quantile(q)
    return float(fwd[m & (signal >= thr)].mean())
