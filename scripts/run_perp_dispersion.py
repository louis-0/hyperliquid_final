#!/usr/bin/env python3
"""Perp-versus-perp funding dispersion test, built on the hlq package.

Pre-registered test of the one captured surface the falsification record has not covered:
short the highest-funding perpetuals against long the lowest-funding perpetuals, no spot leg
and no borrow. Each rebalance ranks the universe by trailing 30-day mean funding and holds
short the top quintile against long the bottom quintile, equal weight, for 30 days. Reported
gross of costs and net of a full-churn bound (every position re-entered every rebalance,
one perp round trip per leg). Falsification criterion: net Sharpe at full churn <= 0.

    python scripts/run_perp_dispersion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, results, stats
from hlq.costs import CostModel

LOOKBACK_D = 30
HOLD_D = 30
COST = CostModel()
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def hourly_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hour-indexed funding and mark-close panels across every coin with both series."""
    fund, marks = {}, {}
    for fp in sorted((data.DATA_ROOT / "funding").glob("*.parquet")):
        coin = fp.stem
        try:
            f = data.load_funding(coin)
            m = data.load_marks(coin)
        except FileNotFoundError:
            continue
        fund[coin] = pd.Series(f["funding_rate"].to_numpy(),
                               index=pd.DatetimeIndex(data.floor_hour(f["ts"])))
        marks[coin] = pd.Series(m["close"].to_numpy(),
                                index=pd.DatetimeIndex(data.floor_hour(m["ts"])))
    f = pd.DataFrame({c: s[~s.index.duplicated()] for c, s in fund.items()})
    m = pd.DataFrame({c: s[~s.index.duplicated()] for c, s in marks.items()})
    common = f.index.intersection(m.index)
    return f.loc[common].sort_index(), m.loc[common].sort_index()


def main() -> None:
    fund, marks = hourly_panel()
    ret = marks.pct_change(fill_method=None)
    # long-perp hourly PnL is price return minus funding; short is the negative
    long_pnl = ret - fund

    hours = fund.index
    step = HOLD_D * 24
    look = LOOKBACK_D * 24
    daily_parts = []
    picks = []
    for i in range(look, len(hours) - 1, step):
        window = fund.iloc[i - look:i]
        alive = window.columns[window.notna().mean() > 0.9]
        if len(alive) < 10:
            continue
        rank = window[alive].mean().sort_values()
        k = max(2, len(alive) // 5)
        lows, highs = list(rank.index[:k]), list(rank.index[-k:])
        seg = (-long_pnl[highs].iloc[i:i + step].mean(axis=1)
               + long_pnl[lows].iloc[i:i + step].mean(axis=1)) / 2
        daily_parts.append(stats.to_daily(seg.dropna()))
        picks.append({"at": f"{hours[i]:%Y-%m-%d}", "short": highs, "long": lows})

    if not daily_parts:
        print("insufficient overlapping history")
        return
    daily = pd.concat(daily_parts)
    gross_sr, gross_ret, _ = stats.annualised_sharpe(daily)
    # full-churn cost bound: both legs re-entered each rebalance
    churn_daily = 2 * COST.round_trip() / (HOLD_D)
    net = daily - churn_daily
    net_sr, net_ret, _ = stats.annualised_sharpe(net)

    span = f"{daily.index.min():%Y-%m-%d}/{daily.index.max():%Y-%m-%d}"
    print("=" * 76)
    print("Perp-vs-perp funding dispersion: short top-quintile funding, long bottom-quintile")
    print(f"  {len(picks)} rebalances, {len(daily)} days, window {span}")
    print("=" * 76)
    for p in picks:
        print(f"  {p['at']}  short {p['short'][:3]}...  long {p['long'][:3]}...")
    print(f"\n  gross      SR {gross_sr:+7.2f}  ret {gross_ret * 100:+7.2f}%")
    print(f"  full-churn SR {net_sr:+7.2f}  ret {net_ret * 100:+7.2f}%   "
          f"(cost {2 * COST.round_trip() * 1e4:.0f} bps per {HOLD_D}-day hold)")

    saved, h = results.record_run(RESULTS_ROOT, "perp_dispersion",
                                  {"lookback_d": LOOKBACK_D, "hold_d": HOLD_D, "window": span},
                                  {"gross_sharpe": round(gross_sr, 3), "gross_ret": round(gross_ret, 5),
                                   "net_sharpe": round(net_sr, 3), "net_ret": round(net_ret, 5),
                                   "rebalances": len(picks)}, span)
    print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}")


if __name__ == "__main__":
    main()
