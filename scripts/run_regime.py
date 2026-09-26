#!/usr/bin/env python3
"""Regime-conditional basis-drift Sharpe, built on the hlq package.

Classifies each UTC day as bear, calm, or bull from BTC's 30-day rolling return
(hlq.signals.regime_label), then slices each coin's basis-drift daily net return by regime and
reports the per-regime annualised Sharpe and return. Coverage is the coins with a Hyperliquid
spot pair (BTC, ETH, SOL, HYPE, ZEC); a coin's day count per regime follows its own spot window.

    python scripts/run_regime.py
    python scripts/run_regime.py BTC ETH
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, results, signals, stats
from hlq.costs import CostModel

CHASSIS = ["BTC", "ETH", "SOL", "HYPE", "ZEC"]
COST = CostModel()
REGIMES = ["bear", "calm", "bull"]
MIN_DAYS = 10
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def coin_daily_net(coin: str) -> pd.Series:
    """Daily net basis-drift return for one coin, indexed by UTC day."""
    pnl = signals.basis_drift_pnl(data.load_funding(coin), data.load_spot(coin), data.load_marks(coin))
    daily = stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))
    daily.index = daily.index.floor("1D")
    return daily - COST.amortised_daily(len(daily))


def main(coins: list[str]) -> None:
    regime = signals.regime_label(data.load_marks("BTC"))
    counts = regime.value_counts()
    print("=" * 74)
    print("Regime-conditional basis-drift Sharpe (long-spot + short-perp, net of cost)")
    print("  classifier: 30-day rolling BTC return; bear below -10%, bull above +10%")
    print(f"  regime days: bear {int(counts.get('bear', 0))}, "
          f"calm {int(counts.get('calm', 0))}, bull {int(counts.get('bull', 0))}")
    print("=" * 74)
    print(f"\n{'coin':<6} {'regime':<6} {'n_days':>7} {'SR':>9} {'ann_ret%':>10} {'ann_vol%':>10}")
    print("-" * 74)

    table: dict[str, dict[str, float]] = {}
    for coin in coins:
        try:
            daily = coin_daily_net(coin)
        except FileNotFoundError:
            print(f"  {coin:<6} (no spot/perp/funding)")
            continue
        joined = pd.DataFrame({"pnl": daily}).join(regime, how="inner")
        for r in REGIMES:
            sl = joined.loc[joined["regime"] == r, "pnl"]
            if len(sl) < MIN_DAYS:
                print(f"  {coin:<6} {r:<6} {len(sl):>7} {'n/a':>9} {'n/a':>10} {'n/a':>10}")
                continue
            sr, ret, vol = stats.annualised_sharpe(sl)
            table.setdefault(coin, {})[r] = round(sr, 6)
            print(f"  {coin:<6} {r:<6} {len(sl):>7} {sr:>9.3f} {ret*100:>9.2f}% {vol*100:>9.2f}%")
        print()

    if table:
        span = f"{regime.index.min():%Y-%m-%d}/{regime.index.max():%Y-%m-%d}"
        saved, h = results.record_run(RESULTS_ROOT, "regime_basis_drift",
                                      {"coins": list(table), "bear": -0.10, "bull": 0.10, "window": span},
                                      {"regime_sharpe": table}, span)
        print(f"[results] {'recorded' if saved else 'already recorded'} {h}")


if __name__ == "__main__":
    main(sys.argv[1:] or CHASSIS)