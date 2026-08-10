#!/usr/bin/env python3
"""Basis-drift backtest, built on the hlq package.

Long-spot / short-perp with the full PnL, not just funding: the residual price drift between
the two legs (the hedge is not perfectly delta-neutral) plus the funding accrued. The drift
term carries most of the variance and cuts the Sharpe well below the funding-only upper bound.
Coverage is the coins with a Hyperliquid spot pair (BTC, ETH, SOL, HYPE); NEAR has no spot,
so the basket excludes it by construction.

As in run_chassis.py, the basket is the intersection-window equal-weight mean (hlq.portfolio)
and the bootstrap CI uses a real block length (hlq.stats).

    python scripts/run_basis_drift.py
    python scripts/run_basis_drift.py BTC ETH
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, portfolio, signals, stats
from hlq.costs import CostModel

CHASSIS = ["BTC", "ETH", "SOL", "HYPE"]      # have Hyperliquid spot; NEAR excluded (no spot)
COST = CostModel()


def coin_daily_gross(coin: str) -> pd.Series:
    """Daily gross basis-drift-inclusive return series (drift + funding) for one coin."""
    pnl = signals.basis_drift_pnl(data.load_funding(coin), data.load_spot(coin), data.load_marks(coin))
    return stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))


def summarise(daily_gross: pd.Series) -> dict:
    daily_net = daily_gross - COST.amortised_daily(len(daily_gross))
    sr_net, ann_ret, ann_vol = stats.annualised_sharpe(daily_net)
    sr_gross, _, _ = stats.annualised_sharpe(daily_gross)
    lo, hi = stats.block_bootstrap_ci(daily_net)
    return {"n": len(daily_gross), "sr_gross": sr_gross, "sr_net": sr_net,
            "ann_ret": ann_ret, "ann_vol": ann_vol, "ci_lo": lo, "ci_hi": hi}


def main(coins: list[str]) -> None:
    print("=" * 88)
    print("Basis-drift backtest: long-spot + short-perp, full PnL (drift + funding)")
    print(f"  round-trip cost {COST.round_trip()*100:.2f}%; 95% block-bootstrap CI")
    print("=" * 88)
    print(f"\n{'coin':<6} {'n_days':>6} {'gross_SR':>10} {'net_SR':>9} "
          f"{'ann_ret%':>10} {'ann_vol%':>10} {'95% CI':>18}")
    print("-" * 88)

    series: dict[str, pd.Series] = {}
    for coin in coins:
        try:
            series[coin] = coin_daily_gross(coin)
        except FileNotFoundError:
            print(f"  {coin:<6} (no spot/perp/funding)")
            continue
        r = summarise(series[coin])
        print(f"  {coin:<6} {r['n']:>6} {r['sr_gross']:>10.3f} {r['sr_net']:>9.3f} "
              f"{r['ann_ret']*100:>9.2f}% {r['ann_vol']*100:>9.2f}% "
              f"[{r['ci_lo']:>+5.2f}, {r['ci_hi']:>+5.2f}]")

    if series:
        b = summarise(portfolio.equal_weight_basket(series))
        print("-" * 88)
        print(f"  {'basket':<6} {b['n']:>6} {'':>10} {b['sr_net']:>9.3f} "
              f"{b['ann_ret']*100:>9.2f}% {b['ann_vol']*100:>9.2f}% "
              f"[{b['ci_lo']:>+5.2f}, {b['ci_hi']:>+5.2f}]")
        print(f"  (equal-weight over the common window: {list(series)})")


if __name__ == "__main__":
    main(sys.argv[1:] or CHASSIS)
