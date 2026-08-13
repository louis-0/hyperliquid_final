#!/usr/bin/env python3
"""Funding-carry chassis backtest, built on the hlq package.

Always-on long-spot / short-perp, funding accrual only (a perfect hedge is assumed, so the
per-hour PnL is the funding rate the short-perp leg receives). Daily-summed, netted by an
amortised round-trip cost, reported as an annualised Sharpe with a moving-block bootstrap CI.
Anchor: He et al. (2024), BTC Sharpe 1.8 retail / 3.5 market-maker.

The basket is the equal-weight mean over the coins' *common* window (hlq.portfolio, an
intersection so a longer-history coin cannot dominate the days it alone covers), and the
bootstrap CI uses a real block length (hlq.stats, block=10) so the interval is not understated
for an autocorrelated series.

    python scripts/run_chassis.py            # default chassis
    python scripts/run_chassis.py BTC ETH
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, portfolio, results, signals, stats
from hlq.costs import CostModel

CHASSIS = ["BTC", "ETH", "SOL", "NEAR", "HYPE"]
COST = CostModel()
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def coin_daily_gross(coin: str) -> pd.Series:
    """Daily gross funding-carry return series for one coin."""
    pnl = signals.funding_carry_pnl(data.load_funding(coin))
    return stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))


def summarise(daily_gross: pd.Series) -> dict:
    """Net the amortised round-trip cost over the hold and annualise."""
    daily_net = daily_gross - COST.amortised_daily(len(daily_gross))
    sr_net, ann_ret, ann_vol = stats.annualised_sharpe(daily_net)
    sr_gross, _, _ = stats.annualised_sharpe(daily_gross)
    lo, hi = stats.block_bootstrap_ci(daily_net)
    return {"n": len(daily_gross), "sr_gross": sr_gross, "sr_net": sr_net,
            "ann_ret": ann_ret, "ann_vol": ann_vol, "ci_lo": lo, "ci_hi": hi}


def main(coins: list[str]) -> None:
    print("=" * 88)
    print("Funding-carry chassis: always-on long-spot + short-perp, funding accrual only")
    print(f"  round-trip cost {COST.round_trip()*100:.2f}%; 95% block-bootstrap CI")
    print("=" * 88)
    print(f"\n{'coin':<6} {'n_days':>6} {'gross_SR':>10} {'net_SR':>9} "
          f"{'ann_ret%':>10} {'ann_vol%':>10} {'95% CI':>18}")
    print("-" * 88)

    series: dict[str, pd.Series] = {}
    net: dict[str, float] = {}
    for coin in coins:
        try:
            series[coin] = coin_daily_gross(coin)
        except FileNotFoundError:
            print(f"  {coin:<6} (no data)")
            continue
        r = summarise(series[coin])
        net[coin] = round(r["sr_net"], 6)
        print(f"  {coin:<6} {r['n']:>6} {r['sr_gross']:>10.3f} {r['sr_net']:>9.3f} "
              f"{r['ann_ret']*100:>9.2f}% {r['ann_vol']*100:>9.2f}% "
              f"[{r['ci_lo']:>+5.2f}, {r['ci_hi']:>+5.2f}]")

    if series:
        basket = portfolio.equal_weight_basket(series)
        b = summarise(basket)
        print("-" * 88)
        print(f"  {'basket':<6} {b['n']:>6} {'':>10} {b['sr_net']:>9.3f} "
              f"{b['ann_ret']*100:>9.2f}% {b['ann_vol']*100:>9.2f}% "
              f"[{b['ci_lo']:>+5.2f}, {b['ci_hi']:>+5.2f}]")
        print(f"  (equal-weight over the common window: {list(series)})")
        span = f"{basket.index.min():%Y-%m-%d}/{basket.index.max():%Y-%m-%d}"
        saved, h = results.record_run(RESULTS_ROOT, "funding_carry_chassis",
                                      {"coins": list(series), "round_trip": COST.round_trip()},
                                      {"net_sharpe": net, "basket_net_sharpe": round(b["sr_net"], 6)}, span)
        print(f"  [results] {'recorded' if saved else 'already recorded'} {h}")

    print("\nHe et al. (2024) anchor: BTC Sharpe 1.8 retail / 3.5 market-maker.")


if __name__ == "__main__":
    main(sys.argv[1:] or CHASSIS)
