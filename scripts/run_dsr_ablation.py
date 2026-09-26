#!/usr/bin/env python3
"""Deflated Sharpe Ratio ablation for the funding-carry chassis, built on the hlq package.

Runs the chassis at four levels of realism on one common window: a funding-only upper bound,
a basis-drift-inclusive rung, a realistic rung charging the full central scenario (1 bp/day
parametric drift + 1 bp/day spot borrow + taker fees), and a realistic-B rung charging fees
and borrow only, since the basis-drift series already carries the realised drift. Reports
every coin and basket composition, deflates each realistic rung at the trial count and
cross-trial variance of the full rung x coin and rung x basket search set, and quotes a
block-bootstrap confidence interval for the deployable rungs.

    python scripts/run_dsr_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, portfolio, results, signals, stats
from hlq.costs import CostModel

SPOT_COINS = ["BTC", "ETH", "SOL", "HYPE", "ZEC"]
DROP = "SOL"
REALISTIC_BPS = 2.0        # central scenario: 1 bp/day parametric drift + 1 bp/day borrow
BORROW_ONLY_BPS = 1.0      # realistic-B: borrow only; realised drift already sits in the PnL
COST = CostModel()
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def funding_daily(coin: str) -> pd.Series:
    pnl = signals.funding_carry_pnl(data.load_funding(coin))
    return stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))


def basis_daily(coin: str) -> pd.Series:
    pnl = signals.basis_drift_pnl(data.load_funding(coin), data.load_spot(coin), data.load_marks(coin))
    return stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))


def net(daily_gross: pd.Series, drag_bps: float = 0.0) -> pd.Series:
    """Net of the amortised round trip plus a per-day drag in basis points."""
    return daily_gross - COST.amortised_daily(len(daily_gross)) - drag_bps / 1e4


def rung_sharpes(f: pd.Series, b: pd.Series) -> tuple[float, float, float, float]:
    """Annualised Sharpe at each rung: funding-only, basis, realistic, realistic-B."""
    return (stats.annualised_sharpe(net(f))[0],
            stats.annualised_sharpe(net(b))[0],
            stats.annualised_sharpe(net(b, REALISTIC_BPS))[0],
            stats.annualised_sharpe(net(b, BORROW_ONLY_BPS))[0])


def dsr_of(daily_net: pd.Series, n_trials: int, var_sr: float) -> float:
    sr, T, skew, kurt = stats.psr_inputs(daily_net)
    return stats.deflated_sharpe(sr, T, skew, kurt, n_trials, var_sr)


def main() -> None:
    basis: dict[str, pd.Series] = {}
    funding: dict[str, pd.Series] = {}
    for coin in SPOT_COINS:
        try:
            b = basis_daily(coin)
        except FileNotFoundError:
            print(f"  {coin}: no spot/perp/funding, skipped")
            continue
        basis[coin] = b
        funding[coin] = funding_daily(coin).reindex(b.index).dropna()

    def basket(names) -> tuple[pd.Series, pd.Series]:
        return (portfolio.equal_weight_basket({c: funding[c] for c in names}),
                portfolio.equal_weight_basket({c: basis[c] for c in names}))

    full = [c for c in basis if c != "ZEC"]
    baskets = {"drop-SOL": basket([c for c in full if c != DROP]),
               "all-4": basket(full),
               "HYPE-only": basket(["HYPE"])}
    if "ZEC" in basis:
        baskets["all-5 (+ZEC)"] = basket(list(basis))

    # Trial set: every rung of every coin and every basket computed here.
    trials: list[float] = []
    for coin in basis:
        for drag in (None, 0.0, REALISTIC_BPS, BORROW_ONLY_BPS):
            s = net(funding[coin]) if drag is None else net(basis[coin], drag)
            trials.append(stats.psr_inputs(s)[0])
    for f_bk, b_bk in baskets.values():
        for drag in (None, 0.0, REALISTIC_BPS, BORROW_ONLY_BPS):
            s = net(f_bk) if drag is None else net(b_bk, drag)
            trials.append(stats.psr_inputs(s)[0])
    N = len(trials)
    var_sr = float(np.var(np.asarray(trials, dtype=float), ddof=1))

    print("=" * 76)
    print("Deflated Sharpe ablation: funding-carry chassis")
    print(f"  trials N={N}; cross-trial Sharpe variance var_sr={var_sr:.5f}; clears at DSR > 0.95")
    print("=" * 76)
    print(f"\n{'coin':<6} {'T':>4} {'funding':>9} {'basis':>8} {'realistic':>10} {'realistic-B':>12}")
    print("-" * 56)
    for coin in basis:
        fs, bs, rs, rb = rung_sharpes(funding[coin], basis[coin])
        print(f"  {coin:<4} {len(basis[coin]):>4} {fs:>9.2f} {bs:>8.2f} {rs:>10.2f} {rb:>12.2f}")

    print(f"\n{'basket':<14} {'T':>4} {'funding':>9} {'basis':>8} {'realistic':>10} {'DSR':>7} "
          f"{'realistic-B':>12} {'DSR-B':>7}")
    print("-" * 78)
    payload: dict[str, dict] = {}
    for name, (f_bk, b_bk) in baskets.items():
        fs, bs, rs, rb = rung_sharpes(f_bk, b_bk)
        d_r = dsr_of(net(b_bk, REALISTIC_BPS), N, var_sr)
        d_b = dsr_of(net(b_bk, BORROW_ONLY_BPS), N, var_sr)
        print(f"  {name:<12} {len(b_bk):>4} {fs:>9.2f} {bs:>8.2f} {rs:>10.2f} {d_r:>7.3f} "
              f"{rb:>12.2f} {d_b:>7.3f}")
        payload[name] = {"funding": round(fs, 4), "basis": round(bs, 4),
                         "realistic": round(rs, 4), "dsr": round(d_r, 4),
                         "realistic_b": round(rb, 4), "dsr_b": round(d_b, 4)}

    f_drop, b_drop = baskets["drop-SOL"]
    for label, drag in (("realistic", REALISTIC_BPS), ("realistic-B", BORROW_ONLY_BPS)):
        lo, hi = stats.block_bootstrap_ci(net(b_drop, drag))
        print(f"  drop-SOL {label:<12} 95% CI on SR [{lo:+.2f}, {hi:+.2f}]")

    span = f"{b_drop.index.min():%Y-%m-%d}/{b_drop.index.max():%Y-%m-%d}"
    saved, h = results.record_run(RESULTS_ROOT, "dsr_ablation",
                                  {"coins": list(basis), "drop": DROP, "window": span,
                                   "n_trials": N, "var_sr": round(var_sr, 6)}, payload, span)
    print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}  window {span}")


if __name__ == "__main__":
    main()
