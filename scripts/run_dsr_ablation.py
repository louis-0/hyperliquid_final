#!/usr/bin/env python3
"""Deflated Sharpe Ratio ablation for the funding-carry chassis, built on the hlq package.

Runs the chassis at three levels of realism on one common window: a funding-only upper bound,
a basis-drift-inclusive rung, and the realistic-cost rung (central scenario: 1 bp/day basis
drift + 1 bp/day spot borrow + standard taker fees). Reports every coin and every basket
composition, then deflates the realistic rung for multiple testing (Bailey and Lopez de Prado
2014); the trial count N and the cross-trial Sharpe variance come from the set of rung x coin
and rung x basket runs.

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

SPOT_COINS = ["BTC", "ETH", "SOL", "HYPE"]       # full-window coins with a Hyperliquid spot pair
DROP = "SOL"                                      # excluded from the deployable (drop-SOL) basket
CENTRAL_DRIFT_BPS_DAY = 1.0                       # R11 central plausible scenario
CENTRAL_BORROW_BPS_DAY = 1.0
COST = CostModel()
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def funding_daily(coin: str) -> pd.Series:
    """Daily funding-only gross series for one coin."""
    pnl = signals.funding_carry_pnl(data.load_funding(coin))
    return stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))


def basis_daily(coin: str) -> pd.Series:
    """Daily basis-drift-inclusive gross series (drift + funding) for one coin."""
    pnl = signals.basis_drift_pnl(data.load_funding(coin), data.load_spot(coin), data.load_marks(coin))
    return stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))


def net(daily_gross: pd.Series, realistic: bool = False) -> pd.Series:
    """Net a gross series: amortised round trip always; central drift+borrow drags when realistic."""
    drag = COST.amortised_daily(len(daily_gross))
    if realistic:
        drag = drag + (CENTRAL_DRIFT_BPS_DAY + CENTRAL_BORROW_BPS_DAY) / 1e4
    return daily_gross - drag


def per_obs_sharpe(daily_net: pd.Series) -> float:
    """Per-observation Sharpe of a netted series, the unit collected into the trial set."""
    return stats.psr_inputs(daily_net)[0]


def rung_sharpes(funding_gross: pd.Series, basis_gross: pd.Series) -> tuple[float, float, float]:
    """Annualised Sharpe at each rung: funding-only, basis-drift, and realistic-cost."""
    return (stats.annualised_sharpe(net(funding_gross))[0],
            stats.annualised_sharpe(net(basis_gross))[0],
            stats.annualised_sharpe(net(basis_gross, realistic=True))[0])


def realistic_dsr(basis_gross: pd.Series, n_trials: int, var_sr: float) -> float:
    """DSR of the realistic-cost rung for a coin or basket."""
    sr, T, skew, kurt = stats.psr_inputs(net(basis_gross, realistic=True))
    return stats.deflated_sharpe(sr, T, skew, kurt, n_trials, var_sr)


def main() -> None:
    # Per-coin series on the basis-drift (spot-limited) window; funding-only is reindexed to it.
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

    def basket(series: dict[str, pd.Series], exclude=()) -> pd.Series:
        return portfolio.equal_weight_basket(series, exclude=exclude)

    # Trial set for the deflation: every rung x coin and rung x (all-coin, drop-SOL) basket.
    trials: list[float] = []
    for coin in basis:
        trials.append(per_obs_sharpe(net(funding[coin])))
        trials.append(per_obs_sharpe(net(basis[coin])))
        trials.append(per_obs_sharpe(net(basis[coin], realistic=True)))
    for exclude in ((), (DROP,)):
        f_bk, b_bk = basket(funding, exclude), basket(basis, exclude)
        trials.append(per_obs_sharpe(net(f_bk)))
        trials.append(per_obs_sharpe(net(b_bk)))
        trials.append(per_obs_sharpe(net(b_bk, realistic=True)))
    N = len(trials)
    var_sr = float(np.var(np.asarray(trials, dtype=float), ddof=1))

    # ZEC has a shorter spot window; it enters the per-coin panel and the all-5 basket only.
    per_coin_f, per_coin_b = dict(funding), dict(basis)
    try:
        zb = basis_daily("ZEC")
        per_coin_b["ZEC"] = zb
        per_coin_f["ZEC"] = funding_daily("ZEC").reindex(zb.index).dropna()
    except FileNotFoundError:
        print("  ZEC: no spot series, omitted from the panel")

    print("=" * 68)
    print("Deflated Sharpe ablation: funding-carry chassis")
    print(f"  trials N={N}; cross-trial Sharpe variance var_sr={var_sr:.5f}; clears at DSR > 0.95")
    print("=" * 68)

    print(f"\n{'coin':<6} {'T':>4} {'funding':>9} {'basis':>8} {'realistic':>10}")
    print("-" * 42)
    for coin in per_coin_b:
        fs, bs, rs = rung_sharpes(per_coin_f[coin], per_coin_b[coin])
        print(f"  {coin:<4} {len(per_coin_b[coin]):>4} {fs:>9.2f} {bs:>8.2f} {rs:>10.2f}")

    baskets = {"drop-SOL": (basket(funding, (DROP,)), basket(basis, (DROP,))),
               "all-4": (basket(funding), basket(basis))}
    if "ZEC" in per_coin_b:
        baskets["all-5 (+ZEC)"] = (portfolio.equal_weight_basket(per_coin_f),
                                   portfolio.equal_weight_basket(per_coin_b))

    print(f"\n{'basket':<14} {'T':>4} {'funding':>9} {'basis':>8} {'realistic':>10} {'DSR':>8}")
    print("-" * 58)
    payload: dict[str, dict] = {}
    for name, (f_bk, b_bk) in baskets.items():
        fs, bs, rs = rung_sharpes(f_bk, b_bk)
        dsr = realistic_dsr(b_bk, N, var_sr)
        print(f"  {name:<12} {len(b_bk):>4} {fs:>9.2f} {bs:>8.2f} {rs:>10.2f} {dsr:>8.3f}")
        payload[name] = {"realistic_sharpe": round(rs, 4), "dsr": round(dsr, 4)}

    f_drop, b_drop = baskets["drop-SOL"]
    print("\nDSR across trial counts N:")
    for label, gross, realistic in (("funding_only", f_drop, False), ("realistic", b_drop, True)):
        sr, T, skew, kurt = stats.psr_inputs(net(gross, realistic))
        row = "  ".join(f"N={n}:{stats.deflated_sharpe(sr, T, skew, kurt, n, var_sr):.3f}" for n in (N, 25, 60))
        print(f"  {label:<14} {row}")

    span = f"{b_drop.index.min():%Y-%m-%d}/{b_drop.index.max():%Y-%m-%d}"
    saved, h = results.record_run(RESULTS_ROOT, "dsr_ablation",
                                  {"coins": list(per_coin_b), "drop": DROP, "n_trials": N}, payload, span)
    print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}  window {span}")


if __name__ == "__main__":
    main()
