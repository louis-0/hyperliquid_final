#!/usr/bin/env python3
"""Strategy-versus-baselines comparison on one common window, built on the hlq package.

Puts the timed strategy's rungs next to the pre-registered baselines on the same daily
window, each netted by one amortised round-trip cost: buy-and-hold BTC (spot closes),
an equal-weight buy-and-hold basket (equal notional in spot BTC, ETH, SOL, and HYPE,
rebalanced daily), and the naive funding-carry chassis without timing. The HLP vault
baseline is not computed; its net-asset-value history is not captured. Every row is
deflated twice: at the trial count and cross-trial variance of the four rungs (funding-only,
basis-drift, realistic, realistic-B) over every coin and basket on this window, the
ablation's search set, and at those of the hedged rungs alone.

    python scripts/run_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import baselines, data, portfolio, results, signals, stats
from hlq.costs import CostModel

SPOT_COINS = ["BTC", "ETH", "SOL", "HYPE"]
DROP = "SOL"
CENTRAL_BPS_DAY = 2.0          # central scenario: 1 bp/day parametric drift + 1 bp/day borrow
BORROW_ONLY_BPS_DAY = 1.0      # realistic-B: borrow only
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


def main() -> None:
    basis = {c: basis_daily(c) for c in SPOT_COINS}
    funding = {c: funding_daily(c).reindex(basis[c].index).dropna() for c in SPOT_COINS}
    bh = {c: baselines.buy_and_hold_daily(data.load_spot(c)) for c in SPOT_COINS}

    # One shared daily window across every series in the comparison.
    common = basis["BTC"].index
    for s in list(basis.values()) + list(funding.values()) + list(bh.values()):
        common = common.intersection(s.index)
    basis = {c: s.reindex(common) for c, s in basis.items()}
    funding = {c: s.reindex(common) for c, s in funding.items()}
    bh = {c: s.reindex(common) for c, s in bh.items()}

    def basket(series: dict[str, pd.Series], exclude=()) -> pd.Series:
        return portfolio.equal_weight_basket(series, exclude=exclude)

    # Two deflation benchmarks: the four rungs of every coin and basket (the ablation's search
    # set on this window), and the hedged rungs alone (basis-drift, realistic, realistic-B).
    all_rungs: list[float] = []
    hedged: list[float] = []
    for coin in basis:
        all_rungs.append(stats.psr_inputs(net(funding[coin]))[0])
        for drag in (0.0, CENTRAL_BPS_DAY, BORROW_ONLY_BPS_DAY):
            sr = stats.psr_inputs(net(basis[coin], drag))[0]
            all_rungs.append(sr)
            hedged.append(sr)
    for exclude in ((), (DROP,)):
        f_bk, b_bk = basket(funding, exclude), basket(basis, exclude)
        all_rungs.append(stats.psr_inputs(net(f_bk))[0])
        for drag in (0.0, CENTRAL_BPS_DAY, BORROW_ONLY_BPS_DAY):
            sr = stats.psr_inputs(net(b_bk, drag))[0]
            all_rungs.append(sr)
            hedged.append(sr)
    n_all = len(all_rungs)
    var_all = float(np.var(np.asarray(all_rungs, dtype=float), ddof=1))
    n_hedged = len(hedged)
    var_hedged = float(np.var(np.asarray(hedged, dtype=float), ddof=1))

    rows = {
        "buy-and-hold BTC": net(bh["BTC"]),
        "equal-weight spot": net(basket(bh)),
        "naive carry (drop-SOL)": net(basket(funding, (DROP,))),
        "basis-drift (drop-SOL)": net(basket(basis, (DROP,))),
        "realistic (drop-SOL)": net(basket(basis, (DROP,)), CENTRAL_BPS_DAY),
        "realistic-B (drop-SOL)": net(basket(basis, (DROP,)), BORROW_ONLY_BPS_DAY),
    }

    print("=" * 84)
    print("Strategy versus baselines: one common window, one amortised round trip")
    print(f"  all rungs N={n_all}, var_sr={var_all:.5f}; hedged rungs N={n_hedged}, var_sr={var_hedged:.5f}; "
          f"clears at DSR > 0.95")
    print("=" * 84)
    print(f"\n{'series':<24} {'T':>4} {'ann_Sharpe':>11} {'ann_ret%':>10} {'DSR_all':>8} {'DSR_hedged':>11}")
    print("-" * 84)
    payload: dict[str, dict] = {}
    for name, daily_net in rows.items():
        sr, T, skew, kurt = stats.psr_inputs(daily_net)
        ann_sr, ann_ret, _ = stats.annualised_sharpe(daily_net)
        d_all = stats.deflated_sharpe(sr, T, skew, kurt, n_all, var_all)
        d_hedged = stats.deflated_sharpe(sr, T, skew, kurt, n_hedged, var_hedged)
        print(f"  {name:<22} {T:>4} {ann_sr:>11.2f} {ann_ret * 100:>9.2f}% {d_all:>8.3f} {d_hedged:>11.3f}")
        payload[name] = {"ann_sharpe": round(ann_sr, 4), "ann_ret": round(ann_ret, 6),
                         "dsr_all": round(d_all, 4), "dsr_hedged": round(d_hedged, 4)}

    span = f"{common.min():%Y-%m-%d}/{common.max():%Y-%m-%d}"
    saved, h = results.record_run(RESULTS_ROOT, "baselines_head_to_head",
                                  {"coins": SPOT_COINS, "drop": DROP, "window": span,
                                   "n_trials": {"all": n_all, "hedged": n_hedged}}, payload, span)
    print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}  window {span}")


if __name__ == "__main__":
    main()
