#!/usr/bin/env python3
"""Strategy-versus-baselines comparison on one common window, built on the hlq package.

Puts the timed strategy's rungs next to the pre-registered baselines on the same daily
window, each netted by one amortised round-trip cost: buy-and-hold BTC (spot closes),
an equal-weight buy-and-hold basket (equal notional in spot BTC, ETH, SOL, and HYPE,
rebalanced daily), and the naive funding-carry chassis without timing. The HLP vault
baseline is not computed; its net-asset-value history is not captured. Every row is
deflated at the trial count and cross-trial variance of the rung x coin and rung x
basket search set, so the comparison and the ablation share one benchmark.

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
CENTRAL_DRIFT_BPS_DAY = 1.0
CENTRAL_BORROW_BPS_DAY = 1.0
COST = CostModel()
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def funding_daily(coin: str) -> pd.Series:
    pnl = signals.funding_carry_pnl(data.load_funding(coin))
    return stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))


def basis_daily(coin: str) -> pd.Series:
    pnl = signals.basis_drift_pnl(data.load_funding(coin), data.load_spot(coin), data.load_marks(coin))
    return stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))


def net(daily_gross: pd.Series, realistic: bool = False) -> pd.Series:
    drag = COST.amortised_daily(len(daily_gross))
    if realistic:
        drag = drag + (CENTRAL_DRIFT_BPS_DAY + CENTRAL_BORROW_BPS_DAY) / 1e4
    return daily_gross - drag


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

    # Deflation benchmark: same search set as the ablation (rung x coin, rung x basket).
    trials: list[float] = []
    for coin in basis:
        trials.append(stats.psr_inputs(net(funding[coin]))[0])
        trials.append(stats.psr_inputs(net(basis[coin]))[0])
        trials.append(stats.psr_inputs(net(basis[coin], realistic=True))[0])
    for exclude in ((), (DROP,)):
        f_bk, b_bk = basket(funding, exclude), basket(basis, exclude)
        trials.append(stats.psr_inputs(net(f_bk))[0])
        trials.append(stats.psr_inputs(net(b_bk))[0])
        trials.append(stats.psr_inputs(net(b_bk, realistic=True))[0])
    n_trials = len(trials)
    var_sr = float(np.var(np.asarray(trials, dtype=float), ddof=1))

    rows = {
        "buy-and-hold BTC": net(bh["BTC"]),
        "equal-weight spot": net(basket(bh)),
        "naive carry (drop-SOL)": net(basket(funding, (DROP,))),
        "basis-drift (drop-SOL)": net(basket(basis, (DROP,))),
        "realistic (drop-SOL)": net(basket(basis, (DROP,)), realistic=True),
    }

    print("=" * 72)
    print("Strategy versus baselines: one common window, one amortised round trip")
    print(f"  trials N={n_trials}; cross-trial Sharpe variance var_sr={var_sr:.5f}; clears at DSR > 0.95")
    print("=" * 72)
    print(f"\n{'series':<24} {'T':>4} {'ann_Sharpe':>11} {'ann_ret%':>10} {'DSR':>8}")
    print("-" * 72)
    payload: dict[str, dict] = {}
    for name, daily_net in rows.items():
        sr, T, skew, kurt = stats.psr_inputs(daily_net)
        ann_sr, ann_ret, _ = stats.annualised_sharpe(daily_net)
        dsr = stats.deflated_sharpe(sr, T, skew, kurt, n_trials, var_sr)
        print(f"  {name:<22} {T:>4} {ann_sr:>11.2f} {ann_ret * 100:>9.2f}% {dsr:>8.3f}")
        payload[name] = {"ann_sharpe": round(ann_sr, 4), "ann_ret": round(ann_ret, 6), "dsr": round(dsr, 4)}

    span = f"{common.min():%Y-%m-%d}/{common.max():%Y-%m-%d}"
    saved, h = results.record_run(RESULTS_ROOT, "baselines_head_to_head",
                                  {"coins": SPOT_COINS, "drop": DROP, "window": span,
                                   "n_trials": n_trials}, payload, span)
    print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}  window {span}")


if __name__ == "__main__":
    main()
