#!/usr/bin/env python3
"""Realistic-cost sensitivity sweep for the basis-drift basket, built on the hlq package.

Sweeps basis-drift and spot-borrow drags with a fee multiplier over the drop-SOL basket's
daily series (fifty configurations), reporting the annualised Sharpe and return per cell,
the share of the plausible zone that is net-positive, and a block-bootstrap confidence
interval (Kunsch 1989) for the central scenario.

    python scripts/run_cost_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, portfolio, results, signals, stats
from hlq.costs import CostModel

SPOT_COINS = ["BTC", "ETH", "SOL", "HYPE"]
DROP = "SOL"
DRIFT_BPS = [0.0, 0.5, 1.0, 2.0, 5.0]
BORROW_BPS = [0.0, 0.5, 1.0, 2.0, 5.0]
FEE_MULT = [1.0, 2.0]
PLAUSIBLE = 2.0            # drags at or below this bound form the plausible zone
COST = CostModel()
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def basis_daily(coin: str) -> pd.Series:
    pnl = signals.basis_drift_pnl(data.load_funding(coin), data.load_spot(coin), data.load_marks(coin))
    return stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))


def main() -> None:
    basket = portfolio.equal_weight_basket({c: basis_daily(c) for c in SPOT_COINS}, exclude=(DROP,))
    n = len(basket)
    cells = []
    for fee_mult in FEE_MULT:
        for drift in DRIFT_BPS:
            for borrow in BORROW_BPS:
                daily = basket - fee_mult * COST.round_trip() / n - (drift + borrow) / 1e4
                sr, ret, _ = stats.annualised_sharpe(daily)
                cells.append({"drift": drift, "borrow": borrow, "fee_mult": fee_mult,
                              "sharpe": round(sr, 3), "ann_ret": round(ret, 5)})
    df = pd.DataFrame(cells)
    plaus = df[(df["drift"] <= PLAUSIBLE) & (df["borrow"] <= PLAUSIBLE) & (df["fee_mult"] == 1.0)]
    central = df[(df["drift"] == 1.0) & (df["borrow"] == 1.0) & (df["fee_mult"] == 1.0)].iloc[0]
    zero = df[(df["drift"] == 0.0) & (df["borrow"] == 0.0) & (df["fee_mult"] == 1.0)].iloc[0]
    stress = df[(df["drift"] == 5.0) & (df["borrow"] == 5.0) & (df["fee_mult"] == 2.0)].iloc[0]
    central_daily = basket - COST.round_trip() / n - 2.0 / 1e4
    lo, hi = stats.block_bootstrap_ci(central_daily)

    span = f"{basket.index.min():%Y-%m-%d}/{basket.index.max():%Y-%m-%d}"
    print("=" * 76)
    print(f"Cost sensitivity sweep: drop-{DROP} basis-drift basket, {len(df)} configurations")
    print(f"  window {span} ({n} days)")
    print("=" * 76)
    print(f"  zero-drag      SR {zero['sharpe']:+8.2f}  ret {zero['ann_ret'] * 100:+7.2f}%")
    print(f"  central        SR {central['sharpe']:+8.2f}  ret {central['ann_ret'] * 100:+7.2f}%"
          f"   95% CI on SR [{lo:+.2f}, {hi:+.2f}]")
    print(f"  stress         SR {stress['sharpe']:+8.2f}  ret {stress['ann_ret'] * 100:+7.2f}%")
    pos = int((plaus["sharpe"] > 0).sum())
    print(f"  plausible zone: {pos} of {len(plaus)} configurations net-positive"
          f" (median SR {plaus['sharpe'].median():+.2f})")

    payload = {"central": {"sharpe": float(central["sharpe"]), "ann_ret": float(central["ann_ret"]),
                           "ci": [round(lo, 3), round(hi, 3)]},
               "zero_cost": {"sharpe": float(zero["sharpe"])},
               "stress": {"sharpe": float(stress["sharpe"])},
               "plausible_positive": pos, "plausible_total": int(len(plaus)),
               "cells": cells}
    saved, h = results.record_run(RESULTS_ROOT, "cost_sweep",
                                  {"coins": SPOT_COINS, "drop": DROP, "window": span}, payload, span)
    print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}")


if __name__ == "__main__":
    main()
