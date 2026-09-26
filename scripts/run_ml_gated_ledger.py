#!/usr/bin/env python3
"""Per-trade ledger for the ML-gated carry, built on the hlq package.

Trades the chassis on the funding-sign classifier's out-of-sample calls: hold the hedged
position through every run of consecutive predicted-positive hours, exit when the model
predicts negative funding, re-enter on the next positive call. Each round trip is debited
the taker fee on all four legs; funding is accrued from the realised series while held.
The always-on chassis on the same window, paying one round trip, is printed alongside, and each
coin's trades are written to results/ml_gated_ledger_{coin}.csv.

    python scripts/run_ml_gated_ledger.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, features, results, timing
from hlq.costs import CostModel

COINS = ["BTC", "ETH", "HYPE"]
COST = CostModel()
RT_COST = 4 * COST.taker_fee            # both legs, both ways
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def gated_trades(df: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    """Out-of-sample trades from the sign model's calls, plus always-on funding and hours."""
    clf, train, test = timing.train_sign_model(df)
    proba = clf.predict_proba(test[features.FEATURES])[:, 1]
    pred = (proba > 0.5).astype(int)
    fund = test["target_fund"].to_numpy()          # realised funding for the predicted hour
    hours = pd.to_datetime(test["hour"]).to_numpy()

    rows = []
    i = 0
    n = len(pred)
    while i < n:
        if pred[i] == 0:
            i += 1
            continue
        j = i
        while j < n and pred[j] == 1:
            j += 1
        take = float(fund[i:j].sum())
        rows.append({"entry": pd.Timestamp(hours[i]), "exit": pd.Timestamp(hours[j - 1]),
                     "hold_h": j - i, "funding_bps": take * 1e4,
                     "cost_bps": RT_COST * 1e4, "net_bps": (take - RT_COST) * 1e4})
        i = j
    return pd.DataFrame(rows), float(fund.sum()), n


def main() -> None:
    print("=" * 88)
    print("ML-gated carry ledger: hold while the funding-sign model predicts positive, out of sample")
    print(f"  round trip {RT_COST * 1e4:.0f} bps per re-entry (four legs); always-on pays it once")
    print("=" * 88)

    summary: dict[str, dict] = {}
    span = None
    for coin in COINS:
        df = features.build_features(data.load_funding(coin), data.load_spot(coin), data.load_marks(coin))
        trades, fund_all, n_hours = gated_trades(df)
        if trades.empty:
            print(f"  {coin}: no trades")
            continue
        trades.to_csv(RESULTS_ROOT / f"ml_gated_ledger_{coin}.csv", index=False)
        gated_net = float(trades["net_bps"].sum())
        always_net = (fund_all - RT_COST) * 1e4
        print(f"\n{coin}: {len(trades)} trades over {n_hours} out-of-sample hours")
        print(trades.head(6).to_string(index=False,
              formatters={"funding_bps": "{:+.2f}".format, "cost_bps": "{:.1f}".format,
                          "net_bps": "{:+.2f}".format}))
        print(f"  gated net {gated_net:+.1f}bp ({len(trades)} round trips)  vs  "
              f"always-on net {always_net:+.1f}bp (one round trip)  ->  "
              f"gating {'adds' if gated_net > always_net else 'costs'} "
              f"{abs(gated_net - always_net):.1f}bp")
        summary[coin] = {"trades": int(len(trades)), "oos_hours": int(n_hours),
                         "gated_net_bps": round(gated_net, 2),
                         "always_on_net_bps": round(always_net, 2),
                         "median_hold_h": float(trades["hold_h"].median())}
        span = f"{trades['entry'].min():%Y-%m-%d}/{trades['exit'].max():%Y-%m-%d}"

    if summary:
        saved, h = results.record_run(RESULTS_ROOT, "ml_gated_ledger",
                                      {"coins": list(summary), "window": span,
                                       "rt_cost_bps": round(RT_COST * 1e4, 1)}, summary, span or "")
        print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}")


if __name__ == "__main__":
    main()
