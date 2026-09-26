#!/usr/bin/env python3
"""Per-trade ledger for the order-book-imbalance signal, built on the hlq package.

Trades the top-of-book imbalance as an event strategy: the entry thresholds are the top and
bottom deciles of the signal over the first 30 percent of bars, and the remaining bars are
traded out of sample. Entries are non-overlapping (one position at a time), held a fixed
number of bars, and each trade is debited the measured bar spread plus the taker fee on both
sides. The full ledger is written per coin as gzipped CSV; the printout shows a sample and totals.

    python scripts/run_obi_ledger.py --start 2026-06-23 --end 2026-08-22
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, results
from hlq.costs import CostModel

COINS = ["BTC", "ETH", "SOL", "HYPE"]
BAR_MS = 5_000
HOLD_BARS = 12                 # nominal one-minute hold
THRESH_FRAC = 0.30             # threshold set on the first 30 percent of bars
COST = CostModel()
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def obi_bars(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    src = data.DATA_ROOT / "ws" / coin / "bbo" / "**" / "*.parquet"
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    return con.execute(f"""
        SELECT (bbo_ms // {BAR_MS}) * {BAR_MS} AS bar,
               AVG((bid_sz - ask_sz) / NULLIF(bid_sz + ask_sz, 0))              AS obi,
               arg_max((bid_px + ask_px) / 2, bbo_ms)                           AS mid,
               AVG((ask_px - bid_px) / NULLIF((ask_px + bid_px) / 2, 0)) * 1e4  AS spread_bps
        FROM read_parquet('{src}')
        WHERE bbo_ms >= {start_ms} AND bbo_ms < {end_ms}
        GROUP BY 1 ORDER BY 1
    """).df()


def coin_trades(g: pd.DataFrame) -> pd.DataFrame:
    """Non-overlapping event trades over the out-of-sample bars."""
    split = int(len(g) * THRESH_FRAC)
    hi = g["obi"].iloc[:split].quantile(0.9)
    lo = g["obi"].iloc[:split].quantile(0.1)
    obi = g["obi"].to_numpy()
    mid = g["mid"].to_numpy()
    spread = g["spread_bps"].to_numpy()
    bar = g["bar"].to_numpy()
    fee_bps = COST.taker_fee * 1e4

    rows = []
    i = split
    last_exit = split
    n = len(g)
    while i < n - HOLD_BARS:
        if i < last_exit or not (obi[i] >= hi or obi[i] <= lo):
            i += 1
            continue
        j = i + HOLD_BARS
        side = 1 if obi[i] >= hi else -1
        gross = side * (mid[j] / mid[i] - 1) * 1e4
        cost = (spread[i] + spread[j]) / 2 + 2 * fee_bps
        rows.append({"entry": pd.Timestamp(bar[i], unit="ms", tz="UTC"),
                     "exit": pd.Timestamp(bar[j], unit="ms", tz="UTC"),
                     "side": "long" if side > 0 else "short",
                     "hold_s": int((bar[j] - bar[i]) // 1000),
                     "entry_mid": mid[i], "exit_mid": mid[j],
                     "gross_bps": gross, "cost_bps": cost, "net_bps": gross - cost})
        last_exit = j
        i = j
    return pd.DataFrame(rows)


def main(start: str, end: str, coins: list[str]) -> None:
    print("=" * 88)
    print(f"OBI trade ledger: decile thresholds from the first {int(THRESH_FRAC * 100)}% of bars, "
          f"{HOLD_BARS * BAR_MS // 1000}s nominal hold, window {start} to {end}")
    print("=" * 88)

    summary: dict[str, dict] = {}
    for coin in coins:
        g = obi_bars(coin,
                     int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000),
                     int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000))
        if len(g) < 1000:
            print(f"  {coin}: insufficient bars")
            continue
        t = coin_trades(g)
        out = RESULTS_ROOT / f"obi_ledger_{coin}.csv.gz"
        t.to_csv(out, index=False)
        wins = (t["net_bps"] > 0).mean() * 100
        gross_wins = (t["gross_bps"] > 0).mean() * 100
        print(f"\n{coin}: {len(t):,} trades -> {out.name}")
        print(t.head(8).to_string(index=False,
              formatters={"entry_mid": "{:.1f}".format, "exit_mid": "{:.1f}".format,
                          "gross_bps": "{:+.2f}".format, "cost_bps": "{:.2f}".format,
                          "net_bps": "{:+.2f}".format}))
        print(f"  direction right {gross_wins:.1f}% of trades; net winners {wins:.1f}%; "
              f"avg gross {t['gross_bps'].mean():+.2f}bp, avg cost {t['cost_bps'].mean():.2f}bp, "
              f"avg net {t['net_bps'].mean():+.2f}bp; TOTAL {t['net_bps'].sum():+,.0f}bp "
              f"over {len(t):,} trades")
        summary[coin] = {"trades": int(len(t)), "gross_win_pct": round(gross_wins, 1),
                         "net_win_pct": round(wins, 1),
                         "avg_gross_bps": round(float(t["gross_bps"].mean()), 3),
                         "avg_cost_bps": round(float(t["cost_bps"].mean()), 3),
                         "avg_net_bps": round(float(t["net_bps"].mean()), 3),
                         "total_net_bps": round(float(t["net_bps"].sum()), 1)}

    if summary:
        saved, h = results.record_run(RESULTS_ROOT, "obi_ledger",
                                      {"coins": list(summary), "window": f"{start}/{end}",
                                       "hold_bars": HOLD_BARS}, summary, f"{start}/{end}")
        print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--start", default="2026-06-23")
    p.add_argument("--end", default="2026-08-22")
    p.add_argument("coins", nargs="*", default=[])
    args = p.parse_args()
    main(args.start, args.end, args.coins or COINS)
