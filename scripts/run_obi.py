#!/usr/bin/env python3
"""Order-book-imbalance study: does top-of-book imbalance predict the next move? Built on hlq.

Aggregates the captured quotes into 5-second bars (mean top-of-book imbalance, last mid) and
reports the rank correlation between imbalance and forward mid returns at several horizons,
with the top-decile forward move against the round-trip taker cost (the window's own median
spread plus taker fees). The signal is anonymous book state, the microstructure counterpart
of the informed-flow study's identity signal.

    python scripts/run_obi.py --start 2026-06-23 --end 2026-08-22
    python scripts/run_obi.py --start 2026-06-23 --end 2026-08-22 BTC xyz:MRVL
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, flow, results
from hlq.costs import CostModel

COINS = ["BTC", "ETH", "SOL", "HYPE", "xyz:MRVL"]
BAR_MS = 5_000
HORIZONS = [1, 6, 12, 60]              # forward bars: 5s, 30s, 1m, 5m
COST = CostModel()
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def obi_bars(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """5-second bars of mean imbalance, last mid, and mean spread from the captured quotes."""
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
    """).df().set_index("bar")


def main(start: str | None, end: str | None, coins: list[str]) -> None:
    start_ms = (int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
                if start else 0)
    end_ms = (int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
              if end else 1 << 62)
    print("=" * 92)
    print("Order-book-imbalance study: 5s bars, imbalance vs forward mid return")
    print(f"  window {start or 'tape start'} to {end or 'tape end'}")
    print("=" * 92)
    print(f"\n{'coin':<10} {'bars':>9} {'spread':>7} | IC @5s/30s/1m/5m | top-decile 1m vs cost")
    print("-" * 92)

    table: dict[str, dict] = {}
    for coin in coins:
        g = obi_bars(coin, start_ms, end_ms)
        if len(g) < 200:
            print(f"  {coin:<10} (insufficient bars)")
            continue
        spread = float(g["spread_bps"].median())
        rt_cost = spread + 2 * COST.taker_fee * 1e4
        ic = {h: flow.spearman_ic(g["obi"], flow.fwd_ret_bps(g["mid"], h)) for h in HORIZONS}
        top = flow.top_decile_move(g["obi"], flow.fwd_ret_bps(g["mid"], 12))
        print(f"  {coin:<10} {len(g):>9,} {spread:>6.2f}b | "
              f"{ic[1]:+.3f} {ic[6]:+.3f} {ic[12]:+.3f} {ic[60]:+.3f} | "
              f"{top:+.2f}bp vs {rt_cost:.1f}bp = {top - rt_cost:+.1f}")
        table[coin] = {"bars": int(len(g)), "spread_bps": round(spread, 3),
                       "ic_obi": {str(h): round(ic[h], 4) for h in HORIZONS},
                       "top_decile_1m_bps": round(top, 2), "rt_cost_bps": round(rt_cost, 2)}

    if table:
        saved, h = results.record_run(RESULTS_ROOT, "obi",
                                      {"coins": list(table), "start": start, "end": end}, table,
                                      f"{start or 'tape-start'}/{end or 'tape-end'}")
        print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--start", default=None, help="window start YYYY-MM-DD")
    p.add_argument("--end", default=None, help="window end YYYY-MM-DD")
    p.add_argument("coins", nargs="*", default=[])
    args = p.parse_args()
    main(args.start, args.end, args.coins or COINS)
