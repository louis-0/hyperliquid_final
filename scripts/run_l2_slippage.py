#!/usr/bin/env python3
"""L2 slippage realism check, built on the hlq package.

Walks the captured perpetual order books (hlq.execution.walk_book) to estimate slippage at
typical trade notionals per coin, reporting the median and p95 of (vwap - mid) / mid in basis
points. Only perpetual books are captured, so this covers the short-perpetual leg.

    python scripts/run_l2_slippage.py
    python scripts/run_l2_slippage.py BTC ETH
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, execution, results

CHASSIS = ["BTC", "ETH", "SOL", "NEAR", "HYPE"]
NOTIONALS = [1_000, 10_000, 100_000]
SAMPLE_EVERY = 60   # roughly one snapshot per minute at the ~1 Hz capture rate
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def coin_slippage(coin: str, root=data.DATA_ROOT) -> dict:
    """Median/p95 ask-side slippage per notional, sampled across a coin's captured books."""
    book_dir = Path(root) / "ws" / coin / "l2book"
    files = sorted(book_dir.rglob("*.parquet")) if book_dir.exists() else []
    samples: dict[int, list[float]] = {n: [] for n in NOTIONALS}
    for fp in files:
        try:
            df = pd.read_parquet(fp, columns=["best_bid_px", "best_ask_px", "levels_json"]).iloc[::SAMPLE_EVERY]
        except Exception:
            continue
        for bb, ba, raw in zip(df["best_bid_px"], df["best_ask_px"], df["levels_json"]):
            parsed = execution.parse_l2_snapshot(bb, ba, raw)
            if parsed is None:
                continue
            mid, asks = parsed
            for n in NOTIONALS:
                slip = execution.walk_book(asks, n, mid)
                if slip is not None:
                    samples[n].append(slip)
    return {"coin": coin, "n_files": len(files), "samples": samples}


def main(coins: list[str]) -> None:
    print("=" * 78)
    print("L2 slippage on captured perp books (short-perp leg), basis points")
    print(f"  sampling: 1 in {SAMPLE_EVERY} snapshots; slippage = (vwap - mid) / mid")
    print("=" * 78)
    header = f"\n{'coin':<6} {'files':>6}"
    for n in NOTIONALS:
        header += f" {'med@' + str(n // 1000) + 'k':>10} {'p95@' + str(n // 1000) + 'k':>10}"
    print(header)
    print("-" * 78)

    table: dict[str, dict[str, float]] = {}
    for coin in coins:
        r = coin_slippage(coin)
        line = f"  {coin:<6} {r['n_files']:>6}"
        for n in NOTIONALS:
            arr = np.asarray(r["samples"][n], dtype=float)
            if arr.size:
                table.setdefault(coin, {})[f"med_bps@{n // 1000}k"] = round(float(np.median(arr) * 1e4), 3)
                line += f" {np.median(arr) * 1e4:>+10.2f} {np.percentile(arr, 95) * 1e4:>+10.2f}"
            else:
                line += f" {'n/a':>10} {'n/a':>10}"
        print(line)

    if table:
        saved, h = results.record_run(RESULTS_ROOT, "l2_slippage",
                                      {"coins": list(table), "notionals": NOTIONALS},
                                      {"median_bps": table})
        print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}")


if __name__ == "__main__":
    main(sys.argv[1:] or CHASSIS)