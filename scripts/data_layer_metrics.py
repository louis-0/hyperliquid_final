#!/usr/bin/env python3
"""Data-layer metrics: coverage, integrity, latency, and robustness of the capture.

Quantifies the WebSocket and REST capture the backtests read: per-channel message counts, the
capture window, hourly coverage of the mark series, trade-capture latency (the gap between a
trade's exchange timestamp and its receipt), and an uptime proxy. Reads the parquet under the
data root directly, so it runs against a populated local capture.

    python scripts/data_layer_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data

CHANNELS = ["trades", "l2book", "bbo", "ctx"]
CHASSIS = ["BTC", "ETH", "SOL", "HYPE"]


def main() -> None:
    root = data.DATA_ROOT
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute("PRAGMA threads=4")

    print("=" * 78)
    print("Data-layer metrics: coverage, integrity, latency, robustness")
    print("=" * 78)

    print("\ncoverage (rows per WebSocket channel, all assets)")
    print("-" * 78)
    total = 0
    for ch in CHANNELS:
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{root}/ws/*/{ch}/**/*.parquet')").fetchone()[0]
        total += n
        print(f"  {ch:<8} {n:>15,}")
    print(f"  {'total':<8} {total:>15,}")

    w = con.execute(f"SELECT min(trade_ms), max(trade_ms) "
                    f"FROM read_parquet('{root}/ws/*/trades/**/*.parquet')").fetchone()
    t0, t1 = pd.to_datetime(w[0], unit="ms", utc=True), pd.to_datetime(w[1], unit="ms", utc=True)
    print(f"\nwindow  {t0.date()} to {t1.date()}  ({(t1 - t0).days} days)")

    print("\nlatency (BTC trade capture, milliseconds)")
    print("-" * 78)
    lat = con.execute(f"""
        SELECT approx_quantile((captured_ms - trade_ms)::DOUBLE, 0.5),
               approx_quantile((captured_ms - trade_ms)::DOUBLE, 0.95)
        FROM read_parquet('{root}/ws/BTC/trades/**/*.parquet')
        WHERE captured_ms IS NOT NULL AND trade_ms IS NOT NULL AND captured_ms >= trade_ms
    """).fetchone()
    print(f"  median {lat[0]:.0f} ms   p95 {lat[1]:.0f} ms")

    print("\nintegrity (hourly mark coverage) and robustness")
    print("-" * 78)
    for coin in CHASSIS:
        m = data.load_marks(coin, root=root)
        hrs = m["ts"].dt.floor("1h").drop_duplicates().sort_values()
        expected = int((hrs.max() - hrs.min()).total_seconds() // 3600) + 1
        print(f"  {coin:<5} marks present {len(hrs):>5} / {expected:>5}   "
              f"missing {100 * (1 - len(hrs) / expected):.2f}%")

    u = con.execute(f"""
        WITH h AS (SELECT DISTINCT (trade_ms // 3600000) AS hr
                   FROM read_parquet('{root}/ws/BTC/trades/**/*.parquet'))
        SELECT count(*), max(hr) - min(hr) + 1 FROM h
    """).fetchone()
    print(f"  BTC hours with a trade {u[0]:,} / {u[1]:,}  ({100 * u[0] / u[1]:.1f}%)")


if __name__ == "__main__":
    main()
