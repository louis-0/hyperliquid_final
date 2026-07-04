#!/usr/bin/env python3
"""F1 vol-clustering replication.

Per coin: compute the correlation between squared 5-min log returns at
t and t-1, in-sample and out-of-sample. Both positive = ARCH effect
survives (Engle 1982; Bollerslev 1986).

The prelim Chapter 4 reports 27/27 sign-survival; this reproduces it
from data/ or data_sample/ over the W10 panel only.

Run from the repo root:
  python3 queries/arch_replication.py              # walks data/ws/
  python3 queries/arch_replication.py data_sample  # walks data_sample/ws/
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BAR_MIN    = 5    # 5-minute bars
TRAIN_FRAC = 0.6  # in-sample portion

# W10 panel per the prelim Chapter 4: 5 chassis + 2 bonus crypto + 20 xyz: TradFi.
# Coins added later are not part of the W10 replication.
W10_PANEL = [
    # 5 chassis
    "BTC", "ETH", "SOL", "NEAR", "HYPE",
    # 2 bonus crypto
    "ZEC", "XMR",
    # 20 xyz: TradFi (WS connection groups 2-4)
    "xyz:SP500", "xyz:XYZ100", "xyz:BRENTOIL", "xyz:GOLD", "xyz:SILVER",
    "xyz:NVDA", "xyz:AMD", "xyz:MU", "xyz:MRVL", "xyz:SNDK", "xyz:INTC", "xyz:SPCX",
    "xyz:GOOGL", "xyz:MSFT", "xyz:META", "xyz:AAPL", "xyz:TSLA", "xyz:ORCL", "xyz:MSTR", "xyz:CRCL",
]


def load_mid(data_root: Path, coin: str) -> pd.DataFrame:
    sql = f"""
    SELECT epoch_ms(bbo_ms) AS ts, (bid_px + ask_px)/2.0 AS mid
    FROM read_parquet('{data_root}/ws/{coin}/bbo/*/*.parquet')
    WHERE bid_px IS NOT NULL AND ask_px IS NOT NULL
    ORDER BY bbo_ms
    """
    return duckdb.sql(sql).df()


def bars(df: pd.DataFrame, minutes: int) -> pd.Series:
    return (df.set_index("ts").sort_index()["mid"]
              .resample(f"{minutes}min").last().dropna())


def sign_survives(prices: pd.Series) -> tuple[bool, int, float, float]:
    log_rets = np.log(prices).diff().dropna()
    sq = log_rets ** 2
    sq_lag = sq.shift(1).dropna()
    sq_t = sq.loc[sq_lag.index]
    n = len(sq_t)
    if n < 100:
        return False, n, 0.0, 0.0
    split = int(n * TRAIN_FRAC)
    train_corr = np.corrcoef(sq_lag.iloc[:split], sq_t.iloc[:split])[0, 1]
    test_corr  = np.corrcoef(sq_lag.iloc[split:], sq_t.iloc[split:])[0, 1]
    ok = train_corr > 0 and test_corr > 0
    return ok, n, train_corr, test_corr


def main(data_root: Path) -> None:
    ws_dir = data_root / "ws"
    available = set(d.name for d in ws_dir.iterdir() if d.is_dir())
    coins = [c for c in W10_PANEL if c in available]
    if not coins:
        print(f"  no W10-panel coins found under {ws_dir}")
        sys.exit(1)

    print(f"=== F1 vol-clustering replication ===")
    print(f"  data root : {data_root}")
    print(f"  coins     : {len(coins)} of {len(W10_PANEL)} W10-panel")
    print(f"  bar width : {BAR_MIN} min")
    print(f"  train frac: {TRAIN_FRAC}")
    print()

    survived = 0
    tested = 0
    for coin in coins:
        try:
            ticks = load_mid(data_root, coin)
            if ticks.empty:
                print(f"  {coin:<18} skipped (no bbo)")
                continue
            ts_bars = bars(ticks, BAR_MIN)
            ok, n, tr, te = sign_survives(ts_bars)
            verdict = "SURVIVED" if ok else "FAILED" if n >= 100 else "TOO_SHORT"
            print(f"  {coin:<18} {verdict:<10} n={n:5d}  train={tr:+.4f}  test={te:+.4f}")
            if n >= 100:
                tested += 1
                if ok:
                    survived += 1
        except Exception as e:
            print(f"  {coin:<18} ERROR: {e!r}")

    print()
    print(f"=== {survived}/{tested} sign-survived ===")


if __name__ == "__main__":
    user_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    if not user_path.is_absolute():
        # resolve relative to script's parent so it works from anywhere
        user_path = Path(__file__).parent.parent / user_path
    main(user_path)
