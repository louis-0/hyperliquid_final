#!/usr/bin/env python3
"""Fetch Hyperliquid REST history: funding rates + mark candles.

Writes Parquet to:
  data/funding/{coin}.parquet
  data/marks/{coin}.parquet                 # 1h default
  data/marks_{interval}/{coin}.parquet      # for any non-1h interval

Idempotent (re-run overwrites).

Examples:
  python3 fetch_rest_history.py                                   # 90 days, 1h, all coins
  python3 fetch_rest_history.py BTC ETH
  python3 fetch_rest_history.py --interval 5m
  python3 fetch_rest_history.py --interval 1h --start 2023-12-01
  python3 fetch_rest_history.py --start 2023-12-01 --end 2026-08-23   # pinned window
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from ws_capture import COINS  # the assets the daemon subscribes to

BASE = Path(__file__).parent
DATA_FUNDING = BASE / "data" / "funding"
LOG_PATH     = BASE / "logs" / "fetch_rest_history.log"

API_URL = "https://api.hyperliquid.xyz/info"
NINETY_DAYS_MS = 90 * 24 * 60 * 60 * 1000
RATE_DELAY = 0.30  # seconds between paginated requests (Hyperliquid 429s under ~10 req/s sustained)

SESSION = requests.Session()
SESSION.headers["Content-Type"] = "application/json"


def marks_dir(interval: str) -> Path:
    """Output dir for mark candles. 1h goes to data/marks/ for backwards compat."""
    if interval == "1h":
        return BASE / "data" / "marks"
    return BASE / "data" / f"marks_{interval}"


def log(*a) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def post(payload: dict) -> object:
    """POST to /info with exponential-backoff retries (handles Hyperliquid 429s)."""
    for attempt in range(7):
        try:
            r = SESSION.post(API_URL, data=json.dumps(payload), timeout=30)
            if r.status_code == 200:
                return r.json()
            log(f"  non-200 status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            log(f"  request error attempt={attempt+1} err={e!r}")
        time.sleep(min(60, 2 ** attempt))  # 1, 2, 4, 8, 16, 32, 60s
    return None


def fetch_funding(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    log(f"[funding] {coin}")
    all_rows: list[dict] = []
    cursor_ms = start_ms
    while cursor_ms < end_ms:
        payload = {"type": "fundingHistory", "coin": coin,
                   "startTime": cursor_ms, "endTime": end_ms}
        data = post(payload)
        if not data or not isinstance(data, list) or len(data) == 0:
            break
        all_rows.extend(data)
        last_ms = int(data[-1]["time"])
        if last_ms <= cursor_ms:
            break
        cursor_ms = last_ms + 1
        time.sleep(RATE_DELAY)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df["premium"] = df["premium"].astype(float)
    df = df[["ts", "coin", "funding_rate", "premium"]].drop_duplicates("ts").sort_values("ts")
    return df


def fetch_marks(coin: str, start_ms: int, end_ms: int, interval: str) -> pd.DataFrame:
    log(f"[marks {interval:>3}] {coin}")
    all_rows: list[dict] = []
    cursor_ms = start_ms
    while cursor_ms < end_ms:
        payload = {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval,
                    "startTime": cursor_ms, "endTime": end_ms},
        }
        data = post(payload)
        if not data or not isinstance(data, list) or len(data) == 0:
            break
        all_rows.extend(data)
        last_ms = int(data[-1]["T"])
        if last_ms <= cursor_ms:
            break
        cursor_ms = last_ms + 1
        time.sleep(RATE_DELAY)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["ts"]     = pd.to_datetime(df["T"], unit="ms", utc=True)
    df["open"]   = df["o"].astype(float)
    df["close"]  = df["c"].astype(float)
    df["high"]   = df["h"].astype(float)
    df["low"]    = df["l"].astype(float)
    df["volume"] = df["v"].astype(float)
    df["coin"]   = coin
    df = df[["ts", "coin", "open", "close", "high", "low", "volume"]].drop_duplicates("ts").sort_values("ts")
    return df


def main(coins: list[str], interval: str, start_ms: int, end_ms: int) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_FUNDING.mkdir(parents=True, exist_ok=True)
    marks_out_dir = marks_dir(interval)
    marks_out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    window_days = (end_ms - start_ms) / (1000 * 60 * 60 * 24)

    started = time.time()
    log(f"=== fetching funding + {interval} marks for {len(coins)} coins ===")
    log(f"  window  : {window_days:.0f} days, {start_dt.date()} to {now.date()}")
    log(f"  output  : data/funding/ + {marks_out_dir.relative_to(BASE)}/")
    log(f"  coins   : {coins}")

    ok: list[str] = []
    failed: list[str] = []
    for coin in coins:
        try:
            df_f = fetch_funding(coin, start_ms, end_ms)
            if not df_f.empty:
                out = DATA_FUNDING / f"{coin}.parquet"
                df_f.to_parquet(out, index=False)
                log(f"  [funding] {coin}: wrote {len(df_f)} rows -> {out.relative_to(BASE)}")
            else:
                log(f"  [funding] {coin}: NO DATA")

            df_m = fetch_marks(coin, start_ms, end_ms, interval)
            if not df_m.empty:
                out = marks_out_dir / f"{coin}.parquet"
                df_m.to_parquet(out, index=False)
                log(f"  [marks  ] {coin}: wrote {len(df_m)} rows -> {out.relative_to(BASE)}")
            else:
                log(f"  [marks  ] {coin}: NO DATA")

            if not df_f.empty and not df_m.empty:
                ok.append(coin)
            else:
                failed.append(coin)
        except Exception as e:
            log(f"  FATAL on {coin}: {e!r}")
            failed.append(coin)

    log(f"=== done in {time.time() - started:.1f}s ===")
    log(f"  ok    : {len(ok)} {ok}")
    log(f"  failed: {len(failed)} {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("coins", nargs="*", help="coins to fetch (default: every captured asset)")
    parser.add_argument("--interval", default="1h",
                        help="candle interval: 1m, 5m, 15m, 1h, 4h, 1d (default: 1h)")
    parser.add_argument("--start", default=None,
                        help="start date YYYY-MM-DD (default: 90 days back)")
    parser.add_argument("--end", default=None,
                        help="end date YYYY-MM-DD (default: now); set for a reproducible pinned window")
    args = parser.parse_args()

    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_ms = int(end_dt.timestamp() * 1000)
    else:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
    else:
        start_ms = end_ms - NINETY_DAYS_MS

    coins = args.coins if args.coins else COINS
    main(coins, args.interval, start_ms, end_ms)
