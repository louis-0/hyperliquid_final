#!/usr/bin/env python3
"""Fetch 90 days of hourly funding rates + 1h mark candles via Hyperliquid REST.

Writes Parquet to:
  data/funding/{coin}.parquet
  data/marks/{coin}.parquet

Idempotent (re-run overwrites).

Run from this folder:
  python3 fetch_rest_history.py            # fetch all 27 configured coins
  python3 fetch_rest_history.py BTC ETH    # fetch specific coins only
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from ws_capture import COINS  # 27 assets the daemon subscribes to

BASE = Path(__file__).parent
DATA_FUNDING = BASE / "data" / "funding"
DATA_MARKS   = BASE / "data" / "marks"
LOG_PATH     = BASE / "logs" / "fetch_rest_history.log"

API_URL = "https://api.hyperliquid.xyz/info"
NINETY_DAYS_MS = 90 * 24 * 60 * 60 * 1000
RATE_DELAY = 0.10  # seconds between paginated requests

SESSION = requests.Session()
SESSION.headers["Content-Type"] = "application/json"


def log(*a) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def post(payload: dict) -> object:
    """POST to /info with retries."""
    for attempt in range(5):
        try:
            r = SESSION.post(API_URL, data=json.dumps(payload), timeout=30)
            if r.status_code == 200:
                return r.json()
            log(f"  non-200 status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            log(f"  request error attempt={attempt+1} err={e!r}")
        time.sleep(1 + attempt)
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


def fetch_marks_1h(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    log(f"[marks]   {coin}")
    all_rows: list[dict] = []
    cursor_ms = start_ms
    while cursor_ms < end_ms:
        payload = {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": "1h",
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


def main(coins: list[str]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_FUNDING.mkdir(parents=True, exist_ok=True)
    DATA_MARKS.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = end_ms - NINETY_DAYS_MS

    started = time.time()
    log(f"=== fetching funding + marks for {len(coins)} coins ===")
    log(f"  window : 90 days ending {now.isoformat(timespec='seconds')}")
    log(f"  coins  : {coins}")

    ok: list[str] = []
    failed: list[str] = []
    for coin in coins:
        try:
            df_f = fetch_funding(coin, start_ms, end_ms)
            if not df_f.empty:
                out = DATA_FUNDING / f"{coin}.parquet"
                df_f.to_parquet(out, index=False)
                log(f"  [funding] {coin}: wrote {len(df_f)} rows -> {out.name}")
            else:
                log(f"  [funding] {coin}: NO DATA")

            df_m = fetch_marks_1h(coin, start_ms, end_ms)
            if not df_m.empty:
                out = DATA_MARKS / f"{coin}.parquet"
                df_m.to_parquet(out, index=False)
                log(f"  [marks]   {coin}: wrote {len(df_m)} rows -> {out.name}")
            else:
                log(f"  [marks]   {coin}: NO DATA")

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
    coins = sys.argv[1:] if len(sys.argv) > 1 else COINS
    main(coins)
