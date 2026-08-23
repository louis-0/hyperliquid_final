#!/usr/bin/env python3
"""Fetch Hyperliquid spot candles for the basis-trade pair set.

Hyperliquid spot pairs are addressed by @{index} identifier; a bare name like
"HYPE" routes to the perpetual on /info, so the @ form avoids the spot/perp
collision. NEAR has no Hyperliquid spot market and is not fetchable.

Writes Parquet to:
  data/spot/{coin}.parquet                  # 1h default
  data/spot_{interval}/{coin}.parquet      # for any non-1h interval

Idempotent (re-run overwrites).

Examples:
  python3 fetch_spot_history.py                                   # 90 days, 1h, all mapped coins
  python3 fetch_spot_history.py BTC ETH
  python3 fetch_spot_history.py --start 2024-01-01 --end 2026-08-23   # pinned window
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fetch_rest_history import BASE, RATE_DELAY, log, post

SPOT_PAIRS = {
    "BTC":  "@142",   # UBTC/USDC
    "ETH":  "@151",   # UETH/USDC
    "SOL":  "@156",   # USOL/USDC
    "HYPE": "@107",   # HYPE/USDC
    "ZEC":  "@272",   # UZEC/USDC (Unit-bridged)
}
NINETY_DAYS_MS = 90 * 24 * 60 * 60 * 1000


def spot_dir(interval: str) -> Path:
    """Output dir for spot candles. 1h goes to data/spot/ for backwards compat."""
    if interval == "1h":
        return BASE / "data" / "spot"
    return BASE / "data" / f"spot_{interval}"


def fetch_spot(coin: str, pair_id: str, start_ms: int, end_ms: int, interval: str) -> pd.DataFrame:
    log(f"[spot  {interval:>3}] {coin} ({pair_id})")
    all_rows: list[dict] = []
    cursor_ms = start_ms
    while cursor_ms < end_ms:
        payload = {
            "type": "candleSnapshot",
            "req": {"coin": pair_id, "interval": interval,
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
    out_dir = spot_dir(interval)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_days = (end_ms - start_ms) / (1000 * 60 * 60 * 24)
    log(f"=== fetching {interval} spot candles for {len(coins)} coins ===")
    log(f"  window : {window_days:.0f} days")
    log(f"  output : {out_dir.relative_to(BASE)}/")

    ok: list[str] = []
    failed: list[str] = []
    for coin in coins:
        pair_id = SPOT_PAIRS.get(coin)
        if pair_id is None:
            log(f"  [spot] {coin}: no Hyperliquid spot pair, skipped")
            failed.append(coin)
            continue
        df = fetch_spot(coin, pair_id, start_ms, end_ms, interval)
        if df.empty:
            log(f"  [spot] {coin}: NO DATA")
            failed.append(coin)
            continue
        out = out_dir / f"{coin}.parquet"
        df.to_parquet(out, index=False)
        log(f"  [spot] {coin}: wrote {len(df)} rows -> {out.relative_to(BASE)}")
        ok.append(coin)

    log(f"  ok    : {len(ok)} {ok}")
    log(f"  failed: {len(failed)} {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("coins", nargs="*", help="coins to fetch (default: all mapped pairs)")
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

    coins = args.coins if args.coins else list(SPOT_PAIRS)
    main(coins, args.interval, start_ms, end_ms)
