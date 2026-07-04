#!/usr/bin/env python3
"""
Hyperliquid WebSocket capture daemon: multi-asset.

Subscribes to l2Book + trades + bbo for the configured COINS and a dedicated
activeAssetCtx channel for the chassis universe. Writes hourly Parquet to
  data/ws/{asset}/{channel}/{YYYY-MM-DD}/{HH}.parquet

Crash-safe (appends to existing file on restart within the same hour).
Auto-reconnect with exponential backoff. SIGTERM flushes buffers cleanly.

Run detached from this folder:
  mkdir -p logs
  nohup python3 -u ws_capture.py > logs/ws_capture.log 2>&1 &
Stop:
  pkill -TERM -f ws_capture.py
"""

import asyncio
import json
import signal
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import websockets

WS_URL = "wss://api.hyperliquid.xyz/ws"

# Universe split into groups. Hyperliquid empirically tolerates ~21-24 subs per connection.
COIN_GROUPS = [
    # Group 1: 5 chassis coins for W14 evaluation + 2 bonus crypto perps (ZEC, XMR) for W22 extension
    ["BTC", "ETH", "SOL", "NEAR", "HYPE", "ZEC", "XMR"],
    # Group 2: xyz indices + commodities (7); WTIOIL rejected by HL
    ["xyz:SP500", "xyz:XYZ100", "xyz:BRENTOIL", "xyz:GOLD", "xyz:SILVER",
     "xyz:COPPER", "xyz:NATGAS"],
    # Group 3: xyz AI + semiconductors (7)
    ["xyz:NVDA", "xyz:AMD", "xyz:MU", "xyz:MRVL", "xyz:SNDK", "xyz:INTC", "xyz:SPCX"],
    # Group 4: xyz mega-cap tech + crypto-equity bridges (8)
    ["xyz:GOOGL", "xyz:MSFT", "xyz:META", "xyz:AAPL", "xyz:TSLA", "xyz:ORCL", "xyz:MSTR", "xyz:CRCL"],
    # Group 5: xyz new US equities + AI infra + memory index (8)
    ["xyz:AMZN", "xyz:HOOD", "xyz:PLTR", "xyz:NBIS", "xyz:CRWV",
     "xyz:CBRS", "xyz:BB", "xyz:DRAM"],
]
COINS = [c for g in COIN_GROUPS for c in g]  # flat for buffer indexing
STREAM_CHANNELS = ["l2Book", "trades", "bbo"]

# Dedicated activeAssetCtx subscriptions for the full universe.
# Own connections so a ctx disconnect doesn't kill the main capture.
# One ctx push contains funding, oracle/mark px, premium, OI. Pushed every few seconds.
# Split into two groups for the same ~21-24 sub-per-connection tolerance.
CTX_GROUPS = [
    # Group 1: 7 crypto perps + 5 xyz indices/commodities
    ["BTC", "ETH", "SOL", "NEAR", "HYPE", "ZEC", "XMR",
     "xyz:SP500", "xyz:XYZ100", "xyz:BRENTOIL", "xyz:GOLD", "xyz:SILVER"],
    # Group 2: xyz single-name stocks (existing)
    ["xyz:NVDA", "xyz:AMD", "xyz:MU", "xyz:MRVL", "xyz:SNDK", "xyz:INTC",
     "xyz:SPCX", "xyz:GOOGL", "xyz:MSFT", "xyz:META", "xyz:AAPL", "xyz:TSLA",
     "xyz:ORCL", "xyz:MSTR", "xyz:CRCL"],
    # Group 3: xyz expansion (10 assets that HL accepts for ctx)
    ["xyz:COPPER", "xyz:NATGAS",
     "xyz:AMZN", "xyz:HOOD", "xyz:PLTR", "xyz:NBIS", "xyz:CRWV",
     "xyz:CBRS", "xyz:BB", "xyz:DRAM"],
]
CTX_CHANNELS = ["activeAssetCtx"]

# Full channels list for startup logging / buffer summary
CHANNELS = STREAM_CHANNELS + CTX_CHANNELS

BASE = Path(__file__).parent
OUT_BASE = BASE / "data" / "ws"

FLUSH_EVERY = 100         # rows per (asset, channel) buffer
FLUSH_EVERY_SEC = 30
PING_INTERVAL = 20
RECONNECT_MAX = 60        # seconds backoff cap
SUB_RATE_DELAY = 0.05     # seconds between subscribe messages
HEALTHY_MSG_THRESHOLD = 100  # received non-ack messages before resetting backoff

# Per-(asset, channel) buffers + last-flush tracker
buffers: dict[tuple[str, str], list[dict]] = defaultdict(list)
last_flush: dict[tuple[str, str], float] = defaultdict(float)
shutdown = False


def hour_path(asset: str, channel: str) -> Path:
    now = datetime.now(timezone.utc)
    day_dir = OUT_BASE / asset / channel / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{now.strftime('%H')}.parquet"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def flush(asset: str, channel: str) -> None:
    key = (asset, channel)
    if not buffers[key]:
        return
    df = pd.DataFrame(buffers[key])
    path = hour_path(asset, channel)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)
    n = len(buffers[key])
    buffers[key].clear()
    last_flush[key] = time.time()
    log(f"  flushed {n:4d} {asset}/{channel} -> {path.relative_to(BASE)}  (total {len(df)})")


def flush_all() -> None:
    for key in list(buffers.keys()):
        flush(*key)


def handle_l2book(payload: dict) -> None:
    book = payload.get("data") or payload
    if "levels" not in book:
        return
    captured_ms = int(time.time() * 1000)
    bids, asks = book["levels"]
    bid_top = bids[0] if bids else None
    ask_top = asks[0] if asks else None
    coin = book.get("coin")
    if coin not in COINS:
        return
    row = {
        "captured_ms": captured_ms,
        "book_ms": book.get("time"),
        "coin": coin,
        "best_bid_px": float(bid_top["px"]) if bid_top else None,
        "best_bid_sz": float(bid_top["sz"]) if bid_top else None,
        "best_ask_px": float(ask_top["px"]) if ask_top else None,
        "best_ask_sz": float(ask_top["sz"]) if ask_top else None,
        "spread": (float(ask_top["px"]) - float(bid_top["px"])) if (bid_top and ask_top) else None,
        "bid_depth_10": sum(float(l["sz"]) * float(l["px"]) for l in bids[:10]),
        "ask_depth_10": sum(float(l["sz"]) * float(l["px"]) for l in asks[:10]),
        "n_bid_lvls": len(bids),
        "n_ask_lvls": len(asks),
        "levels_json": json.dumps(book["levels"], separators=(",", ":")),
    }
    buffers[(coin, "l2book")].append(row)


def handle_trades(payload: dict) -> None:
    data = payload.get("data") or payload
    if not isinstance(data, list):
        return
    captured_ms = int(time.time() * 1000)
    for t in data:
        coin = t.get("coin")
        if coin not in COINS:
            continue
        buffers[(coin, "trades")].append({
            "captured_ms": captured_ms,
            "trade_ms": t.get("time"),
            "coin": coin,
            "side": t.get("side"),
            "px": float(t["px"]),
            "sz": float(t["sz"]),
            "notional": float(t["px"]) * float(t["sz"]),
            "tid": t.get("tid"),
            "hash": t.get("hash"),
            "users_buyer": (t.get("users") or [None, None])[0],
            "users_seller": (t.get("users") or [None, None])[1],
        })


def handle_bbo(payload: dict) -> None:
    data = payload.get("data") or payload
    if "bbo" not in data:
        return
    coin = data.get("coin")
    if coin not in COINS:
        return
    bid, ask = data["bbo"]
    buffers[(coin, "bbo")].append({
        "captured_ms": int(time.time() * 1000),
        "bbo_ms": data.get("time"),
        "coin": coin,
        "bid_px": float(bid["px"]) if bid else None,
        "bid_sz": float(bid["sz"]) if bid else None,
        "ask_px": float(ask["px"]) if ask else None,
        "ask_sz": float(ask["sz"]) if ask else None,
    })


def handle_ctx(payload: dict) -> None:
    """activeAssetCtx: funding rate, mark, oracle, premium, OI per coin."""
    data = payload.get("data") or payload
    coin = data.get("coin")
    if coin not in COINS:
        return
    ctx = data.get("ctx")
    if not isinstance(ctx, dict):
        return
    def f(k):
        v = ctx.get(k)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    buffers[(coin, "ctx")].append({
        "captured_ms": int(time.time() * 1000),
        "coin": coin,
        "funding":      f("funding"),
        "open_interest": f("openInterest"),
        "prev_day_px":  f("prevDayPx"),
        "day_ntl_vlm":  f("dayNtlVlm"),
        "premium":      f("premium"),
        "oracle_px":    f("oraclePx"),
        "mark_px":      f("markPx"),
        "mid_px":       f("midPx"),
        "day_base_vlm": f("dayBaseVlm"),
    })


async def periodic_flush() -> None:
    while not shutdown:
        await asyncio.sleep(5)
        now = time.time()
        for key in list(buffers.keys()):
            if (len(buffers[key]) >= FLUSH_EVERY or
                (buffers[key] and (now - last_flush[key]) > FLUSH_EVERY_SEC)):
                flush(*key)


def buf_summary() -> str:
    """Compact buffer state for periodic logs."""
    rows_per_asset = defaultdict(int)
    for (asset, _), buf in buffers.items():
        rows_per_asset[asset] += len(buf)
    return " ".join(f"{a}={rows_per_asset[a]}" for a in COINS)


# Total messages across all consumer tasks (for the periodic progress line)
total_msgs = 0


async def consume_group(group_idx: int, coins: list[str], channels: list[str]) -> None:
    """One persistent WS connection for a (coins, channels) group."""
    global total_msgs
    backoff = 1
    n_subs = len(coins) * len(channels)
    tag = f"g{group_idx}"
    while not shutdown:
        msgs_this_session = 0
        try:
            async with websockets.connect(
                WS_URL, ping_interval=PING_INTERVAL, max_size=10 * 1024 * 1024
            ) as ws:
                log(f"[{tag}] CONNECTED  {len(coins)} coins x {len(channels)} = {n_subs} subs  "
                    f"channels={channels}  coins={coins}")
                for coin in coins:
                    for ch in channels:
                        sub = {"method": "subscribe",
                               "subscription": {"type": ch, "coin": coin}}
                        await ws.send(json.dumps(sub))
                        await asyncio.sleep(SUB_RATE_DELAY)
                log(f"[{tag}]   all {n_subs} subscribe messages sent ({n_subs * SUB_RATE_DELAY:.1f}s)")
                async for raw in ws:
                    if shutdown:
                        break
                    msg = json.loads(raw)
                    ch = msg.get("channel")
                    if ch == "l2Book":
                        handle_l2book(msg); msgs_this_session += 1
                    elif ch == "trades":
                        handle_trades(msg); msgs_this_session += 1
                    elif ch == "bbo":
                        handle_bbo(msg); msgs_this_session += 1
                    elif ch == "activeAssetCtx":
                        handle_ctx(msg); msgs_this_session += 1
                    elif ch == "error":
                        log(f"[{tag}]   ERROR from server: {msg}")
                    # 'subscriptionResponse' silently consumed
                    total_msgs += 1
                    if msgs_this_session == HEALTHY_MSG_THRESHOLD:
                        backoff = 1
                        log(f"[{tag}]   session healthy ({HEALTHY_MSG_THRESHOLD}+ msgs): backoff reset")
        except Exception as e:
            log(f"[{tag}] DISCONNECT after {msgs_this_session} msgs: {e!r}, retry in {backoff}s")
            # Don't flush_all here; other groups are still running. Their buffers stay.
            if shutdown:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX)


async def progress_logger() -> None:
    while not shutdown:
        await asyncio.sleep(60)
        log(f"  progress: {total_msgs:>8} total msgs  |  buf: {buf_summary()}")


def request_shutdown(*_) -> None:
    global shutdown
    log("shutdown requested: flushing and exiting")
    shutdown = True


async def main() -> None:
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    log(f"=== Hyperliquid WS capture -> {OUT_BASE} ===")
    # Build group list: stream channels per coin group, plus the activeAssetCtx groups.
    groups: list[tuple[list[str], list[str]]] = [
        (g, STREAM_CHANNELS) for g in COIN_GROUPS
    ] + [
        (g, CTX_CHANNELS) for g in CTX_GROUPS
    ]
    total_subs = sum(len(c) * len(ch) for c, ch in groups)
    log(f"  total assets : {len(COINS)}  ({len(groups)} connection groups)")
    log(f"  stream chans : {STREAM_CHANNELS}")
    log(f"  ctx groups   : {len(CTX_GROUPS)} x {CTX_CHANNELS}")
    log(f"  total subs   : {total_subs}")
    for i, (coins, channels) in enumerate(groups):
        log(f"  group {i}: {len(coins)} coins x {len(channels)} channels "
            f"= {len(coins)*len(channels)} subs, channels={channels} coins={coins}")
    tasks = [consume_group(i, coins, channels) for i, (coins, channels) in enumerate(groups)]
    tasks.append(periodic_flush())
    tasks.append(progress_logger())
    await asyncio.gather(*tasks)
    flush_all()
    log("done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        flush_all()
