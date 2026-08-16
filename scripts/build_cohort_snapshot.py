#!/usr/bin/env python3
"""Build the aggregate smart-money snapshot the dashboard reads.

Applies a frozen smart-money cohort (a set of addresses labelled is_smart in a prior cohort
study) to this project's own captured trade flow, and writes a per-coin aggregate: the cohort's
net directional lean, its share of turnover, and the leg count, over the captured window. Only
the collapsed aggregate is written; no wallet address ever leaves this script, so the committed
snapshot exposes cohort-level net flow, not individual wallets.

    python scripts/build_cohort_snapshot.py /path/to/wallet_labels.parquet
    python scripts/build_cohort_snapshot.py /path/to/wallet_labels.parquet BTC ETH

The labels parquet must have a `wallet` column and a boolean `is_smart` column. Its sha256 and
row counts are recorded in the snapshot for provenance.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from app import cohort
from hlq import data

CHASSIS = ["BTC", "ETH", "SOL", "HYPE"]
IS_SMART_RULE = ">=100 trades and realised PnL >= 80th percentile of active wallets"
OUT = Path(__file__).resolve().parent.parent / "app" / "cohort_snapshot.json"


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def smart_wallets(labels_path: Path) -> tuple[set[str], int, int]:
    df = pd.read_parquet(labels_path, columns=["wallet", "is_smart"])
    smart = set(df.loc[df["is_smart"], "wallet"].tolist())
    return smart, len(df), len(smart)


def coin_aggregate(coin: str, smart: set[str], root: Path) -> dict | None:
    """Net cohort notional, cohort turnover, leg count, and window for one coin, or None if the
    coin has no captured trades. Positive net notional means the cohort was a net buyer."""
    src = str(root / "ws" / coin / "trades" / "**" / "*.parquet")
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.register("smart_df", pd.DataFrame({"w": list(smart)}))
    q = f"""
    WITH legs AS (
      SELECT users_buyer AS w, +notional AS signed_n, notional AS n, trade_ms
      FROM read_parquet('{src}') WHERE users_buyer IS NOT NULL
      UNION ALL
      SELECT users_seller, -notional, notional, trade_ms
      FROM read_parquet('{src}') WHERE users_seller IS NOT NULL
    )
    SELECT SUM(CASE WHEN s.w IS NOT NULL THEN l.signed_n ELSE 0 END) AS net_notional,
           SUM(CASE WHEN s.w IS NOT NULL THEN l.n ELSE 0 END)        AS cohort_notional,
           SUM(CASE WHEN s.w IS NOT NULL THEN 1 ELSE 0 END)          AS n_legs,
           SUM(l.n)                                                  AS total_notional,
           MIN(l.trade_ms) AS t0, MAX(l.trade_ms) AS t1
    FROM legs l LEFT JOIN smart_df s ON l.w = s.w
    """
    r = con.execute(q).fetchone()
    if r is None or r[3] is None:
        return None
    net, cohort_n, n_legs, total_n, t0, t1 = r
    agg = cohort.summarise(net or 0.0, cohort_n or 0.0, n_legs or 0, total_n or 0.0)
    agg["window_start"] = pd.to_datetime(t0, unit="ms", utc=True).date().isoformat()
    agg["window_end"] = pd.to_datetime(t1, unit="ms", utc=True).date().isoformat()
    return agg


def main(labels_path: Path, coins: list[str]) -> None:
    smart, n_rows, n_smart = smart_wallets(labels_path)
    print(f"cohort: {n_smart:,} of {n_rows:,} wallets are is_smart  ({labels_path.name})")

    coins_out: dict[str, dict] = {}
    for coin in coins:
        agg = coin_aggregate(coin, smart, data.DATA_ROOT)
        if agg is None:
            print(f"  {coin:5} no captured trades, skipped")
            continue
        coins_out[coin] = agg
        print(f"  {coin:5} {agg['lean']:9} net ${agg['net_notional']:>16,.0f}  "
              f"share {agg['cohort_share']*100:5.1f}%  legs {agg['n_legs']:>9,}")

    spans = [c["window_start"] for c in coins_out.values()] + [c["window_end"] for c in coins_out.values()]
    snapshot = {
        "provenance": {
            "cohort_sha256_16": sha16(labels_path),
            "n_smart_wallets": n_smart,
            "n_cohort_wallets": n_rows,
            "is_smart_rule": IS_SMART_RULE,
            "window_start": min(spans) if spans else None,
            "window_end": max(spans) if spans else None,
            "coins": list(coins_out),
        },
        "coins": coins_out,
    }
    OUT.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: build_cohort_snapshot.py <wallet_labels.parquet> [COIN ...]")
    main(Path(sys.argv[1]), sys.argv[2:] or CHASSIS)
