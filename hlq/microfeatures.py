"""Hourly features from the WebSocket capture: book imbalance, taker flow, premium, cohort flow.

Aggregates the captured tape to the hour so it can join the hourly REST feature matrix
(hlq.features). Quote and trade rows are bucketed by their exchange timestamp, ctx rows by
capture time (the payload carries no exchange time), so a feature at hour H is observable by
the end of H, the same timing as the candle features. The cohort column applies a frozen
is_smart wallet set with the same buyer/seller leg convention as the aggregate snapshot;
only the collapsed hourly share is returned.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from hlq import data

MICRO = ["obi_mean", "spread_bps_mean", "taker_imbalance",
         "premium_mean", "premium_last"]
# ctx_funding_last is computed but excluded from MICRO: the ctx funding field carries the
# venue's next settlement value itself, so as a feature it restates the target.
COHORT = ["cohort_net_share"]


def hourly(coin: str, root: Path = data.DATA_ROOT, smart: set[str] | None = None) -> pd.DataFrame:
    """Hourly micro-feature frame for one coin, joined across the captured channels.

    Returns a `hour` column plus MICRO (and COHORT when a smart set is given). Hours missing
    from any channel carry NaN for that channel's columns and are dropped by the caller's join.
    """
    ws = root / "ws" / coin
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")

    # epoch-ms floored to its hour; to_timestamp takes seconds
    bbo = con.execute(f"""
        SELECT to_timestamp((bbo_ms // 3600000) * 3600) AS hour,
               AVG((bid_sz - ask_sz) / NULLIF(bid_sz + ask_sz, 0))                 AS obi_mean,
               AVG((ask_px - bid_px) / NULLIF((ask_px + bid_px) / 2, 0)) * 1e4     AS spread_bps_mean
        FROM read_parquet('{ws / "bbo" / "**" / "*.parquet"}')
        GROUP BY 1
    """).df()
    trades = con.execute(f"""
        SELECT to_timestamp((trade_ms // 3600000) * 3600) AS hour,
               SUM(CASE WHEN side = 'B' THEN notional ELSE -notional END)
                   / NULLIF(SUM(notional), 0)                                       AS taker_imbalance
        FROM read_parquet('{ws / "trades" / "**" / "*.parquet"}')
        GROUP BY 1
    """).df()
    ctx = con.execute(f"""
        SELECT to_timestamp((captured_ms // 3600000) * 3600) AS hour,
               AVG(premium)                        AS premium_mean,
               arg_max(premium, captured_ms)       AS premium_last,
               arg_max(funding, captured_ms)       AS ctx_funding_last
        FROM read_parquet('{ws / "ctx" / "**" / "*.parquet"}')
        GROUP BY 1
    """).df()
    out = bbo.merge(trades, on="hour", how="outer").merge(ctx, on="hour", how="outer")

    if smart is not None:
        con.register("smart_df", pd.DataFrame({"w": list(smart)}))
        src = ws / "trades" / "**" / "*.parquet"
        cohort = con.execute(f"""
        WITH legs AS (
          SELECT trade_ms, users_buyer AS w, +notional AS signed_n, notional AS n
          FROM read_parquet('{src}') WHERE users_buyer IS NOT NULL
          UNION ALL
          SELECT trade_ms, users_seller, -notional, notional
          FROM read_parquet('{src}') WHERE users_seller IS NOT NULL
        )
        SELECT to_timestamp((trade_ms // 3600000) * 3600) AS hour,
               SUM(CASE WHEN s.w IS NOT NULL THEN l.signed_n ELSE 0 END)
                   / NULLIF(SUM(l.n), 0)                                            AS cohort_net_share
        FROM legs l LEFT JOIN smart_df s ON l.w = s.w
        GROUP BY 1
        """).df()
        out = out.merge(cohort, on="hour", how="outer")

    out["hour"] = pd.to_datetime(out["hour"], utc=True).astype("datetime64[ns, UTC]")
    return out.sort_values("hour").reset_index(drop=True)
