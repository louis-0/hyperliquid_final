#!/usr/bin/env python3
"""Build per-wallet PnL and the is_smart label set from the captured trade tape.

Reconstructs each wallet's net flow per coin from the buyer/seller legs of the captured
trades, marks the residual position at the last mark price inside the window, and labels
a wallet is_smart when it is active (at least 100 legs) and its total PnL sits at or above
the 80th percentile of active wallets. Pass --end to freeze the labels at a date, so a
study on later tape uses a label set built only from earlier data.

    python scripts/build_wallet_labels.py
    python scripts/build_wallet_labels.py --end 2026-06-23
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data

TAPE_COINS = ["BTC", "ETH", "SOL", "NEAR", "HYPE", "XMR", "ZEC"]
ACTIVE_LEGS = 100
SMART_PNL_PCT = 80


def last_mark(coin: str, end_ms: int) -> float | None:
    """Last mark close at or before the freeze point."""
    try:
        m = data.load_marks(coin)
    except FileNotFoundError:
        return None
    ts_ms = pd.to_datetime(m["ts"], utc=True).astype("int64") // 1_000_000
    m = m[ts_ms <= end_ms]
    return float(m["close"].iloc[-1]) if len(m) else None


def main(end: str | None) -> None:
    end_ms = (int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
              if end else int(datetime.now(timezone.utc).timestamp() * 1000))

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='6GB'")
    union = " UNION ALL ".join(
        f"SELECT trade_ms, coin, sz, notional, users_buyer, users_seller "
        f"FROM read_parquet('{data.DATA_ROOT / 'ws' / c / 'trades' / '**' / '*.parquet'}')"
        for c in TAPE_COINS)
    q = f"""
    WITH all_trades AS ({union}),
    legs AS (
      SELECT users_buyer AS wallet, coin, +sz AS d_pos, -notional AS d_cash
      FROM all_trades WHERE users_buyer IS NOT NULL AND trade_ms < {end_ms}
      UNION ALL
      SELECT users_seller, coin, -sz, +notional
      FROM all_trades WHERE users_seller IS NOT NULL AND trade_ms < {end_ms}
    )
    SELECT wallet, coin, SUM(d_pos) AS final_position, SUM(d_cash) AS net_cash,
           COUNT(*) AS n_legs
    FROM legs GROUP BY wallet, coin
    """
    df = con.execute(q).df()
    marks = {c: last_mark(c, end_ms) for c in TAPE_COINS}
    df["mark"] = df["coin"].map(marks)
    df = df[df["mark"].notna()]           # a coin without a mark cannot be valued
    df["pnl_usd"] = df["net_cash"] + df["final_position"] * df["mark"]

    wallets = (df.groupby("wallet")
                 .agg(total_pnl_usd=("pnl_usd", "sum"), n_trades=("n_legs", "sum"))
                 .reset_index())
    active = wallets["n_trades"] >= ACTIVE_LEGS
    smart_thr = float(np.percentile(wallets.loc[active, "total_pnl_usd"], SMART_PNL_PCT))
    wallets["is_smart"] = active & (wallets["total_pnl_usd"] >= smart_thr)

    out = data.DATA_ROOT / (f"wallet_labels_{end}.parquet" if end else "wallet_labels.parquet")
    wallets.attrs["freeze"] = end or ""
    wallets.to_parquet(out, index=False)
    frozen = end or "now"
    print(f"labels frozen at {frozen}: {len(wallets):,} wallets, {int(active.sum()):,} active, "
          f"{int(wallets['is_smart'].sum()):,} is_smart (pnl >= ${smart_thr:,.0f})")
    print(f"wrote {out.relative_to(data.DATA_ROOT.parent)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--end", default=None, help="freeze date YYYY-MM-DD (default: full tape)")
    main(p.parse_args().end)
