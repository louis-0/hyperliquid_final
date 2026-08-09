"""Parquet loaders (funding/marks/spot), hour alignment, and a look-ahead contamination guard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"      # hyperliquid/data (gitignored)


def _read_layer(root, layer: str, coin: str) -> pd.DataFrame:
    path = Path(root) / layer / f"{coin}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def load_funding(coin: str, root=DATA_ROOT) -> pd.DataFrame:
    return _read_layer(root, "funding", coin)


def load_marks(coin: str, root=DATA_ROOT) -> pd.DataFrame:
    return _read_layer(root, "marks", coin)


def load_spot(coin: str, root=DATA_ROOT) -> pd.DataFrame:
    return _read_layer(root, "spot", coin)


def floor_hour(ts: pd.Series) -> pd.Series:
    """Floor a UTC timestamp series to the hour, so funding (HH:00) and candles (HH:59:59) align."""
    return pd.to_datetime(ts, utc=True).dt.floor("1h")


def guard_max_ts(df: pd.DataFrame, cutoff, ts_col: str = "ts") -> pd.DataFrame:
    """Raise if any row is timestamped after cutoff; guards point-in-time work against look-ahead."""
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tz is None else cutoff.tz_convert("UTC")
    ts = pd.to_datetime(df[ts_col], utc=True)
    n_after = int((ts > cutoff).sum())
    if n_after:
        raise ValueError(f"{n_after} row(s) after cutoff {cutoff}")
    return df
