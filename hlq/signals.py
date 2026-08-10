"""PnL engines and regime labelling for the long-spot / short-perp basis trade (He et al. 2024)."""
from __future__ import annotations

import pandas as pd

from hlq import data


def funding_carry_pnl(funding: pd.DataFrame) -> pd.DataFrame:
    """Hourly funding-only PnL; the short-perp leg receives funding when it is positive."""
    df = funding.sort_values("ts").reset_index(drop=True)
    return pd.DataFrame({"ts": df["ts"], "pnl": df["funding_rate"].astype(float)})


def basis_drift_pnl(funding: pd.DataFrame, spot: pd.DataFrame, perp: pd.DataFrame) -> pd.DataFrame:
    """Hourly PnL of long-spot/short-perp: residual basis drift plus funding.

    Drift = (d_spot - d_perp) / prev_perp on the hour-floored inner join of the three legs;
    funding is received when positive. A perfect hedge gives zero drift, leaving funding only.
    Both legs are normalised by the prior perp price (a small-basis approximation of the exact
    per-leg return, exact in the limit where the two legs' prior prices coincide).
    """
    f = funding.assign(hour=data.floor_hour(funding["ts"]))[["hour", "funding_rate"]]
    s = spot.assign(hour=data.floor_hour(spot["ts"]))[["hour", "close"]].rename(columns={"close": "spot"})
    p = perp.assign(hour=data.floor_hour(perp["ts"]))[["hour", "close"]].rename(columns={"close": "perp"})

    df = (p.merge(s, on="hour", how="inner")
            .merge(f, on="hour", how="left")
            .sort_values("hour").reset_index(drop=True))
    df["funding_rate"] = df["funding_rate"].fillna(0.0)
    df["drift_pnl"] = (df["spot"].diff() - df["perp"].diff()) / df["perp"].shift(1)
    df["funding_pnl"] = df["funding_rate"]
    df["pnl"] = df["drift_pnl"] + df["funding_pnl"]
    df = df.dropna(subset=["pnl"]).reset_index(drop=True).rename(columns={"hour": "ts"})
    return df[["ts", "spot", "perp", "drift_pnl", "funding_pnl", "pnl"]]


def regime_label(perp: pd.DataFrame, rolling_days: int = 30,
                 bear: float = -0.10, bull: float = 0.10) -> pd.Series:
    """Daily macro-regime label from a reference perp's rolling return.

    Each UTC day is represented by its last mark; the `rolling_days` return of that daily
    series labels the day 'bear' (return below `bear`), 'bull' (above `bull`), or 'calm'
    otherwise. Days without `rolling_days` of prior history are 'calm'. Indexed by day, so a
    per-day strategy series can be joined to it.
    """
    df = perp.sort_values("ts")
    day = pd.to_datetime(df["ts"], utc=True).dt.floor("1D")
    daily = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(day)).groupby(level=0).last()
    roll_ret = daily.pct_change(periods=rolling_days, fill_method=None)
    label = pd.Series("calm", index=daily.index, dtype="object", name="regime")
    label[roll_ret < bear] = "bear"
    label[roll_ret > bull] = "bull"
    return label
