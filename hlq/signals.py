"""Strategy PnL engines for the long-spot / short-perp basis trade (He et al. 2024)."""
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
