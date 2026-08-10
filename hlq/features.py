"""Leakage-safe feature matrix for the next-hour funding-timing model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hlq import data

FEATURES = [
    "basis", "spot_ret_1h", "perp_ret_1h", "spot_ret_24h", "perp_ret_24h",
    "basis_lag_1", "basis_lag_4", "basis_lag_24",
    "funding_lag_1", "funding_lag_4", "funding_lag_24",
    "basis_roll_std_24", "funding_roll_mean_24", "funding_roll_std_24",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def build_features(funding: pd.DataFrame, spot: pd.DataFrame, perp: pd.DataFrame) -> pd.DataFrame:
    """Hourly feature matrix with the next hour's funding as the target.

    Perp, spot, and funding are floored to the hour and inner-joined (funding left-joined,
    missing filled zero). Every feature is built from current or lagged values only (lags,
    trailing rolling windows, and backward returns), so no feature can see the future. The
    target is the *next* hour's funding (`target_fund`) and its sign (`target_fund_sign`),
    obtained by a forward shift; rows without full history or a next-hour value are dropped.
    """
    f = funding.assign(hour=data.floor_hour(funding["ts"]))[["hour", "funding_rate"]]
    s = spot.assign(hour=data.floor_hour(spot["ts"]))[["hour", "close"]].rename(columns={"close": "spot"})
    p = perp.assign(hour=data.floor_hour(perp["ts"]))[["hour", "close"]].rename(columns={"close": "perp"})

    df = (p.merge(s, on="hour", how="inner")
            .merge(f, on="hour", how="left")
            .sort_values("hour").reset_index(drop=True))
    df["funding_rate"] = df["funding_rate"].fillna(0.0)

    df["basis"] = (df["perp"] - df["spot"]) / df["spot"]
    df["spot_ret_1h"] = df["spot"].pct_change(1, fill_method=None)
    df["perp_ret_1h"] = df["perp"].pct_change(1, fill_method=None)
    df["spot_ret_24h"] = df["spot"].pct_change(24, fill_method=None)
    df["perp_ret_24h"] = df["perp"].pct_change(24, fill_method=None)

    for lag in (1, 4, 24):
        df[f"basis_lag_{lag}"] = df["basis"].shift(lag)
        df[f"funding_lag_{lag}"] = df["funding_rate"].shift(lag)

    df["basis_roll_std_24"] = df["basis"].rolling(24).std()
    df["funding_roll_mean_24"] = df["funding_rate"].rolling(24).mean()
    df["funding_roll_std_24"] = df["funding_rate"].rolling(24).std()

    hour = df["hour"].dt.hour
    dow = df["hour"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df["target_fund"] = df["funding_rate"].shift(-1)
    df["target_fund_sign"] = (df["target_fund"] > 0).astype(int)

    return df.dropna().reset_index(drop=True)