"""hlq.features: leakage-safe feature matrix for the next-hour funding-timing model."""
import numpy as np
import pandas as pd
import pytest

from hlq import features


def _hourly(n, funding=None, spot=None, perp=None):
    hours = pd.date_range("2026-06-01", periods=n, freq="1h", tz="UTC")
    i = np.arange(n, dtype=float)
    perp = 100.0 + i if perp is None else perp
    spot = 100.0 + 0.9 * i if spot is None else spot
    funding = np.full(n, 0.0002) if funding is None else funding
    return (pd.DataFrame({"ts": hours, "funding_rate": funding}),
            pd.DataFrame({"ts": hours, "close": spot}),
            pd.DataFrame({"ts": hours, "close": perp}))


def test_build_features_has_18_features_and_target():
    f, s, p = _hourly(40)
    df = features.build_features(f, s, p)
    assert len(features.FEATURES) == 18
    for col in features.FEATURES:
        assert col in df.columns
    assert {"target_fund", "target_fund_sign"} <= set(df.columns)
    assert not df.empty


def test_no_nans_in_feature_matrix():
    f, s, p = _hourly(40)
    df = features.build_features(f, s, p)
    assert not df[features.FEATURES].isna().any().any()
    assert not df["target_fund"].isna().any()


def test_basis_is_perp_over_spot():
    f, s, p = _hourly(40)
    df = features.build_features(f, s, p)
    assert df["basis"].to_numpy() == pytest.approx(((df["perp"] - df["spot"]) / df["spot"]).to_numpy())


def test_features_use_past_and_target_uses_future():
    # funding[i] = i on a gapless hourly grid, so the shift directions are readable:
    #   lags reach BACKWARD (past), the target reaches FORWARD (next hour) -> no look-ahead.
    n = 40
    f, s, p = _hourly(n, funding=np.arange(n, dtype=float))
    df = features.build_features(f, s, p)
    assert (df["funding_lag_1"] == df["funding_rate"] - 1).all()
    assert (df["funding_lag_4"] == df["funding_rate"] - 4).all()
    assert (df["target_fund"] == df["funding_rate"] + 1).all()


def test_changing_a_future_row_leaves_earlier_features_unchanged():
    # Strong no-look-ahead guard: perturb a late hour's inputs; every feature at every
    # earlier hour must be byte-identical, because no feature may reach forward in time.
    n = 48
    k = 40
    df0 = features.build_features(*_hourly(n))
    f1, s1, p1 = _hourly(n)
    p1.loc[k, "close"] += 50.0
    s1.loc[k, "close"] += 50.0
    f1.loc[k, "funding_rate"] += 0.01
    df1 = features.build_features(f1, s1, p1)

    hour_k = pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(hours=k)
    merged = df0.merge(df1, on="hour", suffixes=("_0", "_1"))
    earlier = merged[merged["hour"] < hour_k]
    assert not earlier.empty
    for col in features.FEATURES:
        assert (earlier[f"{col}_0"] == earlier[f"{col}_1"]).all(), col