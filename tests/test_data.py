"""hlq.data: parquet loaders, hour alignment, and the look-ahead (max_ts) guard."""
import pandas as pd
import pytest

from hlq import data


def test_load_funding_returns_utc_sorted(data_sample):
    df = data.load_funding("BTC", root=data_sample)
    assert {"ts", "funding_rate"} <= set(df.columns)
    assert str(df["ts"].dt.tz) == "UTC"
    assert df["ts"].is_monotonic_increasing


def test_load_marks_has_close(data_sample):
    df = data.load_marks("BTC", root=data_sample)
    assert "close" in df.columns
    assert str(df["ts"].dt.tz) == "UTC"


def test_load_missing_coin_raises(data_sample):
    with pytest.raises(FileNotFoundError):
        data.load_funding("NOTACOIN", root=data_sample)


def test_floor_hour_drops_sub_hour_parts():
    ts = pd.to_datetime(["2026-06-01 05:59:59.999", "2026-06-01 05:00:00.095"], utc=True).to_series()
    assert (data.floor_hour(ts) == pd.Timestamp("2026-06-01 05:00", tz="UTC")).all()


def test_guard_max_ts_passes_when_clean():
    df = pd.DataFrame({"ts": pd.to_datetime(["2026-06-01", "2026-06-02"], utc=True)})
    assert len(data.guard_max_ts(df, "2026-06-02")) == 2


def test_guard_max_ts_rejects_rows_after_cutoff():
    df = pd.DataFrame({"ts": pd.to_datetime(["2026-06-01", "2026-06-03"], utc=True)})
    with pytest.raises(ValueError):
        data.guard_max_ts(df, "2026-06-02")
