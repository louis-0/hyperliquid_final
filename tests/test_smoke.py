"""Harness smoke: hlq imports and the fixtures resolve."""
import hlq


def test_hlq_imports():
    assert hlq.__version__


def test_data_sample_present(data_sample):
    assert (data_sample / "funding" / "BTC.parquet").exists()
    assert (data_sample / "marks" / "BTC.parquet").exists()
