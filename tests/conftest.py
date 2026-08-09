"""Shared fixtures. Tests run off the committed data_sample/, not the gitignored data/."""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATA_SAMPLE = REPO / "data_sample"


@pytest.fixture(scope="session")
def data_sample() -> Path:
    assert DATA_SAMPLE.exists(), "data_sample/ is missing"
    return DATA_SAMPLE
