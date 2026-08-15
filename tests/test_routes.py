"""app.main: FastAPI route smoke tests, run off data_sample via HLQ_DATA_ROOT."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SAMPLE = str(Path(__file__).resolve().parent.parent / "data_sample")
VERDICTS = {"do not deploy", "marginal: maker or lower fee tier only", "deployable"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("HLQ_DATA_ROOT", SAMPLE)
    from app.main import app
    return TestClient(app)


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Cost-floor calculator" in r.text


def test_api_net_returns_valid_metrics(client):
    m = client.get("/api/net", params={"coin": "BTC", "fee_bps": 11, "borrow_bps": 1, "drift_bps": 1}).json()
    assert {"net_sharpe", "net_apr", "gross_apr", "verdict", "series"} <= m.keys()
    assert m["verdict"] in VERDICTS


def test_higher_cost_lowers_net_sharpe(client):
    cheap = client.get("/api/net", params={"coin": "BTC", "fee_bps": 3, "borrow_bps": 0, "drift_bps": 0}).json()
    heavy = client.get("/api/net", params={"coin": "BTC", "fee_bps": 20, "borrow_bps": 5, "drift_bps": 5}).json()
    assert heavy["net_sharpe"] < cheap["net_sharpe"]           # the cost floor bites


def test_unknown_coin_returns_error(client):
    assert "error" in client.get("/api/net", params={"coin": "NOPE"}).json()
