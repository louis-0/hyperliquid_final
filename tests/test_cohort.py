"""app.cohort: the aggregate smart-money summariser, loader, and the aggregate-only guarantee."""
import json
from pathlib import Path

from app import cohort

SNAPSHOT = Path(__file__).resolve().parent.parent / "app" / "cohort_snapshot.json"


def test_summarise_share_and_lean():
    s = cohort.summarise(net_notional=100.0, cohort_notional=400.0, n_legs=8, total_notional=1000.0)
    assert s["cohort_share"] == 0.4                     # 400 / 1000 of turnover is cohort
    assert s["lean"] == "net long"                      # positive net notional == net buyer
    assert cohort.summarise(-5.0, 10.0, 2, 500.0)["lean"] == "net short"


def test_summarise_guards_zero_turnover():
    s = cohort.summarise(0.0, 0.0, 0, 0.0)
    assert s["cohort_share"] == 0.0                     # no division by zero when nothing traded
    assert s["lean"] == "flat"                          # a flat or empty cohort is not "net short"


def test_load_snapshot_missing_returns_none(tmp_path):
    assert cohort.load_snapshot(tmp_path / "nope.json") is None


def test_committed_snapshot_is_aggregate_only():
    # the shipped artefact must expose cohort-level numbers only, never a wallet address
    raw = SNAPSHOT.read_text()
    assert "0x" not in raw.lower()                      # no address leaked into the aggregate
    snap = json.loads(raw)
    assert snap["provenance"]["n_smart_wallets"] > 0
    for c in snap["coins"].values():
        assert set(c) >= {"net_notional", "cohort_share", "lean", "n_legs"}
