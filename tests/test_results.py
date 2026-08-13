"""hlq.results: a ResultsStore keyed by config hash, refusing silent overwrites."""
import pytest

from hlq import results


def test_config_hash_is_deterministic_and_order_independent():
    a = results.config_hash({"strategy": "chassis", "coins": ["BTC", "ETH"], "cost": 0.0011})
    b = results.config_hash({"cost": 0.0011, "coins": ["BTC", "ETH"], "strategy": "chassis"})
    assert a == b                                                  # key order must not matter
    assert a != results.config_hash({"strategy": "chassis", "coins": ["BTC"], "cost": 0.0011})


def test_save_then_load_round_trips(tmp_path):
    store = results.ResultsStore(tmp_path)
    cfg = {"strategy": "chassis", "coins": ["BTC"]}
    res = {"sharpe": 21.265, "n_days": 91}
    store.save(cfg, res, data_span="2026-03-30/2026-06-28", saved_at="2026-08-11T00:00:00+00:00")
    rec = store.load(cfg)
    assert rec["config"] == cfg
    assert rec["result"] == res
    assert rec["data_span"] == "2026-03-30/2026-06-28"
    assert rec["saved_at"] == "2026-08-11T00:00:00+00:00"
    assert rec["config_hash"] == results.config_hash(cfg)


def test_save_refuses_to_overwrite(tmp_path):
    store = results.ResultsStore(tmp_path)
    cfg = {"strategy": "chassis"}
    store.save(cfg, {"sharpe": 1.0})
    with pytest.raises(FileExistsError):
        store.save(cfg, {"sharpe": 2.0})                          # same config, no silent clobber


def test_exists_reflects_saved_state(tmp_path):
    store = results.ResultsStore(tmp_path)
    cfg = {"strategy": "basis_drift"}
    assert not store.exists(cfg)
    store.save(cfg, {"sharpe": 5.024})
    assert store.exists(cfg)


def test_record_once_is_idempotent(tmp_path):
    store = results.ResultsStore(tmp_path)
    cfg = {"strategy": "chassis"}
    assert results.record_once(store, cfg, {"sharpe": 1.0}) is True    # first write
    assert results.record_once(store, cfg, {"sharpe": 2.0}) is False   # already present, no raise
    assert store.load(cfg)["result"] == {"sharpe": 1.0}                # original preserved


def test_record_run_builds_config_and_is_idempotent(tmp_path):
    saved, h = results.record_run(tmp_path, "chassis", {"coins": ["BTC"]}, {"sharpe": 1.0})
    assert saved is True
    assert h == results.config_hash({"strategy": "chassis", "coins": ["BTC"]})
    saved2, h2 = results.record_run(tmp_path, "chassis", {"coins": ["BTC"]}, {"sharpe": 2.0})
    assert saved2 is False and h2 == h                                  # same config, idempotent