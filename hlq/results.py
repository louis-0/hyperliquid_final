"""ResultsStore: record each backtest run keyed by a config hash, with no silent overwrite."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def config_hash(config: dict) -> str:
    """Deterministic short hash of a config dict, independent of key order."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class ResultsStore:
    """Filesystem store of run records, one JSON file per config hash. A save for a config
    that is already recorded raises rather than overwriting, so results are never clobbered
    silently. Each record carries the config, its hash, the data span, a UTC timestamp, and
    the result payload."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, config: dict) -> Path:
        return self.root / f"{config_hash(config)}.json"

    def exists(self, config: dict) -> bool:
        return self._path(config).exists()

    def save(self, config: dict, result: dict, data_span: str = "",
             saved_at: str | None = None) -> Path:
        path = self._path(config)
        if path.exists():
            raise FileExistsError(f"result already recorded for this config: {path.name}")
        record = {
            "config": config,
            "config_hash": config_hash(config),
            "data_span": data_span,
            "saved_at": saved_at or datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        path.write_text(json.dumps(record, indent=2, sort_keys=True))
        return path

    def load(self, config: dict) -> dict:
        return json.loads(self._path(config).read_text())


def record_once(store: ResultsStore, config: dict, result: dict, data_span: str = "") -> bool:
    """Save the result unless one is already recorded for this config. Returns True if it was
    saved, False if a record already existed, so a repeated run is an idempotent write."""
    if store.exists(config):
        return False
    store.save(config, result, data_span=data_span)
    return True


def record_run(root, strategy: str, config_extra: dict, result: dict,
               data_span: str = "") -> tuple[bool, str]:
    """Record a runner's headline result under `root`, keyed by strategy plus `config_extra`.
    Returns (saved, config_hash), saved False if a record for this config already existed."""
    config = {"strategy": strategy, **config_extra}
    saved = record_once(ResultsStore(root), config, result, data_span=data_span)
    return saved, config_hash(config)