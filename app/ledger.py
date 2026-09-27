"""Ledger snapshot: the loader for the three per-trade ledgers the dashboard shows."""
from __future__ import annotations

import json
from pathlib import Path


def load_snapshot(path) -> dict | None:
    """Read the committed ledger snapshot, or None if it has not been built."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None
