"""Aggregate smart-money cohort snapshot: the summariser and the loader the dashboard reads."""
from __future__ import annotations

import json
from pathlib import Path


def summarise(net_notional: float, cohort_notional: float, n_legs: int,
              total_notional: float) -> dict:
    """Assemble a coin's aggregate from the raw sums: attach the cohort's share of turnover and
    its directional lean. Kept pure so the build script and its test agree on the arithmetic. No
    wallet identity is involved; the inputs are already collapsed across the cohort."""
    return {
        "net_notional": float(net_notional),
        "cohort_notional": float(cohort_notional),
        "n_legs": int(n_legs),
        "total_notional": float(total_notional),
        "cohort_share": float(cohort_notional / total_notional) if total_notional else 0.0,
        "lean": "net long" if net_notional > 0 else "net short" if net_notional < 0 else "flat",
    }


def load_snapshot(path) -> dict | None:
    """Read the committed aggregate cohort snapshot, or None if it has not been built."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None
