#!/usr/bin/env python3
"""Build the ledger snapshot the dashboard reads.

Collects the three per-trade ledgers recorded under results/ (the carry basket at the default
notional, the model-gated carry, and the imbalance scalp on all four coins) into one per-strategy
table: window, round trips, net per coin, basket net, and the record hash. Only the recorded
summaries are read; the ledgers themselves stay under results/.

    python scripts/build_ledger_snapshot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data

RESULTS_ROOT = data.DATA_ROOT.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "app" / "ledger_snapshot.json"
CARRY_NOTIONAL = 10_000.0
COST_MODEL = ("taker 4.5 bp per side; slippage walked from the captured book for the carry and taken "
              "from the measured bar spread for the scalp; a hedged toggle pays 18 bp in fees")


def records() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS_ROOT.glob("*.json"))]


def pick(recs: list[dict], strategy: str, **match) -> dict:
    """The one record of `strategy` whose config carries every key in `match`."""
    hits = [r for r in recs if r["config"]["strategy"] == strategy
            and all(r["config"].get(k) == v for k, v in match.items())]
    if len(hits) != 1:
        raise SystemExit(f"{strategy} {match}: {len(hits)} records, expected one")
    return hits[0]


def main() -> None:
    recs = records()
    carry = pick(recs, "carry_ledger", notional=CARRY_NOTIONAL)
    gated = pick(recs, "ml_gated_ledger")
    obi = pick(recs, "obi_ledger", coins=["BTC", "ETH", "SOL", "HYPE"])

    coins_c = [c for c in carry["result"] if c != "basket"]
    strategies = [
        {"name": "untimed carry", "window": carry["config"]["window"],
         "round_trips": len(coins_c),
         "per_coin_bps": {c: carry["result"][c]["net_bps"] for c in coins_c},
         "basket_bps": carry["result"]["basket"]["net_bps"],
         "record": carry["config_hash"]},
        {"name": "ML-gated carry", "window": gated["config"]["window"],
         "round_trips": sum(v["trades"] for v in gated["result"].values()),
         "per_coin_bps": {c: v["gated_net_bps"] for c, v in gated["result"].items()},
         "basket_bps": round(sum(v["gated_net_bps"] for v in gated["result"].values()) / len(gated["result"]), 2),
         "record": gated["config_hash"]},
        {"name": "imbalance scalp", "window": obi["config"]["window"],
         "round_trips": sum(v["trades"] for v in obi["result"].values()),
         "per_coin_bps": {c: v["total_net_bps"] for c, v in obi["result"].items()},
         "basket_bps": round(sum(v["total_net_bps"] for v in obi["result"].values()) / len(obi["result"]), 1),
         "record": obi["config_hash"]},
    ]

    print(f"{'strategy':<16} {'window':<23} {'round trips':>11} {'basket bps':>12}  record")
    for s in strategies:
        print(f"  {s['name']:<14} {s['window']:<23} {s['round_trips']:>11,} {s['basket_bps']:>+12,.1f}  {s['record']}")

    snapshot = {"provenance": {"records": [s["record"] for s in strategies],
                               "cost_model": COST_MODEL,
                               "carry_notional": CARRY_NOTIONAL},
                "strategies": strategies}
    OUT.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
