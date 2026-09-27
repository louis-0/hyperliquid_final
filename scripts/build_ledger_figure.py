#!/usr/bin/env python3
"""Draw the three ledgers' cumulative net P&L on one axis, from the recorded ledger files.

Reads results/carry_ledger_daily.csv (basket daily net, bps), results/ml_gated_ledger_{coin}.csv
and results/obi_ledger_{coin}.csv.gz (per-trade nets, bps). Each signal strategy's trades are
pooled in exit order and the running sum divided by its coin count, so every curve is the
per-coin mean net. The axis is symlog, linear inside 100 bps.

    python scripts/build_ledger_figure.py
    python scripts/build_ledger_figure.py --out figures/fig4_ledger_trilogy.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data

RESULTS_ROOT = data.DATA_ROOT.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "app" / "static" / "ledger_trilogy.png"
GATED_COINS = ["BTC", "ETH", "HYPE"]
OBI_COINS = ["BTC", "ETH", "SOL", "HYPE"]


def pooled(paths: list[Path]) -> pd.Series:
    """Running per-coin mean of net bps across the pooled trades, indexed by exit time."""
    t = pd.concat([pd.read_csv(p, usecols=["exit", "net_bps"]) for p in paths])
    t["exit"] = pd.to_datetime(t["exit"], utc=True)
    t = t.sort_values("exit")
    return pd.Series(t["net_bps"].cumsum().to_numpy() / len(paths), index=t["exit"])


def main(out: Path) -> None:
    carry = pd.read_csv(RESULTS_ROOT / "carry_ledger_daily.csv", index_col="day", parse_dates=True)["basket"].cumsum()
    gated = pooled([RESULTS_ROOT / f"ml_gated_ledger_{c}.csv" for c in GATED_COINS])
    obi = pooled([RESULTS_ROOT / f"obi_ledger_{c}.csv.gz" for c in OBI_COINS])

    fig, ax = plt.subplots(figsize=(8.36, 4.6), dpi=200)      # the dashboard's 836 px content width, at 2x
    ax.plot(carry.index, carry.to_numpy(), lw=2.2,
            label=f"carry, one round trip per coin (ends {carry.iloc[-1]:+.0f} bps)")
    ax.plot(gated.index, gated.to_numpy(), lw=2.2,
            label=f"ML-gated carry, {len(gated):,} round trips (ends {gated.iloc[-1]:+.0f} bps)")
    ax.plot(obi.index, obi.to_numpy(), lw=2.2,
            label=f"OBI scalp, {len(obi):,} round trips (ends {obi.iloc[-1] / 1e3:+.0f}k bps)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylabel("cumulative net P&L (bps, log-scaled beyond 100)")
    ax.set_title("Three strategies, one cost model: net P&L by round-trip count")
    ax.legend(loc="lower left", frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"carry ends {carry.iloc[-1]:+.1f} bps over {len(carry)} days; "
          f"gated {len(gated)} trades ends {gated.iloc[-1]:+.1f}; obi {len(obi):,} trades ends {obi.iloc[-1]:+,.0f}")
    root = Path(__file__).resolve().parent.parent
    print(f"wrote {out.relative_to(root) if out.is_relative_to(root) else out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=OUT)
    main(p.parse_args().out)
