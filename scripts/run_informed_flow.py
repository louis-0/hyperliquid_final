#!/usr/bin/env python3
"""Informed-flow study: does the frozen cohort's net flow predict the next move? Built on hlq.

Aggregates the captured trade legs into 30-second bars and compares two signals on the same
bars: the frozen is_smart cohort's rolling net flow (identity) and the raw aggressor flow
(anonymous). Reports the rank correlation with forward returns at several horizons and the
top-decile forward move against the round-trip taker cost (median top-of-book spread from
the captured quotes plus taker fees). Pass --start so the evaluation window begins after
the label freeze; the labels then come only from earlier tape.

    python scripts/run_informed_flow.py data/wallet_labels.parquet --start 2026-06-23
    python scripts/run_informed_flow.py data/wallet_labels.parquet --start 2026-06-23 BTC ETH
"""
from __future__ import annotations

import argparse
import glob
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, flow, results
from hlq.costs import CostModel

CHASSIS = ["BTC", "ETH", "SOL", "HYPE"]
HORIZONS = [1, 2, 10, 30]              # forward bars: 30s, 1m, 5m, 15m
COST = CostModel()
RESULTS_ROOT = data.DATA_ROOT.parent / "results"
COLS = ["trade_ms", "side", "px", "notional", "users_buyer", "users_seller"]


def median_spread_bps(coin: str, sample: int = 12) -> float:
    """Median top-of-book spread over a sample of captured quote files, in bps."""
    vals = []
    for f in sorted(glob.glob(str(data.DATA_ROOT / "ws" / coin / "bbo" / "*" / "*.parquet")))[:sample]:
        d = pd.read_parquet(f, columns=["bid_px", "ask_px"])
        mid = 0.5 * (d["bid_px"] + d["ask_px"])
        vals.append(float(((d["ask_px"] - d["bid_px"]) / mid * 1e4).median()))
    return float(np.nanmedian(vals)) if vals else float("nan")


def coin_bars(coin: str, smart: set[str], start_ms: int, end_ms: int) -> pd.DataFrame:
    """Per-file bar aggregation over the evaluation tape, combined into one bar series."""
    parts = []
    for f in sorted(glob.glob(str(data.DATA_ROOT / "ws" / coin / "trades" / "*" / "*.parquet"))):
        d = pd.read_parquet(f, columns=COLS)
        d = d[(d["trade_ms"] >= start_ms) & (d["trade_ms"] < end_ms)]
        if d.empty:
            continue
        parts.append(flow.bar_aggregate(flow.signed_flows(d, smart)))
    if not parts:
        return pd.DataFrame()
    return (pd.concat(parts).groupby(level=0)
              .agg({"inf_net": "sum", "taker": "sum", "px": "last", "ntl": "sum"})
              .sort_index())


def main(labels_path: Path, start: str | None, end: str | None, coins: list[str]) -> None:
    lab = pd.read_parquet(labels_path, columns=["wallet", "is_smart"])
    smart = set(lab.loc[lab["is_smart"], "wallet"].tolist())
    start_ms = (int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
                if start else 0)
    end_ms = (int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
              if end else 1 << 62)
    print(f"cohort: {len(smart):,} is_smart wallets ({labels_path.name}); "
          f"evaluation {start or 'tape start'} to {end or 'tape end'}")
    print("=" * 96)
    print("Informed-flow study: cohort net flow (identity) vs raw taker flow (anonymous), 30s bars")
    print("=" * 96)
    print(f"\n{'coin':<6} {'bars':>7} {'spread':>7} | IC inf_rate @30s/1m/5m/15m | IC taker @30s/15m"
          f" | top-decile 15m vs cost")
    print("-" * 96)

    table: dict[str, dict] = {}
    for coin in coins:
        g = coin_bars(coin, smart, start_ms, end_ms)
        if len(g) < 200:
            print(f"  {coin:<6} (insufficient bars)")
            continue
        g["inf_rate"] = g["inf_net"].rolling(flow.FLOW_WIN).sum()
        g["taker_rate"] = g["taker"].rolling(flow.FLOW_WIN).sum()
        spread = median_spread_bps(coin)
        rt_cost = spread + 2 * COST.taker_fee * 1e4
        ic_inf = {h: flow.spearman_ic(g["inf_rate"], flow.fwd_ret_bps(g["px"], h)) for h in HORIZONS}
        ic_tak = {h: flow.spearman_ic(g["taker_rate"], flow.fwd_ret_bps(g["px"], h)) for h in HORIZONS}
        top = flow.top_decile_move(g["inf_rate"], flow.fwd_ret_bps(g["px"], 30))
        print(f"  {coin:<6} {len(g):>7,} {spread:>6.1f}b | "
              f"{ic_inf[1]:+.3f} {ic_inf[2]:+.3f} {ic_inf[10]:+.3f} {ic_inf[30]:+.3f}      | "
              f"{ic_tak[1]:+.3f} {ic_tak[30]:+.3f}    | {top:+.1f}bp vs {rt_cost:.1f}bp = {top - rt_cost:+.1f}")
        table[coin] = {"bars": int(len(g)), "spread_bps": round(spread, 2),
                       "ic_inf_rate": {str(h): round(ic_inf[h], 4) for h in HORIZONS},
                       "ic_taker": {str(h): round(ic_tak[h], 4) for h in HORIZONS},
                       "top_decile_15m_bps": round(top, 2), "rt_cost_bps": round(rt_cost, 2)}

    if table:
        span = f"{start or 'tape-start'}/eval"
        saved, h = results.record_run(RESULTS_ROOT, "informed_flow",
                                      {"coins": list(table), "start": start, "end": end,
                                       "labels": labels_path.name}, table, span)
        print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("labels", help="wallet labels parquet (wallet, is_smart)")
    p.add_argument("--start", default=None, help="evaluation start YYYY-MM-DD (after the label freeze)")
    p.add_argument("--end", default=None, help="evaluation end YYYY-MM-DD (default: tape end)")
    p.add_argument("coins", nargs="*", default=[])
    args = p.parse_args()
    main(Path(args.labels), args.start, args.end, args.coins or CHASSIS)
