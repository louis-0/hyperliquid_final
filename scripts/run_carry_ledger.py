#!/usr/bin/env python3
"""Trade-ledger P&L for the carry basket on the captured window, built on the hlq package.

Books the drop-SOL basket as one round trip per coin: entry and exit slippage measured by
walking the first and last captured order books at the target notional, taker fees charged
per leg on both sides, spot borrow charged per day, funding and basis drift accrued from the
realised hourly series. Every cost in the ledger is either
measured from the capture or an explicit fee-schedule constant. At the default notional the per-coin
round trips are written to results/carry_ledger.csv and the daily net series, in basis points,
to results/carry_ledger_daily.csv.

    python scripts/run_carry_ledger.py
    python scripts/run_carry_ledger.py --notional 100000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, execution, results, signals, stats
from hlq.costs import CostModel

COINS = ["BTC", "ETH", "HYPE"]         # the drop-SOL basket
START, END = "2026-06-04", "2026-08-22"  # captured-book window, complete days
BORROW_BPS_DAY = 1.0
DEFAULT_NOTIONAL = 10_000.0            # the ledger files are written at this notional only
COST = CostModel()
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def book_slippage(coin: str, day: str, notional: float, first: bool) -> float | None:
    """Median walk-the-book slippage over one day's snapshots, sampled once a minute."""
    files = sorted((data.DATA_ROOT / "ws" / coin / "l2book" / day).glob("*.parquet"))
    if not files:
        return None
    fp = files[0] if first else files[-1]
    df = pd.read_parquet(fp, columns=["best_bid_px", "best_ask_px", "levels_json"]).iloc[::60]
    slips = []
    for bb, ba, raw in zip(df["best_bid_px"], df["best_ask_px"], df["levels_json"]):
        parsed = execution.parse_l2_snapshot(bb, ba, raw)
        if parsed is None:
            continue
        slip = execution.walk_book(parsed[1], notional, parsed[0])
        if slip is not None:
            slips.append(slip)
    return float(np.median(slips)) if slips else None


def coin_ledger(coin: str, notional: float) -> dict | None:
    """Daily net return series and cost breakdown for one coin's round trip."""
    pnl = signals.basis_drift_pnl(data.load_funding(coin), data.load_spot(coin), data.load_marks(coin))
    pnl = pnl[(pnl["ts"] >= pd.Timestamp(START, tz="UTC")) & (pnl["ts"] < pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=1))]
    if pnl.empty:
        return None
    daily = stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))
    funding_total = float(pnl["funding_pnl"].sum())
    drift_total = float(pnl["drift_pnl"].sum())

    slip_in = book_slippage(coin, START, notional, first=True)
    slip_out = book_slippage(coin, END, notional, first=False)
    if slip_in is None or slip_out is None:
        return None
    # both legs pay the taker fee each way; both legs cross a spread, measured on the perp book
    fees = 4 * COST.taker_fee
    slippage = 2 * (slip_in + slip_out)
    borrow_daily = BORROW_BPS_DAY / 1e4

    net = daily - borrow_daily
    net.iloc[0] -= fees / 2 + 2 * slip_in
    net.iloc[-1] -= fees / 2 + 2 * slip_out
    return {"daily_net": net, "funding": funding_total, "drift": drift_total,
            "fees": fees, "slippage": slippage, "borrow": borrow_daily * len(daily),
            "slip_in_bps": slip_in * 1e4, "slip_out_bps": slip_out * 1e4}


def main(notional: float) -> None:
    print("=" * 78)
    print(f"Carry trade ledger: drop-SOL basket, {START} to {END}, ${notional:,.0f} per coin")
    print("  every cost measured (book walk) or fee-schedule; no parametric drift")
    print("=" * 78)
    print(f"\n{'coin':<6} {'days':>5} {'funding':>9} {'drift':>8} {'fees':>7} {'slip':>7} "
          f"{'borrow':>7} {'net':>8} {'annSR':>7}")
    print("-" * 78)

    ledgers = {}
    trade_rows = []
    for coin in COINS:
        led = coin_ledger(coin, notional)
        if led is None:
            print(f"  {coin:<6} (no books)")
            continue
        ledgers[coin] = led
        d = led["daily_net"]
        trade_rows.append((coin, f"{d.index.min():%Y-%m-%d}", f"{d.index.max():%Y-%m-%d}",
                           len(d), float(d.sum())))
        net_total = float(led["daily_net"].sum())
        sr = stats.annualised_sharpe(led["daily_net"])[0]
        print(f"  {coin:<6} {len(led['daily_net']):>5} {led['funding'] * 1e4:>8.1f}b {led['drift'] * 1e4:>7.1f}b "
              f"{led['fees'] * 1e4:>6.1f}b {led['slippage'] * 1e4:>6.2f}b {led['borrow'] * 1e4:>6.1f}b "
              f"{net_total * 1e4:>7.1f}b {sr:>7.2f}")

    if trade_rows:
        print("\ntrade ledger (one round trip per coin):")
        print(f"{'coin':<6} {'entry':>12} {'exit':>12} {'hold_d':>7} {'net_bps':>9} {'net_$':>10}")
        for c, t0, t1, hd, nb in trade_rows:
            print(f"  {c:<6} {t0:>12} {t1:>12} {hd:>7} {nb * 1e4:>+9.1f} {nb * notional:>+10.2f}")
        if notional == DEFAULT_NOTIONAL:
            pd.DataFrame(trade_rows, columns=["coin", "entry", "exit", "hold_d", "net"]).assign(
                net_bps=lambda d: d["net"] * 1e4, notional=notional).to_csv(RESULTS_ROOT / "carry_ledger.csv", index=False)

    if ledgers:
        daily = pd.concat([l["daily_net"] for l in ledgers.values()], axis=1, keys=list(ledgers)).dropna()
        basket = daily.mean(axis=1)
        if notional == DEFAULT_NOTIONAL:
            daily.assign(basket=basket).mul(1e4).rename_axis("day").to_csv(
                RESULTS_ROOT / "carry_ledger_daily.csv", float_format="%.4f")
        sr, ret, _ = stats.annualised_sharpe(basket)
        total = float(basket.sum())
        print("-" * 78)
        print(f"  basket net {total * 1e4:+.1f} bps over {len(basket)} days "
              f"(${total * notional:+,.2f} per ${notional:,.0f}/coin); ann ret {ret * 100:+.2f}%, Sharpe {sr:+.2f}")
        payload = {c: {"net_bps": round(float(l["daily_net"].sum()) * 1e4, 2),
                       "slip_in_bps": round(l["slip_in_bps"], 3), "slip_out_bps": round(l["slip_out_bps"], 3)}
                   for c, l in ledgers.items()}
        payload["basket"] = {"net_bps": round(total * 1e4, 2), "ann_ret": round(ret, 5),
                             "ann_sharpe": round(sr, 3)}
        saved, h = results.record_run(RESULTS_ROOT, "carry_ledger",
                                      {"coins": list(ledgers), "window": f"{START}/{END}",
                                       "notional": notional}, payload, f"{START}/{END}")
        print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--notional", type=float, default=DEFAULT_NOTIONAL)
    main(p.parse_args().notional)
