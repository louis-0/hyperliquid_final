#!/usr/bin/env python3
"""Fused funding-sign classifier: candle features plus the captured tape, built on hlq.

Joins the hourly REST feature matrix (hlq.features) with hourly aggregates from the
WebSocket capture (hlq.microfeatures: book imbalance, spread, taker flow, intra-hour
premium, and, when a labels parquet is given, the frozen cohort's net flow), then scores
the funding-sign classifier on the candle features alone and on the fused set over the
same rows and the same time-ordered split, so the tape's marginal contribution is
measured directly. The joined window is bounded by the capture start.

    python scripts/run_fused_ml.py
    python scripts/run_fused_ml.py /path/to/wallet_labels.parquet
    python scripts/run_fused_ml.py /path/to/wallet_labels.parquet BTC ETH
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import data, features, microfeatures, results, timing

CHASSIS = ["BTC", "ETH", "SOL", "HYPE"]
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def fused_frame(coin: str, smart: set[str] | None) -> tuple[pd.DataFrame, list[str]]:
    """REST feature matrix inner-joined with the hourly tape aggregates for one coin."""
    base = features.build_features(data.load_funding(coin), data.load_spot(coin), data.load_marks(coin))
    micro = microfeatures.hourly(coin, smart=smart)
    extra = microfeatures.MICRO + (microfeatures.COHORT if smart is not None else [])
    df = base.merge(micro, on="hour", how="inner").dropna(subset=extra).reset_index(drop=True)
    return df, extra


def main(labels_path: Path | None, coins: list[str]) -> None:
    smart = None
    if labels_path is not None:
        lab = pd.read_parquet(labels_path, columns=["wallet", "is_smart"])
        smart = set(lab.loc[lab["is_smart"], "wallet"].tolist())
        print(f"cohort: {len(smart):,} is_smart wallets ({labels_path.name})")

    print("=" * 100)
    print("Fused funding-sign classifier: candle features vs candle + tape features, same rows/split")
    print("=" * 100)
    print(f"\n{'coin':<6} {'n_test':>6} {'major':>7} {'candle acc/auc':>15} {'fused acc/auc':>15}"
          f" {'d_auc':>7}  fused top drivers")
    print("-" * 100)

    table: dict[str, dict] = {}
    span = None
    for coin in coins:
        try:
            df, extra = fused_frame(coin, smart)
        except FileNotFoundError:
            print(f"  {coin:<6} (no data)")
            continue
        if len(df) < 200:
            print(f"  {coin:<6} (only {len(df)} joined hours)")
            continue
        fused_cols = features.FEATURES + extra
        candle = timing.compare_classifiers(df)
        fused = timing.compare_classifiers(df, feature_cols=fused_cols)
        drivers = timing.sign_drivers(df, k=3, feature_cols=fused_cols)
        c, f = candle["models"]["xgboost"], fused["models"]["xgboost"]
        names = [n for n, _ in drivers["drivers"]]
        print(f"  {coin:<6} {fused['n_test']:>6} {fused['majority_acc']:>7.3f} "
              f"{c['acc']:>7.3f}/{c['auc']:.3f} {f['acc']:>7.3f}/{f['auc']:.3f} "
              f"{f['auc'] - c['auc']:>+7.3f}  {', '.join(names)}")
        table[coin] = {"n_test": fused["n_test"], "majority_acc": round(fused["majority_acc"], 4),
                       "candle": {k: round(v, 4) for k, v in c.items()},
                       "fused": {k: round(v, 4) for k, v in f.items()},
                       "drivers": names}
        span = f"{df['hour'].min():%Y-%m-%d}/{df['hour'].max():%Y-%m-%d}"

    if table:
        saved, h = results.record_run(RESULTS_ROOT, "fused_ml",
                                      {"coins": list(table), "window": span,
                                       "cohort": labels_path.name if labels_path else None},
                                      table, span or "")
        print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}  window {span}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    labels = Path(argv[0]) if argv and argv[0].endswith(".parquet") else None
    coins = argv[1:] if labels else argv
    main(labels, coins or CHASSIS)
