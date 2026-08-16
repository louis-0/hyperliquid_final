#!/usr/bin/env python3
"""ML feasibility: can next-hour funding be predicted beyond naive baselines?

Builds the leakage-safe feature matrix (hlq.features), trains an XGBoost sign classifier and a
magnitude regressor on a time-ordered 70/30 split, compares them to naive baselines (majority
class for direction, last-hour funding for magnitude), and reports the SHAP drivers behind the
sign call (hlq.explain). A pass needs sign AUC above 0.55 and a regression RMSE below naive.

    python scripts/run_ml_feasibility.py
    python scripts/run_ml_feasibility.py BTC ETH
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import baselines, data, features, results, timing

CHASSIS = ["BTC", "ETH", "SOL", "HYPE"]
RESULTS_ROOT = data.DATA_ROOT.parent / "results"


def feasibility(coin: str) -> dict:
    df = features.build_features(data.load_funding(coin), data.load_spot(coin), data.load_marks(coin))
    if len(df) < 200:
        return {"coin": coin, "ok": False}
    split = int(len(df) * timing.TRAIN_FRAC)
    train, test = df.iloc[:split], df.iloc[split:]
    mag_test = test["target_fund"].to_numpy()

    sign = timing.sign_drivers(df, k=3)                    # classifier, SHAP drivers, AUC vs naive
    naive_rmse = baselines.persistence_rmse(mag_test, test["funding_lag_1"].to_numpy())

    reg = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                           random_state=timing.SEED, n_jobs=2, tree_method="hist")
    reg.fit(train[features.FEATURES], train["target_fund"])
    xgb_rmse = float(np.sqrt(mean_squared_error(mag_test, reg.predict(test[features.FEATURES]))))

    return {"coin": coin, "ok": True, "n_test": sign["n_test"], "naive_acc": sign["naive_acc"],
            "acc": sign["acc"], "auc": sign["auc"], "naive_rmse": naive_rmse, "xgb_rmse": xgb_rmse,
            "drivers": [name for name, _ in sign["drivers"]]}


def main(coins: list[str]) -> None:
    print("=" * 84)
    print("ML feasibility: next-hour funding, XGBoost vs naive baselines")
    print("  split: 70/30 time-ordered; pass: sign AUC > 0.55 and RMSE below naive")
    print("=" * 84)
    print(f"\n{'coin':<6} {'n_test':>7} {'naive_acc':>10} {'xgb_acc':>9} {'AUC':>7}"
          f" {'naive_RMSE':>12} {'xgb_RMSE':>12}  top SHAP drivers")
    print("-" * 84)
    table: dict[str, dict] = {}
    for coin in coins:
        try:
            r = feasibility(coin)
        except FileNotFoundError:
            print(f"  {coin:<6} (missing spot/perp/funding)")
            continue
        if not r["ok"]:
            print(f"  {coin:<6} (insufficient data)")
            continue
        table[coin] = {"auc": round(r["auc"], 4), "xgb_acc": round(r["acc"], 4),
                       "naive_acc": round(r["naive_acc"], 4), "top_drivers": r["drivers"]}
        print(f"  {r['coin']:<6} {r['n_test']:>7} {r['naive_acc']:>10.3f} {r['acc']:>9.3f}"
              f" {r['auc']:>7.3f} {r['naive_rmse']:>12.2e} {r['xgb_rmse']:>12.2e}  {', '.join(r['drivers'])}")

    if table:
        saved, h = results.record_run(RESULTS_ROOT, "ml_feasibility",
                                      {"coins": list(table), "train_frac": timing.TRAIN_FRAC,
                                       "seed": timing.SEED}, {"per_coin": table})
        print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}")

    print("\nAUC is inflated by class imbalance; the SHAP drivers show how much of the call"
          "\nrests on the funding lags, i.e. the persistence a naive baseline already captures.")


if __name__ == "__main__":
    main(sys.argv[1:] or CHASSIS)