#!/usr/bin/env python3
"""ML feasibility: can next-hour funding be predicted beyond naive baselines?

Builds the leakage-safe feature matrix (hlq.features) and, on a time-ordered 70/30 split, compares
three classifier families for the funding-sign call (XGBoost, a random forest, and a logistic
regression) against the majority-class baseline, trains a magnitude regressor against the last-value
baseline, and reports the SHAP drivers behind the XGBoost call (hlq.explain).

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

    cmp = timing.compare_classifiers(df)                   # xgboost / random_forest / logistic vs majority
    sign = timing.sign_drivers(df, k=3)                    # SHAP drivers behind the xgboost call
    stab = timing.driver_stability(df)                     # top driver across five refit seeds
    naive_rmse = baselines.persistence_rmse(mag_test, test["funding_lag_1"].to_numpy())

    reg = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                           random_state=timing.SEED, n_jobs=2, tree_method="hist")
    reg.fit(train[features.FEATURES], train["target_fund"])
    xgb_rmse = float(np.sqrt(mean_squared_error(mag_test, reg.predict(test[features.FEATURES]))))

    return {"coin": coin, "ok": True, "n_test": cmp["n_test"], "majority_acc": cmp["majority_acc"],
            "models": cmp["models"], "naive_rmse": naive_rmse, "xgb_rmse": xgb_rmse,
            "drivers": [name for name, _ in sign["drivers"]],
            "lead_driver": stab["lead_driver"], "lead_share": stab["lead_share"],
            "window": f"{df['hour'].min():%Y-%m-%d}/{df['hour'].max():%Y-%m-%d}"}


def main(coins: list[str]) -> None:
    print("=" * 112)
    print("ML feasibility: next-hour funding sign, classifier families vs the majority-class baseline")
    print("  split: 70/30 time-ordered; acc and AUC out-of-sample across xgboost, random forest, logistic")
    print("  stab: share of five refit seeds with the same top SHAP driver; rmse: xgboost / last-value, magnitude")
    print("=" * 112)
    print(f"\n{'coin':<6} {'n_test':>7} {'major':>7} {'xgb_acc':>8} {'xgb_auc':>8}"
          f" {'rf_acc':>7} {'rf_auc':>7} {'lr_acc':>7} {'lr_auc':>7} {'stab':>5} {'rmse':>5}  top SHAP drivers")
    print("-" * 112)
    table: dict[str, dict] = {}
    span = None
    for coin in coins:
        try:
            r = feasibility(coin)
        except FileNotFoundError:
            print(f"  {coin:<6} (missing spot/perp/funding)")
            continue
        if not r["ok"]:
            print(f"  {coin:<6} (insufficient data)")
            continue
        m = r["models"]
        table[coin] = {"majority_acc": round(r["majority_acc"], 4),
                       "xgboost": {k: round(v, 4) for k, v in m["xgboost"].items()},
                       "random_forest": {k: round(v, 4) for k, v in m["random_forest"].items()},
                       "logistic": {k: round(v, 4) for k, v in m["logistic"].items()},
                       "top_drivers": r["drivers"],
                       "lead_driver": r["lead_driver"], "lead_share": r["lead_share"],
                       "naive_rmse": r["naive_rmse"], "xgb_rmse": r["xgb_rmse"]}
        span = span or r["window"]
        print(f"  {r['coin']:<6} {r['n_test']:>7} {r['majority_acc']:>7.3f}"
              f" {m['xgboost']['acc']:>8.3f} {m['xgboost']['auc']:>8.3f}"
              f" {m['random_forest']['acc']:>7.3f} {m['random_forest']['auc']:>7.3f}"
              f" {m['logistic']['acc']:>7.3f} {m['logistic']['auc']:>7.3f}"
              f" {r['lead_share']:>5.2f} {r['xgb_rmse'] / r['naive_rmse']:>5.2f}  {', '.join(r['drivers'])}")

    if table:
        saved, h = results.record_run(RESULTS_ROOT, "ml_feasibility",
                                      {"coins": list(table), "train_frac": timing.TRAIN_FRAC,
                                       "seed": timing.SEED, "window": span,
                                       "models": ["xgboost", "random_forest", "logistic"]},
                                      {"per_coin": table}, span or "")
        print(f"\n[results] {'recorded' if saved else 'already recorded'} {h}  window {span}")

    print("\nAUC is inflated by class imbalance, and the three families land within a point or two of"
          "\neach other, with logistic regression competitive with or ahead of XGBoost. With the SHAP"
          "\ndrivers loading on the funding lags, the skill is the persistence any family reads, not an"
          "\nedge that gradient boosting uniquely captures.")


if __name__ == "__main__":
    main(sys.argv[1:] or CHASSIS)