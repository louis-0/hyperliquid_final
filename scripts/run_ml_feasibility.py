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
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import hlq when run as a script

from hlq import baselines, data, explain, features

CHASSIS = ["BTC", "ETH", "SOL", "HYPE"]
TRAIN_FRAC = 0.70
SEED = 42


def feasibility(coin: str) -> dict:
    df = features.build_features(data.load_funding(coin), data.load_spot(coin), data.load_marks(coin))
    if len(df) < 200:
        return {"coin": coin, "ok": False}
    split = int(len(df) * TRAIN_FRAC)
    train, test = df.iloc[:split], df.iloc[split:]
    x_train, x_test = train[features.FEATURES], test[features.FEATURES]
    sign_train, sign_test = train["target_fund_sign"], test["target_fund_sign"]
    mag_test = test["target_fund"].to_numpy()

    naive_acc = baselines.majority_class_accuracy(sign_train, sign_test)
    naive_rmse = baselines.persistence_rmse(mag_test, test["funding_lag_1"].to_numpy())

    clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            eval_metric="logloss", random_state=SEED, n_jobs=2, tree_method="hist")
    clf.fit(x_train, sign_train)
    proba = clf.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(sign_test, proba)) if sign_test.nunique() > 1 else float("nan")
    acc = float(accuracy_score(sign_test, (proba > 0.5).astype(int)))

    reg = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                           random_state=SEED, n_jobs=2, tree_method="hist")
    reg.fit(x_train, train["target_fund"])
    xgb_rmse = float(np.sqrt(mean_squared_error(mag_test, reg.predict(x_test))))

    shap_values, _ = explain.explain(clf, x_test)
    drivers = shap_values.abs().mean().sort_values(ascending=False).head(3)

    return {"coin": coin, "ok": True, "n_test": len(test), "naive_acc": naive_acc, "acc": acc,
            "auc": auc, "naive_rmse": naive_rmse, "xgb_rmse": xgb_rmse,
            "drivers": list(drivers.index)}


def main(coins: list[str]) -> None:
    print("=" * 84)
    print("ML feasibility: next-hour funding, XGBoost vs naive baselines")
    print("  split: 70/30 time-ordered; pass: sign AUC > 0.55 and RMSE below naive")
    print("=" * 84)
    print(f"\n{'coin':<6} {'n_test':>7} {'naive_acc':>10} {'xgb_acc':>9} {'AUC':>7}"
          f" {'naive_RMSE':>12} {'xgb_RMSE':>12}  top SHAP drivers")
    print("-" * 84)
    for coin in coins:
        try:
            r = feasibility(coin)
        except FileNotFoundError:
            print(f"  {coin:<6} (missing spot/perp/funding)")
            continue
        if not r["ok"]:
            print(f"  {coin:<6} (insufficient data)")
            continue
        print(f"  {r['coin']:<6} {r['n_test']:>7} {r['naive_acc']:>10.3f} {r['acc']:>9.3f}"
              f" {r['auc']:>7.3f} {r['naive_rmse']:>12.2e} {r['xgb_rmse']:>12.2e}  {', '.join(r['drivers'])}")

    print("\nAUC is inflated by class imbalance; the SHAP drivers show how much of the call"
          "\nrests on the funding lags, i.e. the persistence a naive baseline already captures.")


if __name__ == "__main__":
    main(sys.argv[1:] or CHASSIS)