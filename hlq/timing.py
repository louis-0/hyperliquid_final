"""Next-hour funding-sign model: the funding-timing classifier and its SHAP drivers.

Shared by the ML feasibility backtest and the dashboard so the model definition (feature set,
time-ordered split, and hyper-parameters) lives in one place rather than being restated per caller.
"""
from __future__ import annotations

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

from hlq import baselines, explain, features

TRAIN_FRAC = 0.70
SEED = 42


def train_sign_model(df: pd.DataFrame, train_frac: float = TRAIN_FRAC, seed: int = SEED):
    """Fit the next-hour funding-sign classifier on a time-ordered split.

    Splits chronologically with no shuffle, so the test set is strictly future to the train set,
    then fits an XGBoost classifier on the leakage-safe feature columns. Returns the fitted model
    with the train and test frames.
    """
    split = int(len(df) * train_frac)
    train, test = df.iloc[:split], df.iloc[split:]
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            eval_metric="logloss", random_state=seed, n_jobs=2, tree_method="hist")
    clf.fit(train[features.FEATURES], train["target_fund_sign"])
    return clf, train, test


def sign_drivers(df: pd.DataFrame, k: int = 5) -> dict:
    """Top-k SHAP drivers behind the funding-sign call, with the out-of-sample skill it earns.

    Returns the strongest drivers by mean absolute SHAP over the test window, the test AUC, the
    classifier's test accuracy, the majority-class baseline accuracy, and the test-window length.
    """
    clf, train, test = train_sign_model(df)
    x_test = test[features.FEATURES]
    shap_frame, _ = explain.explain(clf, x_test)
    sign_test = test["target_fund_sign"]
    proba = clf.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(sign_test, proba)) if sign_test.nunique() > 1 else float("nan")
    return {
        "drivers": explain.mean_abs_drivers(shap_frame, k),
        "auc": auc,
        "acc": float(accuracy_score(sign_test, (proba > 0.5).astype(int))),
        "naive_acc": baselines.majority_class_accuracy(train["target_fund_sign"], sign_test),
        "n_test": int(len(test)),
    }
