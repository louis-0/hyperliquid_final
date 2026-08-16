"""Next-hour funding-sign model: the funding-timing classifier and its SHAP drivers.

Shared by the ML feasibility backtest and the dashboard so the model definition (feature set,
time-ordered split, and hyper-parameters) lives in one place rather than being restated per caller.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hlq import baselines, explain, features

TRAIN_FRAC = 0.70
SEED = 42


def _xgb_classifier(seed: int = SEED):
    return xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                             eval_metric="logloss", random_state=seed, n_jobs=2, tree_method="hist")


def train_sign_model(df: pd.DataFrame, train_frac: float = TRAIN_FRAC, seed: int = SEED):
    """Fit the next-hour funding-sign classifier on a time-ordered split.

    Splits chronologically with no shuffle, so the test set is strictly future to the train set,
    then fits an XGBoost classifier on the leakage-safe feature columns. Returns the fitted model
    with the train and test frames.
    """
    split = int(len(df) * train_frac)
    train, test = df.iloc[:split], df.iloc[split:]
    clf = _xgb_classifier(seed)
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


def _score(model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan")
    return {"auc": auc, "acc": float(accuracy_score(y_test, (proba > 0.5).astype(int)))}


def compare_classifiers(df: pd.DataFrame, train_frac: float = TRAIN_FRAC, seed: int = SEED) -> dict:
    """Out-of-sample skill of the funding-sign call across model families on one time-ordered
    split: XGBoost, a random forest, and a logistic regression, each against the majority-class
    baseline. This measures the choice of XGBoost against classical alternatives rather than
    asserting it, and shows whether any family reads more than the persistence the baseline holds.
    """
    split = int(len(df) * train_frac)
    train, test = df.iloc[:split], df.iloc[split:]
    x_train, x_test = train[features.FEATURES], test[features.FEATURES]
    y_train, y_test = train["target_fund_sign"], test["target_fund_sign"]
    models = {
        "xgboost": _xgb_classifier(seed),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                                 random_state=seed, n_jobs=2),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
    }
    scores = {name: _score(m, x_train, y_train, x_test, y_test) for name, m in models.items()}
    return {
        "models": scores,
        "majority_acc": baselines.majority_class_accuracy(y_train, y_test),
        "n_test": int(len(test)),
    }


def driver_stability(df: pd.DataFrame, seeds=(0, 1, 2, 3, 4), k: int = 3) -> dict:
    """Stability of the SHAP driver ranking under model randomness: refit the sign model with
    several seeds, take the top driver each time, and report the most common leader with the share
    of refits it leads. A stable leader means the interpretation is robust, not a one-seed artefact.
    """
    tops = []
    for s in seeds:
        clf, _, test = train_sign_model(df, seed=s)
        shap_frame, _ = explain.explain(clf, test[features.FEATURES])
        tops.append(explain.mean_abs_drivers(shap_frame, k)[0][0])
    lead, count = Counter(tops).most_common(1)[0]
    return {"top_per_seed": tops, "lead_driver": lead, "lead_share": count / len(seeds)}
