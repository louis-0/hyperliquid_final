"""hlq.timing: the funding-sign model and its SHAP drivers, on a synthetic learnable frame."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("shap")

from hlq import features, timing


def _learnable_frame(n=400, seed=0):
    """A feature matrix shaped like build_features output, but where the next-hour funding sign
    is driven by the `basis` column alone. A working model must lean on basis and beat chance."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in features.FEATURES})
    signal = df["basis"] + rng.normal(scale=0.3, size=n)
    df["target_fund"] = signal
    df["target_fund_sign"] = (signal > 0).astype(int)
    return df


def test_train_sign_model_split_is_time_ordered_no_shuffle():
    clf, train, test = timing.train_sign_model(_learnable_frame(n=300), train_frac=0.7)
    assert len(train) == 210 and len(test) == 90          # chronological 70/30, no shuffle
    assert train.index.max() < test.index.min()           # the test window strictly follows train


def test_sign_drivers_finds_the_driving_feature_and_beats_chance():
    r = timing.sign_drivers(_learnable_frame(), k=3)
    assert len(r["drivers"]) == 3
    assert r["drivers"][0][0] == "basis"                  # basis drives the label, so it ranks first
    assert r["auc"] > 0.6                                  # real out-of-sample skill, not chance
    assert r["n_test"] == 120                             # 30% of 400 rows


def test_compare_classifiers_all_beat_the_majority_baseline():
    r = timing.compare_classifiers(_learnable_frame())
    assert set(r["models"]) == {"xgboost", "random_forest", "logistic"}
    for name, s in r["models"].items():
        assert s["auc"] > 0.6, f"{name} did not learn the driving feature"   # basis drives the label
    assert r["n_test"] == 120
    assert 0.0 <= r["majority_acc"] <= 1.0
