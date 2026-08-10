"""hlq.explain: SHAP attribution over a fitted tree model (Lundberg and Lee 2017)."""
import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")
pytest.importorskip("shap")

from hlq import explain


def _fitted_model(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    y = ((X["a"] + 0.5 * X["b"] + rng.normal(scale=0.1, size=n)) > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3, random_state=seed).fit(X, y)
    return model, X


def test_shap_additivity_reconstructs_the_margin():
    # SHAP's defining guarantee: base value + a row's contributions == the model's raw margin
    model, X = _fitted_model()
    sv, base = explain.explain(model, X)
    recon = base + sv.sum(axis=1).to_numpy()
    margin = model.predict(X, output_margin=True)
    assert recon == pytest.approx(margin, abs=1e-3)


def test_explain_frame_is_aligned_to_features():
    model, X = _fitted_model()
    sv, base = explain.explain(model, X)
    assert list(sv.columns) == list(X.columns)
    assert len(sv) == len(X)
    assert isinstance(base, float)


def test_top_drivers_ranks_by_absolute_contribution():
    # |-0.9| > |0.5| > |-0.2| > |0.1|  ->  the two strongest drivers are b then a
    row = pd.Series({"a": 0.5, "b": -0.9, "c": 0.1, "d": -0.2})
    assert [name for name, _ in explain.top_drivers(row, k=2)] == ["b", "a"]