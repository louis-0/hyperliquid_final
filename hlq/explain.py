"""SHAP attribution over a fitted tree model (Lundberg and Lee 2017)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def explain(model, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per-prediction SHAP values and base value for a fitted tree model.

    Uses shap.TreeExplainer (Lundberg and Lee 2017). By SHAP's additivity property, for each
    row the base value plus that row's SHAP values sums to the model's raw margin prediction,
    so the attribution is exact rather than a heuristic importance. Returns the SHAP values as
    a DataFrame aligned to the feature columns, and the scalar base value.
    """
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X))
    base = float(np.ravel(explainer.expected_value)[0])
    return pd.DataFrame(values, columns=list(X.columns), index=X.index), base


def top_drivers(shap_row: pd.Series, k: int = 3) -> list[tuple[str, float]]:
    """The k features with the largest absolute SHAP contribution for one prediction,
    as (feature, value) pairs ordered strongest first."""
    ranked = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)
    return list(ranked.head(k).items())