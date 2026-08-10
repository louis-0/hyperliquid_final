"""Portfolio construction: equal-weight baskets over the coins' common window."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


def equal_weight_basket(series: Mapping[str, pd.Series], exclude: Iterable[str] = ()) -> pd.Series:
    """Equal-weight basket of per-coin return series over their common (intersection) window.

    The coins are aligned on their shared dates and averaged with equal weight. The
    intersection is deliberate, not the union: heterogeneous per-coin histories (e.g. a coin
    whose funding series is longer than the rest) would otherwise let that single coin
    dominate the days on which it is the only one present, distorting the basket. `exclude`
    builds the named sub-baskets (drop-SOL, drop-NEAR) without the caller pre-filtering.
    """
    excluded = set(exclude)
    cols = {name: s for name, s in series.items() if name not in excluded}
    if not cols:
        return pd.Series(dtype=float)
    aligned = pd.concat(cols.values(), axis=1, keys=cols.keys()).dropna()
    return aligned.mean(axis=1)
