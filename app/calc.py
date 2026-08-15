"""Cost-floor advisor: net metrics and the deploy verdict for the funding carry."""
from __future__ import annotations

import pandas as pd

from hlq import stats

RETAIL_ANCHOR = 1.8   # He et al. (2024) Bitcoin retail Sharpe
MM_ANCHOR = 3.5       # He et al. (2024) Bitcoin market-maker Sharpe


def verdict(net_sharpe: float) -> str:
    """Deploy verdict from the net Sharpe, calibrated to the He et al. (2024) anchors."""
    if net_sharpe <= 0:
        return "do not deploy"
    if net_sharpe < RETAIL_ANCHOR:
        return "marginal: maker or lower fee tier only"
    return "deployable"


def net_metrics(daily_gross: pd.Series, fee_round_trip: float, borrow_bps_day: float,
                drift_bps_day: float) -> dict:
    """Apply a user's cost stack to a daily gross funding series and return the net picture.

    The round-trip fee is amortised once over the hold; borrow and residual basis drift are
    charged as per-day drags in basis points. Returns gross and net annualised return and
    Sharpe, the verdict, and the He et al. (2024) anchors the verdict is calibrated to.
    """
    n = len(daily_gross)
    daily_drag = (fee_round_trip / n if n else 0.0) + (borrow_bps_day + drift_bps_day) / 1e4
    daily_net = daily_gross - daily_drag
    gross_sharpe, gross_apr, _ = stats.annualised_sharpe(daily_gross)
    net_sharpe, net_apr, _ = stats.annualised_sharpe(daily_net)
    return {
        "gross_apr": gross_apr,
        "net_apr": net_apr,
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "verdict": verdict(net_sharpe),
        "retail_anchor": RETAIL_ANCHOR,
        "mm_anchor": MM_ANCHOR,
    }
