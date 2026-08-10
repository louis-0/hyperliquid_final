"""Order-book execution: volume-weighted fill price and slippage from an L2 book."""
from __future__ import annotations

from collections.abc import Sequence


def walk_book(asks: Sequence[dict], target_notional: float, mid: float) -> float | None:
    """Slippage from filling `target_notional` USD against the ask side of an L2 book.

    Walks the levels in order (cheapest first), taking USD from each until the order is
    filled, and returns (vwap - mid) / mid. The VWAP is a true cost-per-unit (total USD paid
    divided by units received), so it is not biased by weighting price against notional.
    Returns None if the book lacks the depth to fill the order.
    """
    filled_usd = 0.0
    units = 0.0
    for level in asks:
        px = float(level["px"])
        take_usd = min(px * float(level["sz"]), target_notional - filled_usd)
        units += take_usd / px
        filled_usd += take_usd
        if filled_usd >= target_notional:
            return (target_notional / units - mid) / mid
    return None