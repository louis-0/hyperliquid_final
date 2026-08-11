"""Order-book execution: volume-weighted fill price and slippage from an L2 book."""
from __future__ import annotations

import json
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


def parse_l2_snapshot(best_bid, best_ask, levels_json) -> tuple[float, list[dict]] | None:
    """Parse a captured L2 snapshot into (mid, asks), or None if it lacks a usable ask side.

    `levels_json` is Hyperliquid's [bids, asks] structure serialised as a JSON string, each
    level a {'px', 'sz'} mapping; the ask side is the second element. `mid` is the midpoint of
    the best bid and ask.
    """
    try:
        levels = json.loads(levels_json)
    except (TypeError, ValueError):
        return None
    if not (isinstance(levels, list) and len(levels) >= 2 and levels[1]):
        return None
    return (float(best_bid) + float(best_ask)) / 2, levels[1]