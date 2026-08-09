"""Transaction-cost model: taker/maker fees, slippage, and per-turnover round-trip cost."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.00045      # 4.5 bps per side
    maker_fee: float = 0.00015      # 1.5 bps per side (negative = rebate)
    slippage: float = 0.0001        # 1 bp per side, taker only (maker does not cross the spread)

    def round_trip(self, maker: bool = False) -> float:
        """Entry + exit cost as a fraction. Taker crosses the spread; a maker posts and does not."""
        if maker:
            return 2 * self.maker_fee
        return 2 * (self.taker_fee + self.slippage)

    def amortised_daily(self, n_days: int, maker: bool = False) -> float:
        """Round-trip cost spread over an n-day always-on hold (entered and exited once)."""
        return self.round_trip(maker) / n_days if n_days else 0.0
