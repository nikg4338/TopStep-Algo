"""
risk/position_sizer.py — Fixed-fractional MES position sizer.

Calculates the number of MES contracts to trade based on:
  - Fixed dollar risk per trade ($20–$40)
  - Stop distance in points
  - MES point value ($5.00 per point)

Always returns at least 1 contract, never 0.
"""

from __future__ import annotations

import logging
import math

from config import (
    MLL_PROXIMITY_BUFFER,
    POINT_VALUE,
    RISK_PER_TRADE,
    RISK_PER_TRADE_MAX,
    RISK_PER_TRADE_MIN,
    TICK_SIZE,
    TICK_VALUE,
)

logger = logging.getLogger(__name__)


class PositionSizer:
    """Compute MES contract quantity for a given stop distance."""

    def __init__(self, risk_per_trade: float = RISK_PER_TRADE) -> None:
        if not (RISK_PER_TRADE_MIN <= risk_per_trade <= RISK_PER_TRADE_MAX):
            raise ValueError(
                f"risk_per_trade must be in [{RISK_PER_TRADE_MIN}, {RISK_PER_TRADE_MAX}], "
                f"got {risk_per_trade}"
            )
        self.risk_per_trade = risk_per_trade

    def calculate(
        self,
        stop_distance_points: float,
        *,
        mll_proximity: bool = False,
    ) -> int:
        """Return number of MES contracts.

        Parameters
        ----------
        stop_distance_points : float
            Distance from entry to stop in index points (e.g., 4.0 = 4 points = 16 ticks).
        mll_proximity : bool
            If True, use minimum risk ($20) regardless of configured risk.

        Returns
        -------
        int  Always >= 1.
        """
        if stop_distance_points <= 0:
            logger.warning("stop_distance_points <= 0 (%.2f) — defaulting to 1 contract", stop_distance_points)
            return 1

        effective_risk = RISK_PER_TRADE_MIN if mll_proximity else self.risk_per_trade

        # Dollar risk per contract = stop_distance_points × POINT_VALUE
        risk_per_contract = stop_distance_points * POINT_VALUE

        if risk_per_contract <= 0:
            return 1

        contracts = max(1, math.floor(effective_risk / risk_per_contract))

        # Safety cap: ensure total dollar exposure never exceeds max risk
        actual_risk = contracts * risk_per_contract
        if actual_risk > RISK_PER_TRADE_MAX:
            contracts = max(1, math.floor(RISK_PER_TRADE_MAX / risk_per_contract))

        logger.debug(
            "Sizer: stop=%.2f pts, risk=$%.0f, contracts=%d (actual $%.0f)",
            stop_distance_points,
            effective_risk,
            contracts,
            contracts * risk_per_contract,
        )
        return contracts

    @staticmethod
    def stop_distance_to_ticks(stop_points: float) -> int:
        """Convert a stop distance in points to MES ticks."""
        return max(1, round(stop_points / TICK_SIZE))

    @staticmethod
    def ticks_to_dollars(ticks: int, contracts: int = 1) -> float:
        """Convert tick count to dollar value."""
        return ticks * TICK_VALUE * contracts
