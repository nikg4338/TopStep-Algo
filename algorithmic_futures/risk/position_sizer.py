"""
risk/position_sizer.py — Fixed-fractional MES position sizer.

Calculates the number of MES contracts to trade based on:
  - Fixed dollar risk per trade ($20–$40)
  - Stop distance in points
  - MES point value ($5.00 per point)

Rejects trades when one minimum contract would exceed available risk.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from config import (
    MIN_CONTRACT_RISK_SAFETY_FRACTION,
    POINT_VALUE,
    RISK_PER_TRADE,
    RISK_PER_TRADE_MAX,
    RISK_PER_TRADE_MIN,
    TICK_SIZE,
    TICK_VALUE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionSizeResult:
    """Structured position sizing decision."""

    allowed: bool
    quantity: int
    rejection_reason: str = ""
    stop_distance_points: float = 0.0
    risk_per_contract: float = 0.0
    max_allowed_trade_risk: float = 0.0
    remaining_daily_loss_budget: float | None = None
    remaining_mll_headroom: float | None = None
    safety_fraction: float = MIN_CONTRACT_RISK_SAFETY_FRACTION
    mll_headroom_safety_fraction: float = MIN_CONTRACT_RISK_SAFETY_FRACTION
    projected_trade_risk: float = 0.0


class PositionSizer:
    """Compute MES contract quantity for a given stop distance."""

    def __init__(
        self,
        risk_per_trade: float = RISK_PER_TRADE,
        *,
        dollars_per_point: float = POINT_VALUE,
        safety_fraction: float = MIN_CONTRACT_RISK_SAFETY_FRACTION,
    ) -> None:
        if not (RISK_PER_TRADE_MIN <= risk_per_trade <= RISK_PER_TRADE_MAX):
            raise ValueError(
                f"risk_per_trade must be in [{RISK_PER_TRADE_MIN}, {RISK_PER_TRADE_MAX}], "
                f"got {risk_per_trade}"
            )
        if dollars_per_point <= 0:
            raise ValueError(f"dollars_per_point must be > 0, got {dollars_per_point}")
        if not (0 < safety_fraction <= 1):
            raise ValueError(f"safety_fraction must be in (0, 1], got {safety_fraction}")
        self.risk_per_trade = risk_per_trade
        self.dollars_per_point = dollars_per_point
        self.safety_fraction = safety_fraction

    def calculate(
        self,
        stop_distance_points: float,
        *,
        mll_proximity: bool = False,
    ) -> int:
        """Return number of MES contracts, or 0 when risk cannot be honored.

        Parameters
        ----------
        stop_distance_points : float
            Distance from entry to stop in index points (e.g., 4.0 = 4 points = 16 ticks).
        mll_proximity : bool
            If True, use minimum risk ($20) regardless of configured risk.

        Returns
        -------
        int  Contract quantity. 0 means rejected by the minimum-contract risk gate.
        """
        return self.calculate_with_risk_gate(
            stop_distance_points=stop_distance_points,
            mll_proximity=mll_proximity,
        ).quantity

    def calculate_with_risk_gate(
        self,
        stop_distance_points: float,
        *,
        mll_proximity: bool = False,
        remaining_daily_loss_budget: float | None = None,
        remaining_mll_headroom: float | None = None,
        safety_fraction: float | None = None,
        mll_headroom_safety_fraction: float | None = None,
        max_allowed_trade_risk: float | None = None,
    ) -> PositionSizeResult:
        """Return a structured sizing decision with explicit rejection reasons."""
        active_safety_fraction = self.safety_fraction if safety_fraction is None else safety_fraction
        if not (0 < active_safety_fraction <= 1):
            raise ValueError(f"safety_fraction must be in (0, 1], got {active_safety_fraction}")
        active_mll_fraction = (
            active_safety_fraction
            if mll_headroom_safety_fraction is None
            else mll_headroom_safety_fraction
        )
        if not (0 < active_mll_fraction <= 1):
            raise ValueError(
                f"mll_headroom_safety_fraction must be in (0, 1], got {active_mll_fraction}"
            )

        effective_risk = RISK_PER_TRADE_MIN if mll_proximity else self.risk_per_trade
        risk_budget = effective_risk if max_allowed_trade_risk is None else max_allowed_trade_risk

        if stop_distance_points <= 0:
            reason = f"INVALID_STOP_DISTANCE: stop_distance_points {stop_distance_points:.2f} must be > 0"
            logger.warning("Sizer rejected trade: %s", reason)
            return PositionSizeResult(
                allowed=False,
                quantity=0,
                rejection_reason=reason,
                stop_distance_points=stop_distance_points,
                max_allowed_trade_risk=risk_budget,
                remaining_daily_loss_budget=remaining_daily_loss_budget,
                remaining_mll_headroom=remaining_mll_headroom,
                safety_fraction=active_safety_fraction,
                mll_headroom_safety_fraction=active_mll_fraction,
            )

        # Dollar risk per minimum contract = stop_distance_points * dollars_per_point
        risk_per_contract = stop_distance_points * self.dollars_per_point

        if risk_per_contract > risk_budget:
            reason = (
                "MIN_CONTRACT_RISK_EXCEEDS_TRADE_RISK: "
                f"one contract risks ${risk_per_contract:.2f} > allowed ${risk_budget:.2f}"
            )
            logger.warning("Sizer rejected trade: %s", reason)
            return PositionSizeResult(
                allowed=False,
                quantity=0,
                rejection_reason=reason,
                stop_distance_points=stop_distance_points,
                risk_per_contract=risk_per_contract,
                max_allowed_trade_risk=risk_budget,
                remaining_daily_loss_budget=remaining_daily_loss_budget,
                remaining_mll_headroom=remaining_mll_headroom,
                safety_fraction=active_safety_fraction,
                mll_headroom_safety_fraction=active_mll_fraction,
            )

        if remaining_daily_loss_budget is not None:
            daily_budget_cap = max(0.0, remaining_daily_loss_budget) * active_safety_fraction
            if risk_per_contract > daily_budget_cap:
                reason = (
                    "MIN_CONTRACT_RISK_EXCEEDS_DAILY_LOSS_BUDGET: "
                    f"one contract risks ${risk_per_contract:.2f} > daily budget cap ${daily_budget_cap:.2f}"
                )
                logger.warning("Sizer rejected trade: %s", reason)
                return PositionSizeResult(
                    allowed=False,
                    quantity=0,
                    rejection_reason=reason,
                    stop_distance_points=stop_distance_points,
                    risk_per_contract=risk_per_contract,
                    max_allowed_trade_risk=risk_budget,
                    remaining_daily_loss_budget=remaining_daily_loss_budget,
                    remaining_mll_headroom=remaining_mll_headroom,
                    safety_fraction=active_safety_fraction,
                    mll_headroom_safety_fraction=active_mll_fraction,
                )

        if remaining_mll_headroom is not None:
            mll_headroom_cap = max(0.0, remaining_mll_headroom) * active_mll_fraction
            if risk_per_contract > mll_headroom_cap:
                reason = (
                    "MIN_CONTRACT_RISK_EXCEEDS_MLL_HEADROOM: "
                    f"one contract risks ${risk_per_contract:.2f} > MLL headroom cap ${mll_headroom_cap:.2f}"
                )
                logger.warning("Sizer rejected trade: %s", reason)
                return PositionSizeResult(
                    allowed=False,
                    quantity=0,
                    rejection_reason=reason,
                    stop_distance_points=stop_distance_points,
                    risk_per_contract=risk_per_contract,
                    max_allowed_trade_risk=risk_budget,
                    remaining_daily_loss_budget=remaining_daily_loss_budget,
                    remaining_mll_headroom=remaining_mll_headroom,
                    safety_fraction=active_safety_fraction,
                    mll_headroom_safety_fraction=active_mll_fraction,
                )

        contracts = math.floor(risk_budget / risk_per_contract)
        if contracts < 1:
            reason = (
                "MIN_CONTRACT_RISK_EXCEEDS_TRADE_RISK: "
                f"one contract risks ${risk_per_contract:.2f} > allowed ${risk_budget:.2f}"
            )
            logger.warning("Sizer rejected trade: %s", reason)
            return PositionSizeResult(
                allowed=False,
                quantity=0,
                rejection_reason=reason,
                stop_distance_points=stop_distance_points,
                risk_per_contract=risk_per_contract,
                max_allowed_trade_risk=risk_budget,
                remaining_daily_loss_budget=remaining_daily_loss_budget,
                remaining_mll_headroom=remaining_mll_headroom,
                safety_fraction=active_safety_fraction,
                mll_headroom_safety_fraction=active_mll_fraction,
            )

        projected_trade_risk = contracts * risk_per_contract

        if remaining_daily_loss_budget is not None:
            daily_budget_cap = max(0.0, remaining_daily_loss_budget) * active_safety_fraction
            if projected_trade_risk > daily_budget_cap:
                reason = (
                    "PROJECTED_TRADE_RISK_EXCEEDS_DAILY_LOSS_BUDGET: "
                    f"sized trade risks ${projected_trade_risk:.2f} > daily budget cap ${daily_budget_cap:.2f}"
                )
                logger.warning("Sizer rejected trade: %s", reason)
                return PositionSizeResult(
                    allowed=False,
                    quantity=0,
                    rejection_reason=reason,
                    stop_distance_points=stop_distance_points,
                    risk_per_contract=risk_per_contract,
                    max_allowed_trade_risk=risk_budget,
                    remaining_daily_loss_budget=remaining_daily_loss_budget,
                    remaining_mll_headroom=remaining_mll_headroom,
                    safety_fraction=active_safety_fraction,
                    mll_headroom_safety_fraction=active_mll_fraction,
                    projected_trade_risk=projected_trade_risk,
                )

        if remaining_mll_headroom is not None:
            mll_headroom_cap = max(0.0, remaining_mll_headroom) * active_mll_fraction
            if projected_trade_risk > mll_headroom_cap:
                reason = (
                    "PROJECTED_TRADE_RISK_EXCEEDS_MLL_HEADROOM: "
                    f"sized trade risks ${projected_trade_risk:.2f} > MLL headroom cap ${mll_headroom_cap:.2f}"
                )
                logger.warning("Sizer rejected trade: %s", reason)
                return PositionSizeResult(
                    allowed=False,
                    quantity=0,
                    rejection_reason=reason,
                    stop_distance_points=stop_distance_points,
                    risk_per_contract=risk_per_contract,
                    max_allowed_trade_risk=risk_budget,
                    remaining_daily_loss_budget=remaining_daily_loss_budget,
                    remaining_mll_headroom=remaining_mll_headroom,
                    safety_fraction=active_safety_fraction,
                    mll_headroom_safety_fraction=active_mll_fraction,
                    projected_trade_risk=projected_trade_risk,
                )

        logger.debug(
            "Sizer: stop=%.2f pts, risk=$%.0f, contracts=%d (actual $%.0f)",
            stop_distance_points,
            risk_budget,
            contracts,
            projected_trade_risk,
        )
        return PositionSizeResult(
            allowed=True,
            quantity=contracts,
            stop_distance_points=stop_distance_points,
            risk_per_contract=risk_per_contract,
            max_allowed_trade_risk=risk_budget,
            remaining_daily_loss_budget=remaining_daily_loss_budget,
            remaining_mll_headroom=remaining_mll_headroom,
            safety_fraction=active_safety_fraction,
            mll_headroom_safety_fraction=active_mll_fraction,
            projected_trade_risk=projected_trade_risk,
        )

    @staticmethod
    def stop_distance_to_ticks(stop_points: float) -> int:
        """Convert a stop distance in points to MES ticks."""
        return max(1, round(stop_points / TICK_SIZE))

    @staticmethod
    def ticks_to_dollars(ticks: int, contracts: int = 1) -> float:
        """Convert tick count to dollar value."""
        return ticks * TICK_VALUE * contracts
