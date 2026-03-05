"""
risk/risk_governor.py — Risk Governor & Consistency Cap for Algorithmic Futures.

Provides two composable components:

1. **ConsistencyCapEngine** — enforces per-mode daily-profit caps so that no
   single day accounts for more than X % of cumulative realised P&L.
2. **RiskGovernor** — pre-trade gate that checks daily loss, profit, trade
   count, session timing, no-trade windows, and (optionally) the consistency
   cap before approving a new entry.

All thresholds are imported from ``config.py``; nothing is hardcoded here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import (
    ACCOUNT_MODE,
    CC_SOFT_CAP_AMOUNT,
    CC_SOFT_CAP_ENABLED,
    CONSISTENCY_CAP,
    DAILY_LOSS_LIMIT_INTERNAL,
    DAILY_PROFIT_HALT,
    PROFIT_TARGET,
    RG_DAILY_LOSS_HALT,
    RG_FLATTEN_CUTOFF_TIME,
    RG_MAX_TRADES_PER_DAY,
    RG_NO_TRADE_WINDOWS,
    RG_STRATEGY_DAILY_PROFIT_CAP,
)

logger = logging.getLogger(__name__)

# ── Rejection / approval reason constants ───────────────────────────────
APPROVE: str = "APPROVE"
REJECT_DAILY_LOSS_HALT: str = "REJECT_DAILY_LOSS_HALT"
REJECT_DAILY_PROFIT_HALT: str = "REJECT_DAILY_PROFIT_HALT"
REJECT_CONSISTENCY_CAP: str = "REJECT_CONSISTENCY_CAP"
REJECT_MAX_TRADES: str = "REJECT_MAX_TRADES"
REJECT_NO_TRADE_WINDOW: str = "REJECT_NO_TRADE_WINDOW"
REJECT_SESSION_CUTOFF: str = "REJECT_SESSION_CUTOFF"


# ── Result dataclasses ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ConsistencyResult:
    """Outcome of a consistency-cap evaluation."""

    allowed_profit_remaining: float
    """Dollars of additional profit allowed today before the cap binds."""

    effective_daily_cap: float
    """Absolute dollar cap for the best single day in this mode."""

    best_day_pct: float
    """Current best-day P&L as a percentage of total realised P&L (0-100)."""

    cap_pct: float
    """Mode-specific cap percentage (e.g. 50.0 for *combine*)."""

    mode: str
    """Account mode used for this evaluation."""

    capped: bool
    """True when today's remaining room is constrained by the cap."""

    reason: str
    """Human-readable explanation of the result."""


@dataclass(frozen=True)
class GovernorResult:
    """Outcome of a risk-governor pre-trade check."""

    approved: bool
    """True when the proposed trade is permitted."""

    reasons: list[str] = field(default_factory=list)
    """List of reason constants explaining the decision."""


# ── Consistency Cap Engine ──────────────────────────────────────────────

class ConsistencyCapEngine:
    """Enforce per-mode consistency caps.

    Rules by mode
    -------------
    * **combine** — best day ≤ 50 % of *profit_target* ($1 500 at default).
    * **express_funded** — best day ≤ 40 % of *profit_target*.
    * **xfa_standard** — no strict cap; optional soft cap via
      ``CC_SOFT_CAP_ENABLED`` / ``CC_SOFT_CAP_AMOUNT``.
    """

    def __init__(self, mode: str = ACCOUNT_MODE) -> None:
        self.mode = mode
        self._cap_pct: float | None = CONSISTENCY_CAP.get(mode)

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        total_realized_pnl: float,
        best_day_pnl: float,
        today_realized_pnl: float,
        profit_target: float = PROFIT_TARGET,
    ) -> ConsistencyResult:
        """Return a :class:`ConsistencyResult` describing how much room
        the trader still has today under the consistency cap.

        Parameters
        ----------
        total_realized_pnl:
            Cumulative realised P&L across all sessions (*including* today).
        best_day_pnl:
            The highest single-day realised P&L recorded so far (may be
            today if today is already the best day).
        today_realized_pnl:
            Realised P&L for the current session so far.
        profit_target:
            The challenge / account profit target.
        """

        # --- xfa_standard: no strict percentage cap -------------------- #
        if self._cap_pct is None:
            return self._evaluate_xfa(
                total_realized_pnl, best_day_pnl, today_realized_pnl
            )

        cap_pct = self._cap_pct
        effective_cap = cap_pct * profit_target

        # --- Edge cases ------------------------------------------------ #
        # First day or no profit yet: allow trading up to the effective cap.
        if total_realized_pnl <= 0:
            return ConsistencyResult(
                allowed_profit_remaining=max(effective_cap - today_realized_pnl, 0.0),
                effective_daily_cap=effective_cap,
                best_day_pct=0.0,
                cap_pct=cap_pct * 100,
                mode=self.mode,
                capped=today_realized_pnl >= effective_cap,
                reason=(
                    "Total P&L is zero or negative; cap based on profit target."
                ),
            )

        # --- Normal operation ------------------------------------------ #
        # The best day must not exceed cap_pct of total realised P&L.
        # Calculate how much more today can earn without breaching that.
        #
        # Two constraints apply simultaneously:
        #   A) best_day_pnl must stay ≤ cap_pct × total_realized_pnl
        #   B) today_realized_pnl must stay ≤ effective_cap (hard ceiling)
        #
        # If today is currently the best day we solve for the marginal
        # dollar that would push the ratio over cap_pct.

        # Determine if today is (or would become) the best day.
        other_best = best_day_pnl if best_day_pnl > today_realized_pnl else 0.0
        effective_best = max(best_day_pnl, today_realized_pnl)

        best_day_pct = (
            (effective_best / total_realized_pnl) * 100
            if total_realized_pnl > 0
            else 0.0
        )

        # Room under the percentage rule:
        #   today + Δ ≤ cap_pct × (total + Δ)
        #   Δ × (1 − cap_pct) ≤ cap_pct × total − today
        #   Δ ≤ (cap_pct × total − today) / (1 − cap_pct)
        if today_realized_pnl >= effective_best:
            # Today is the best day — percentage constraint binds.
            numerator = cap_pct * total_realized_pnl - today_realized_pnl
            denominator = 1.0 - cap_pct
            room_pct = numerator / denominator if denominator > 0 else 0.0
        else:
            # Today is NOT the best day — only the hard ceiling applies.
            room_pct = float("inf")

        # Room under the absolute cap:
        room_abs = effective_cap - today_realized_pnl

        allowed = max(min(room_pct, room_abs), 0.0)
        capped = allowed <= 0

        reason = "Within consistency limits."
        if capped:
            reason = (
                f"Consistency cap reached: best day would exceed "
                f"{cap_pct * 100:.0f}% of total P&L or the "
                f"${effective_cap:,.0f} daily ceiling."
            )

        return ConsistencyResult(
            allowed_profit_remaining=allowed,
            effective_daily_cap=effective_cap,
            best_day_pct=best_day_pct,
            cap_pct=cap_pct * 100,
            mode=self.mode,
            capped=capped,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    def _evaluate_xfa(
        self,
        total_realized_pnl: float,
        best_day_pnl: float,
        today_realized_pnl: float,
    ) -> ConsistencyResult:
        """Handle *xfa_standard* mode (no strict cap, optional soft cap)."""

        if CC_SOFT_CAP_ENABLED:
            soft_cap = CC_SOFT_CAP_AMOUNT
            allowed = max(soft_cap - today_realized_pnl, 0.0)
            capped = allowed <= 0
            reason = (
                f"Soft cap active: ${soft_cap:,.0f}/day."
                if not capped
                else f"Soft cap of ${soft_cap:,.0f} reached."
            )
        else:
            allowed = float("inf")
            capped = False
            soft_cap = 0.0
            reason = "No consistency cap for xfa_standard mode."

        best_day_pct = (
            (max(best_day_pnl, today_realized_pnl) / total_realized_pnl) * 100
            if total_realized_pnl > 0
            else 0.0
        )

        return ConsistencyResult(
            allowed_profit_remaining=allowed,
            effective_daily_cap=soft_cap,
            best_day_pct=best_day_pct,
            cap_pct=0.0,
            mode=self.mode,
            capped=capped,
            reason=reason,
        )


# ── Risk Governor ───────────────────────────────────────────────────────

class RiskGovernor:
    """Pre-trade gate that aggregates all session-level risk checks.

    Instantiate once per session and call :meth:`evaluate` before every
    prospective entry.  The governor is *stateless* — callers supply the
    running tallies.
    """

    def __init__(
        self,
        consistency_engine: ConsistencyCapEngine | None = None,
    ) -> None:
        self.consistency_engine = consistency_engine

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        daily_pnl: float,
        daily_trade_count: int,
        current_time_str_HHMM: str,
        total_realized_pnl: float = 0.0,
        best_day_pnl: float = 0.0,
    ) -> GovernorResult:
        """Run all pre-trade checks and return a :class:`GovernorResult`.

        Parameters
        ----------
        daily_pnl:
            Session P&L so far (positive = profit).
        daily_trade_count:
            Number of round-trip trades executed today.
        current_time_str_HHMM:
            Current time as ``"HH:MM"`` in ET.
        total_realized_pnl:
            Lifetime cumulative realised P&L (for consistency cap).
        best_day_pnl:
            Best single-day P&L recorded historically.
        """

        reasons: list[str] = []

        # 1) Daily loss halt
        if daily_pnl <= -RG_DAILY_LOSS_HALT:
            reasons.append(REJECT_DAILY_LOSS_HALT)
            logger.warning(
                "Daily loss halt triggered: P&L $%.2f ≤ -$%.2f",
                daily_pnl,
                RG_DAILY_LOSS_HALT,
            )

        # 2) Daily profit halt
        if daily_pnl >= RG_STRATEGY_DAILY_PROFIT_CAP:
            reasons.append(REJECT_DAILY_PROFIT_HALT)
            logger.info(
                "Daily profit halt: P&L $%.2f ≥ $%.2f cap",
                daily_pnl,
                RG_STRATEGY_DAILY_PROFIT_CAP,
            )

        # 3) Consistency cap
        if self.consistency_engine is not None and daily_pnl > 0:
            cc_result = self.consistency_engine.evaluate(
                total_realized_pnl=total_realized_pnl,
                best_day_pnl=best_day_pnl,
                today_realized_pnl=daily_pnl,
            )
            if cc_result.capped:
                reasons.append(REJECT_CONSISTENCY_CAP)
                logger.info(
                    "Consistency cap binding: %s", cc_result.reason
                )

        # 4) Max trades per day
        if daily_trade_count >= RG_MAX_TRADES_PER_DAY:
            reasons.append(REJECT_MAX_TRADES)
            logger.info(
                "Max trades reached: %d ≥ %d",
                daily_trade_count,
                RG_MAX_TRADES_PER_DAY,
            )

        # 5) No-trade windows
        if self._in_no_trade_window(current_time_str_HHMM):
            reasons.append(REJECT_NO_TRADE_WINDOW)
            logger.info("Inside no-trade window at %s", current_time_str_HHMM)

        # 6) Session cutoff
        if current_time_str_HHMM >= RG_FLATTEN_CUTOFF_TIME:
            reasons.append(REJECT_SESSION_CUTOFF)
            logger.info(
                "Past session cutoff %s (current %s)",
                RG_FLATTEN_CUTOFF_TIME,
                current_time_str_HHMM,
            )

        if reasons:
            return GovernorResult(approved=False, reasons=reasons)

        return GovernorResult(approved=True, reasons=[APPROVE])

    # ------------------------------------------------------------------ #
    @staticmethod
    def _in_no_trade_window(time_str: str) -> bool:
        """Return True if *time_str* (``"HH:MM"``) falls inside any
        configured no-trade window (inclusive on both ends)."""
        for start, end in RG_NO_TRADE_WINDOWS:
            if start <= time_str <= end:
                return True
        return False
