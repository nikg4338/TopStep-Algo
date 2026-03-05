"""
execution/circuit_breakers.py — Safety-critical trading halt logic.

Runs as a gate before EVERY order placement.  If any breaker is active,
the order is rejected and trading halts for the remainder of the session.

Breakers:
  1. Daily Loss Hard Stop         -$240 realized
  2. Topstep External Limit Guard -$1,000 realized
  3. Daily Profit Halt            +$1,200 realized
  4. Consistency Cap Check        any day ≥ cap% of cumulative profit
  5. MLL Proximity Warning        account within $400 of MLL
  6. Trade Count Cap              per-strategy daily trade limit
  7. EOD Time Stop                current time ≥ EOD_CLOSE
  8. Crisis Regime Stop           HMM State 2 active
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

import pytz

from config import (
    ACCOUNT_MODE,
    CONSISTENCY_CAP,
    DAILY_LOSS_LIMIT_EXTERNAL,
    DAILY_LOSS_LIMIT_INTERNAL,
    DAILY_PROFIT_HALT,
    EOD_CLOSE,
    LAST_ENTRY_CUTOFF,
    MAX_LOSS_LIMIT,
    MLL_PROXIMITY_BUFFER,
    ORB_TRADES_PER_DAY,
    TIMEZONE,
    VWAP_TRADES_PER_DAY,
)
from regime.regime_state import BreakerType, RegimeState

logger = logging.getLogger(__name__)

ET = pytz.timezone(TIMEZONE)


@dataclass
class BreakerEvent:
    """Immutable record of a circuit breaker trigger."""
    breaker: BreakerType
    timestamp: str
    trigger_value: float | int | str
    threshold: float | int | str
    message: str


@dataclass
class BreakerCheckResult:
    """Result of a full breaker sweep."""
    allowed: bool
    active_breakers: list[BreakerEvent] = field(default_factory=list)
    mll_proximity: bool = False  # True → reduce sizing to minimum

    @property
    def reasons(self) -> list[str]:
        return [e.message for e in self.active_breakers]


class CircuitBreakers:
    """Stateless breaker evaluator.

    Call ``check_all()`` before every order placement.  Reads current
    session state and returns a BreakerCheckResult.
    """

    def __init__(self, account_mode: str = ACCOUNT_MODE) -> None:
        self.account_mode = account_mode
        self._events: list[BreakerEvent] = []  # session log

    @property
    def events(self) -> list[BreakerEvent]:
        return list(self._events)

    def reset(self) -> None:
        """Clear breaker event log at session open."""
        self._events.clear()

    # ── main gate ───────────────────────────────────────────────────────

    def check_all(
        self,
        daily_pnl: float,
        cumulative_pnl: float,
        account_balance: float,
        account_high_water_mark: float,
        daily_trade_count: int,
        active_strategy: str,  # "VWAP" or "ORB"
        current_regime: RegimeState,
        now: datetime | None = None,
    ) -> BreakerCheckResult:
        """Evaluate all 8 circuit breakers. Returns ``BreakerCheckResult``."""
        now = now or datetime.now(ET)
        active: list[BreakerEvent] = []
        mll_proximity = False

        # 1. Daily loss hard stop
        if daily_pnl <= -DAILY_LOSS_LIMIT_INTERNAL:
            active.append(self._event(
                BreakerType.DAILY_LOSS, daily_pnl, -DAILY_LOSS_LIMIT_INTERNAL,
                f"Daily P&L ${daily_pnl:.0f} hit internal limit -${DAILY_LOSS_LIMIT_INTERNAL}",
            ))

        # 2. External loss limit guard
        if daily_pnl <= -DAILY_LOSS_LIMIT_EXTERNAL:
            active.append(self._event(
                BreakerType.EXTERNAL_LOSS, daily_pnl, -DAILY_LOSS_LIMIT_EXTERNAL,
                f"Daily P&L ${daily_pnl:.0f} hit Topstep external limit -${DAILY_LOSS_LIMIT_EXTERNAL}",
            ))

        # 3. Daily profit halt
        if daily_pnl >= DAILY_PROFIT_HALT:
            active.append(self._event(
                BreakerType.DAILY_PROFIT, daily_pnl, DAILY_PROFIT_HALT,
                f"Daily P&L ${daily_pnl:.0f} hit profit halt +${DAILY_PROFIT_HALT}",
            ))

        # 4. Consistency cap (mode-based)
        cap_pct = CONSISTENCY_CAP.get(self.account_mode, 0.50)
        if cumulative_pnl > 0:
            max_day_allowed = cumulative_pnl * cap_pct
            # Check if today's P&L would violate the cap
            if daily_pnl >= max_day_allowed:
                active.append(self._event(
                    BreakerType.CONSISTENCY_CAP, daily_pnl, max_day_allowed,
                    f"Daily P&L ${daily_pnl:.0f} ≥ {cap_pct:.0%} of cumulative ${cumulative_pnl:.0f} "
                    f"(cap ${max_day_allowed:.0f})",
                ))

        # 5. MLL proximity warning (sizing reduction, not a hard halt)
        # Trailing drawdown framing:
        #   drawdown = high_water_mark - current_balance
        #   distance_to_mll = MAX_LOSS_LIMIT - drawdown
        if account_high_water_mark > 0:
            drawdown = max(0.0, account_high_water_mark - account_balance)
            distance_to_mll = MAX_LOSS_LIMIT - drawdown
            if distance_to_mll <= MLL_PROXIMITY_BUFFER:
                mll_proximity = True
                logger.warning(
                    "MLL proximity: drawdown $%.0f, distance-to-MLL $%.0f <= $%d",
                    drawdown, distance_to_mll, MLL_PROXIMITY_BUFFER,
                )

        # 6. Trade count cap
        max_trades = VWAP_TRADES_PER_DAY if active_strategy.upper() == "VWAP" else ORB_TRADES_PER_DAY
        if daily_trade_count >= max_trades:
            active.append(self._event(
                BreakerType.TRADE_COUNT, daily_trade_count, max_trades,
                f"Trade count {daily_trade_count} ≥ max {max_trades} for {active_strategy}",
            ))

        # 7. EOD time stop
        cutoff_hour_str, cutoff_minute_str = LAST_ENTRY_CUTOFF.split(":")
        cutoff_time = time(hour=int(cutoff_hour_str), minute=int(cutoff_minute_str))
        current_time = now.time()
        if current_time >= cutoff_time:
            active.append(self._event(
                BreakerType.EOD_TIME_STOP, str(current_time), LAST_ENTRY_CUTOFF,
                f"Time {current_time} past last entry cutoff {LAST_ENTRY_CUTOFF}",
            ))

        # 8. Crisis regime
        if current_regime == RegimeState.CRISIS:
            active.append(self._event(
                BreakerType.CRISIS_REGIME, current_regime.name, "CRISIS",
                "Crisis regime active — no trading permitted",
            ))

        # Record events
        self._events.extend(active)

        result = BreakerCheckResult(
            allowed=len(active) == 0,
            active_breakers=active,
            mll_proximity=mll_proximity,
        )

        if not result.allowed:
            for evt in active:
                logger.warning("CIRCUIT BREAKER: %s", evt.message)

        return result

    # ── helpers ─────────────────────────────────────────────────────────

    def _event(
        self,
        breaker: BreakerType,
        trigger_value: Any,
        threshold: Any,
        message: str,
    ) -> BreakerEvent:
        return BreakerEvent(
            breaker=breaker,
            timestamp=datetime.now(ET).isoformat(),
            trigger_value=trigger_value,
            threshold=threshold,
            message=message,
        )
