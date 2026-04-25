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
    CONSISTENCY_CAP_PROJECTED_RISK_FRACTION,
    CONSISTENCY_CAP_RISK_MODE,
    CONSISTENCY_CAP,
    DAILY_LOSS_LIMIT_EXTERNAL,
    DAILY_LOSS_LIMIT_INTERNAL,
    DAILY_PROFIT_HALT,
    EOD_CLOSE,
    HALT_ON_PASS_STATE_REACHED,
    LAST_ENTRY_CUTOFF,
    MLL_PROXIMITY_BUFFER,
    ORB_TRADES_PER_DAY,
    PRETRADE_DAILY_LOSS_BUDGET_MIN,
    PRETRADE_DAILY_RISK_BUDGET_FRACTION,
    PRETRADE_MLL_HEADROOM_MIN,
    PRETRADE_MLL_PROJECTED_RISK_FRACTION,
    TIMEZONE,
    VWAP_TRADES_PER_DAY,
)
from regime.regime_state import BreakerType, RegimeState
from risk.combine_pass_state import CombinePassStateCalculator
from risk.account_state import AccountRiskSnapshot

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
        self._combine_pass_calc = CombinePassStateCalculator()

    PASS_STATE_REACHED = "PASS_STATE_REACHED"
    MLL_HEADROOM_TOO_LOW = "MLL_HEADROOM_TOO_LOW"
    CONSISTENCY_CAP_RISK = "CONSISTENCY_CAP_RISK"
    DAILY_LOSS_BUDGET_LOW = "DAILY_LOSS_BUDGET_LOW"
    MIN_CONTRACT_RISK_TOO_HIGH = "MIN_CONTRACT_RISK_TOO_HIGH"

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
        current_best_day_pnl: float = 0.0,
        projected_trade_risk: float = 0.0,
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
        account_risk: AccountRiskSnapshot | None = None
        if account_high_water_mark > 0 or account_balance > 0:
            account_risk = AccountRiskSnapshot(
                account_balance=account_balance,
                account_high_water_mark=account_high_water_mark,
                daily_realized_pnl=daily_pnl,
                cumulative_realized_pnl=cumulative_pnl,
            )
            distance_to_mll = account_risk.remaining_mll_headroom
            if distance_to_mll <= MLL_PROXIMITY_BUFFER:
                mll_proximity = True
                logger.warning(
                    "MLL proximity: floor $%.0f, distance-to-MLL $%.0f <= $%d",
                    account_risk.current_mll_floor, distance_to_mll, MLL_PROXIMITY_BUFFER,
                )

        # 5b. Combine pass-state / pre-trade risk guards
        pass_state = self._combine_pass_calc.calculate(
            current_cumulative_profit=cumulative_pnl,
            current_best_day=current_best_day_pnl,
            todays_realized_pnl=daily_pnl,
            todays_unrealized_pnl=0.0,
            account_high_water_mark=account_high_water_mark if account_high_water_mark > 0 else None,
            halt_when_passed=HALT_ON_PASS_STATE_REACHED,
        )
        if HALT_ON_PASS_STATE_REACHED and pass_state.stopping_now_would_pass:
            active.append(self._event(
                BreakerType.DAILY_PROFIT,
                pass_state.current_cumulative_profit,
                pass_state.required_total_profit_under_consistency,
                self.PASS_STATE_REACHED,
            ))

        if pass_state.mll_headroom <= PRETRADE_MLL_HEADROOM_MIN:
            active.append(self._event(
                BreakerType.MLL_PROXIMITY,
                pass_state.mll_headroom,
                PRETRADE_MLL_HEADROOM_MIN,
                self.MLL_HEADROOM_TOO_LOW,
            ))

        projected_risk = max(0.0, projected_trade_risk)
        if projected_risk > 0 and pass_state.mll_headroom > 0:
            max_mll_projected_risk = pass_state.mll_headroom * PRETRADE_MLL_PROJECTED_RISK_FRACTION
            if projected_risk > max_mll_projected_risk:
                active.append(self._event(
                    BreakerType.MLL_PROXIMITY,
                    projected_risk,
                    max_mll_projected_risk,
                    self.MLL_HEADROOM_TOO_LOW,
                ))

        remaining_daily_budget = max(0.0, DAILY_LOSS_LIMIT_INTERNAL + daily_pnl)
        if remaining_daily_budget <= PRETRADE_DAILY_LOSS_BUDGET_MIN:
            active.append(self._event(
                BreakerType.DAILY_LOSS,
                remaining_daily_budget,
                PRETRADE_DAILY_LOSS_BUDGET_MIN,
                self.DAILY_LOSS_BUDGET_LOW,
            ))
        if projected_risk > 0 and remaining_daily_budget > 0:
            max_daily_projected_risk = remaining_daily_budget * PRETRADE_DAILY_RISK_BUDGET_FRACTION
            if projected_risk > max_daily_projected_risk:
                active.append(self._event(
                    BreakerType.DAILY_LOSS,
                    projected_risk,
                    max_daily_projected_risk,
                    self.DAILY_LOSS_BUDGET_LOW,
                ))

        if projected_risk > 0:
            max_consistency_risk = (
                pass_state.remaining_safe_profit_today_before_target_inflation
                * CONSISTENCY_CAP_PROJECTED_RISK_FRACTION
            )
            if projected_risk > max_consistency_risk:
                if CONSISTENCY_CAP_RISK_MODE == "reduce":
                    mll_proximity = True
                else:
                    active.append(self._event(
                        BreakerType.CONSISTENCY_CAP,
                        projected_risk,
                        max_consistency_risk,
                        self.CONSISTENCY_CAP_RISK,
                    ))

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
