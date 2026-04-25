"""risk/combine_pass_state.py — Topstep Combine pass-state calculator.

Computes pass readiness, consistency-driven target inflation, safe remaining
profit for today, and MLL headroom-based halt guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import (
    CONSISTENCY_CAP,
    MAX_LOSS_LIMIT,
    MLL_PROXIMITY_BUFFER,
    NOMINAL_ACCOUNT_SIZE,
    PROFIT_TARGET,
)
from risk.account_state import AccountRiskSnapshot


@dataclass(frozen=True)
class CombinePassSettings:
    """Configuration for Combine pass-state calculations."""

    starting_balance: float = float(NOMINAL_ACCOUNT_SIZE)
    profit_target: float = float(PROFIT_TARGET)
    max_loss_limit: float = float(MAX_LOSS_LIMIT)
    consistency_cap_pct: float = float(CONSISTENCY_CAP.get("combine", 0.50))
    mll_proximity_buffer: float = float(MLL_PROXIMITY_BUFFER)


@dataclass(frozen=True)
class CombinePassState:
    """Computed snapshot of current Combine pass status."""

    starting_balance: float
    current_cumulative_profit: float
    current_best_day: float
    todays_realized_pnl: float
    todays_unrealized_pnl: float
    profit_target: float
    consistency_cap_percentage: float
    maximum_allowed_best_day: float
    required_total_profit_under_consistency: float
    remaining_safe_profit_today_before_target_inflation: float
    current_mll_floor: float
    mll_headroom: float
    stopping_now_would_pass: bool
    should_halt_new_trades: bool


class CombinePassStateCalculator:
    """Pure calculator for Topstep-Combine pass state."""

    def __init__(self, settings: CombinePassSettings | None = None) -> None:
        self.settings = settings or CombinePassSettings()

    def calculate(
        self,
        *,
        current_cumulative_profit: float,
        current_best_day: float,
        todays_realized_pnl: float,
        todays_unrealized_pnl: float = 0.0,
        account_high_water_mark: float | None = None,
        halt_when_passed: bool = True,
    ) -> CombinePassState:
        """Calculate pass state using current account/session values.

        Notes
        -----
        * "Stopping now" assumes unrealized P&L is flattened immediately.
        * Consistency inflation uses:
          required_total >= max(profit_target, best_day / consistency_cap_pct)
        * Safe profit remaining before inflation uses:
          max_allowed_best_day = consistency_cap_pct * profit_target
        """

        s = self.settings
        cap = max(0.0, min(1.0, s.consistency_cap_pct))

        stop_now_cumulative = current_cumulative_profit + todays_unrealized_pnl
        stop_now_today = todays_realized_pnl + todays_unrealized_pnl
        stop_now_best_day = max(current_best_day, stop_now_today)

        maximum_allowed_best_day = s.profit_target * cap
        if cap > 0:
            required_total_under_consistency = max(
                s.profit_target,
                stop_now_best_day / cap,
            )
        else:
            required_total_under_consistency = s.profit_target

        remaining_safe_today = max(0.0, maximum_allowed_best_day - stop_now_today)
        stopping_now_would_pass = (
            stop_now_cumulative >= required_total_under_consistency
        )

        account_balance = s.starting_balance + current_cumulative_profit
        if account_high_water_mark is None:
            account_high_water_mark = max(s.starting_balance, account_balance)

        account = AccountRiskSnapshot(
            starting_balance=s.starting_balance,
            account_balance=account_balance,
            account_high_water_mark=account_high_water_mark,
            daily_realized_pnl=todays_realized_pnl,
            daily_unrealized_pnl=todays_unrealized_pnl,
            cumulative_realized_pnl=current_cumulative_profit,
            max_loss_limit=s.max_loss_limit,
            mll_locks_at_starting_balance=True,
        )

        mll_headroom = account.remaining_mll_headroom
        low_mll_headroom = mll_headroom <= s.mll_proximity_buffer
        consistency_halt = remaining_safe_today <= 0.0

        should_halt_new_trades = (
            low_mll_headroom
            or consistency_halt
            or (halt_when_passed and stopping_now_would_pass)
        )

        return CombinePassState(
            starting_balance=s.starting_balance,
            current_cumulative_profit=current_cumulative_profit,
            current_best_day=current_best_day,
            todays_realized_pnl=todays_realized_pnl,
            todays_unrealized_pnl=todays_unrealized_pnl,
            profit_target=s.profit_target,
            consistency_cap_percentage=cap,
            maximum_allowed_best_day=maximum_allowed_best_day,
            required_total_profit_under_consistency=required_total_under_consistency,
            remaining_safe_profit_today_before_target_inflation=remaining_safe_today,
            current_mll_floor=account.current_mll_floor,
            mll_headroom=mll_headroom,
            stopping_now_would_pass=stopping_now_would_pass,
            should_halt_new_trades=should_halt_new_trades,
        )
