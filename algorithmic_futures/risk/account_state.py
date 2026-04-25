"""
risk/account_state.py — account-level Topstep risk calculations.

This module is intentionally pure: it calculates account/equity, trailing MLL
floor, and remaining headroom without placing orders or changing strategy
signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import MAX_LOSS_LIMIT, NOMINAL_ACCOUNT_SIZE


@dataclass(frozen=True)
class AccountRiskSnapshot:
    """Point-in-time account state for pre-trade risk checks."""

    starting_balance: float = float(NOMINAL_ACCOUNT_SIZE)
    account_balance: float = 0.0
    account_high_water_mark: float = 0.0
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0
    cumulative_realized_pnl: float = 0.0
    max_loss_limit: float = float(MAX_LOSS_LIMIT)
    mll_locks_at_starting_balance: bool = True

    @classmethod
    def from_session_state(
        cls,
        state: Any,
        *,
        daily_unrealized_pnl: float = 0.0,
    ) -> AccountRiskSnapshot:
        """Build a snapshot from execution.order_manager.SessionState-like objects."""
        cumulative_realized = float(getattr(state, "cumulative_pnl", 0.0))
        account_balance = float(getattr(state, "account_balance", 0.0))
        if account_balance <= 0:
            account_balance = float(NOMINAL_ACCOUNT_SIZE) + cumulative_realized

        return cls(
            account_balance=account_balance,
            account_high_water_mark=float(getattr(state, "account_high_water_mark", 0.0)),
            daily_realized_pnl=float(getattr(state, "daily_pnl", 0.0)),
            daily_unrealized_pnl=float(daily_unrealized_pnl),
            cumulative_realized_pnl=cumulative_realized,
        )

    @property
    def current_account_balance(self) -> float:
        """Current realized account balance."""
        return self.account_balance

    @property
    def current_equity(self) -> float:
        """Current account equity, including unrealized P&L if supplied."""
        return self.current_account_balance + self.daily_unrealized_pnl

    @property
    def trailing_maximum_loss_limit(self) -> float:
        """Topstep trailing maximum loss limit amount."""
        return self.max_loss_limit

    @property
    def effective_high_water_mark(self) -> float:
        """Conservative high-water mark used to trail the MLL floor."""
        return max(self.starting_balance, self.account_high_water_mark, self.current_equity)

    @property
    def current_mll_floor(self) -> float:
        """Current MLL floor.

        Formula:
        - Initial floor = starting_balance - max_loss_limit.
        - Trailing floor = high_water_mark - max_loss_limit.
        - For Combine-style accounts, floor locks at starting_balance once the
          trailing floor reaches it.
        """
        initial_floor = self.starting_balance - self.max_loss_limit
        trailing_floor = max(initial_floor, self.effective_high_water_mark - self.max_loss_limit)
        if self.mll_locks_at_starting_balance:
            return min(self.starting_balance, trailing_floor)
        return trailing_floor

    @property
    def remaining_mll_headroom(self) -> float:
        """Current equity cushion above the MLL floor."""
        return max(0.0, self.current_equity - self.current_mll_floor)

    @property
    def current_daily_realized_pnl(self) -> float:
        return self.daily_realized_pnl

    @property
    def current_daily_unrealized_pnl(self) -> float:
        return self.daily_unrealized_pnl
