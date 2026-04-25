"""
tests/test_account_state.py — Topstep account/MLL risk snapshot tests.
"""

from __future__ import annotations

import pytest

from risk.account_state import AccountRiskSnapshot


def test_mll_floor_starts_at_starting_balance_minus_limit() -> None:
    snapshot = AccountRiskSnapshot(
        starting_balance=50_000.0,
        account_balance=50_000.0,
        account_high_water_mark=50_000.0,
        max_loss_limit=2_000.0,
    )

    assert snapshot.current_mll_floor == pytest.approx(48_000.0)
    assert snapshot.remaining_mll_headroom == pytest.approx(2_000.0)


def test_mll_floor_trails_high_water_mark() -> None:
    snapshot = AccountRiskSnapshot(
        starting_balance=50_000.0,
        account_balance=51_000.0,
        account_high_water_mark=51_000.0,
        max_loss_limit=2_000.0,
    )

    assert snapshot.current_mll_floor == pytest.approx(49_000.0)
    assert snapshot.remaining_mll_headroom == pytest.approx(2_000.0)


def test_mll_floor_locks_at_starting_balance() -> None:
    snapshot = AccountRiskSnapshot(
        starting_balance=50_000.0,
        account_balance=53_000.0,
        account_high_water_mark=53_000.0,
        max_loss_limit=2_000.0,
    )

    assert snapshot.current_mll_floor == pytest.approx(50_000.0)
    assert snapshot.remaining_mll_headroom == pytest.approx(3_000.0)


def test_current_equity_includes_unrealized_pnl() -> None:
    snapshot = AccountRiskSnapshot(
        starting_balance=50_000.0,
        account_balance=50_000.0,
        account_high_water_mark=50_000.0,
        daily_realized_pnl=100.0,
        daily_unrealized_pnl=-125.0,
        max_loss_limit=2_000.0,
    )

    assert snapshot.current_account_balance == pytest.approx(50_000.0)
    assert snapshot.current_equity == pytest.approx(49_875.0)
    assert snapshot.current_daily_realized_pnl == pytest.approx(100.0)
    assert snapshot.current_daily_unrealized_pnl == pytest.approx(-125.0)
