"""Tests for risk/combine_pass_state.py."""

from __future__ import annotations

import pytest

from risk.combine_pass_state import CombinePassSettings, CombinePassStateCalculator


@pytest.fixture
def calc() -> CombinePassStateCalculator:
    return CombinePassStateCalculator(
        CombinePassSettings(
            starting_balance=50_000.0,
            profit_target=3_000.0,
            max_loss_limit=2_000.0,
            consistency_cap_pct=0.50,
            mll_proximity_buffer=400.0,
        )
    )


def test_below_target_not_passing(calc: CombinePassStateCalculator) -> None:
    state = calc.calculate(
        current_cumulative_profit=2_500.0,
        current_best_day=1_200.0,
        todays_realized_pnl=300.0,
    )

    assert state.stopping_now_would_pass is False
    assert state.required_total_profit_under_consistency == pytest.approx(3_000.0)


def test_target_reached_consistency_satisfied_passing(calc: CombinePassStateCalculator) -> None:
    state = calc.calculate(
        current_cumulative_profit=3_100.0,
        current_best_day=1_400.0,
        todays_realized_pnl=500.0,
    )

    assert state.stopping_now_would_pass is True
    assert state.should_halt_new_trades is True


def test_target_reached_best_day_too_large_not_passing(calc: CombinePassStateCalculator) -> None:
    state = calc.calculate(
        current_cumulative_profit=3_000.0,
        current_best_day=2_000.0,
        todays_realized_pnl=200.0,
    )

    assert state.required_total_profit_under_consistency == pytest.approx(4_000.0)
    assert state.stopping_now_would_pass is False


def test_best_day_causes_required_target_inflation(calc: CombinePassStateCalculator) -> None:
    state = calc.calculate(
        current_cumulative_profit=2_600.0,
        current_best_day=1_700.0,
        todays_realized_pnl=100.0,
    )

    assert state.required_total_profit_under_consistency == pytest.approx(3_400.0)


def test_safe_profit_remaining_today(calc: CombinePassStateCalculator) -> None:
    state = calc.calculate(
        current_cumulative_profit=2_000.0,
        current_best_day=1_000.0,
        todays_realized_pnl=1_100.0,
    )

    assert state.maximum_allowed_best_day == pytest.approx(1_500.0)
    assert state.remaining_safe_profit_today_before_target_inflation == pytest.approx(400.0)


def test_low_mll_headroom_halt(calc: CombinePassStateCalculator) -> None:
    state = calc.calculate(
        current_cumulative_profit=-1_700.0,
        current_best_day=400.0,
        todays_realized_pnl=-100.0,
    )

    assert state.mll_headroom == pytest.approx(300.0)
    assert state.should_halt_new_trades is True


def test_stop_now_pass_condition_includes_unrealized(calc: CombinePassStateCalculator) -> None:
    state = calc.calculate(
        current_cumulative_profit=2_900.0,
        current_best_day=1_450.0,
        todays_realized_pnl=50.0,
        todays_unrealized_pnl=120.0,
    )

    assert state.stopping_now_would_pass is True
