"""
tests/test_circuit_breakers.py — Tests for execution/circuit_breakers.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime
from typing import Any

import pytz

from execution.circuit_breakers import CircuitBreakers, BreakerCheckResult, BreakerEvent
from regime.regime_state import BreakerType, RegimeState

ET = pytz.timezone("US/Eastern")


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_time(hour: int, minute: int) -> datetime:
    """Create an ET-aware datetime well within trading hours."""
    return ET.localize(datetime(2026, 2, 20, hour, minute, 0))


NORMAL_TIME = _make_time(10, 30)  # 10:30 AM ET — safely inside RTH


@pytest.fixture
def cb():
    return CircuitBreakers(account_mode="combine")


def _check_normal(cb: CircuitBreakers, **overrides) -> BreakerCheckResult:
    """Run check_all with safe defaults; override any parameter."""
    defaults: dict[str, Any] = dict(
        daily_pnl=0.0,
        cumulative_pnl=500.0,
        account_balance=50_000.0,
        account_high_water_mark=50_000.0,
        daily_trade_count=0,
        active_strategy="VWAP",
        current_regime=RegimeState.BALANCED,
        now=NORMAL_TIME,
    )
    defaults.update(overrides)
    return cb.check_all(**defaults)


# ── Normal conditions ───────────────────────────────────────────────────


class TestNormalConditions:
    def test_all_clear(self, cb):
        result = _check_normal(cb)
        assert result.allowed is True
        assert result.active_breakers == []
        assert result.mll_proximity is False


# ── Daily loss breaker ──────────────────────────────────────────────────


class TestDailyLoss:
    def test_daily_loss_at_limit(self, cb):
        """daily_pnl == -240 fires DAILY_LOSS."""
        result = _check_normal(cb, daily_pnl=-240.0)
        assert result.allowed is False
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.DAILY_LOSS in types

    def test_daily_loss_beyond_limit(self, cb):
        result = _check_normal(cb, daily_pnl=-300.0)
        assert result.allowed is False
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.DAILY_LOSS in types

    def test_daily_loss_just_above_limit(self, cb):
        """daily_pnl = -239 should NOT fire DAILY_LOSS."""
        result = _check_normal(cb, daily_pnl=-239.0)
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.DAILY_LOSS not in types


# ── Daily profit halt ───────────────────────────────────────────────────


class TestDailyProfit:
    def test_profit_halt_at_limit(self, cb):
        """daily_pnl == +1200 fires DAILY_PROFIT."""
        result = _check_normal(cb, daily_pnl=1200.0)
        assert result.allowed is False
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.DAILY_PROFIT in types

    def test_profit_halt_above_limit(self, cb):
        result = _check_normal(cb, daily_pnl=1500.0)
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.DAILY_PROFIT in types

    def test_profit_below_limit_ok(self, cb):
        result = _check_normal(cb, daily_pnl=1199.0)
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.DAILY_PROFIT not in types


# ── Trade count cap ─────────────────────────────────────────────────────


class TestTradeCount:
    def test_vwap_trade_count_at_max(self, cb):
        """3 trades for VWAP strategy fires TRADE_COUNT."""
        result = _check_normal(cb, daily_trade_count=3, active_strategy="VWAP")
        assert result.allowed is False
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.TRADE_COUNT in types

    def test_vwap_trade_count_below_max(self, cb):
        result = _check_normal(cb, daily_trade_count=2, active_strategy="VWAP")
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.TRADE_COUNT not in types

    def test_orb_trade_count_at_max(self, cb):
        """2 trades for ORB strategy fires TRADE_COUNT."""
        result = _check_normal(cb, daily_trade_count=2, active_strategy="ORB")
        assert result.allowed is False
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.TRADE_COUNT in types

    def test_orb_trade_count_below_max(self, cb):
        result = _check_normal(cb, daily_trade_count=1, active_strategy="ORB")
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.TRADE_COUNT not in types


# ── Crisis regime ───────────────────────────────────────────────────────


class TestCrisisRegime:
    def test_crisis_fires_breaker(self, cb):
        result = _check_normal(cb, current_regime=RegimeState.CRISIS)
        assert result.allowed is False
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.CRISIS_REGIME in types

    def test_balanced_no_crisis(self, cb):
        result = _check_normal(cb, current_regime=RegimeState.BALANCED)
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.CRISIS_REGIME not in types

    def test_directional_no_crisis(self, cb):
        result = _check_normal(cb, current_regime=RegimeState.DIRECTIONAL)
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.CRISIS_REGIME not in types


# ── MLL proximity ──────────────────────────────────────────────────────


class TestMLLProximity:
    def test_within_buffer(self, cb):
        """Account balance within $400 of MLL → mll_proximity = True."""
        # HWM 50,000 and MLL 2,000:
        # drawdown 1,700 => distance-to-MLL = 300 (within $400 buffer)
        result = _check_normal(
            cb,
            account_high_water_mark=50_000.0,
            account_balance=48_300.0,
        )
        assert result.mll_proximity is True

    def test_at_buffer_boundary(self, cb):
        """Exactly at MLL + buffer → mll_proximity = True."""
        # drawdown 1,600 => distance-to-MLL = 400 (boundary)
        result = _check_normal(
            cb,
            account_high_water_mark=50_000.0,
            account_balance=48_400.0,
        )
        assert result.mll_proximity is True

    def test_above_buffer(self, cb):
        """Safely above MLL + buffer → mll_proximity = False."""
        # drawdown 1,500 => distance-to-MLL = 500 (outside buffer)
        result = _check_normal(
            cb,
            account_high_water_mark=50_000.0,
            account_balance=48_500.0,
        )
        assert result.mll_proximity is False

    def test_mll_proximity_does_not_block_trading_alone(self, cb):
        """MLL proximity is a sizing warning, not a halt — allowed can still be True."""
        result = _check_normal(
            cb,
            account_high_water_mark=50_000.0,
            account_balance=48_300.0,
        )
        # No other breakers are active, so allowed should be True
        assert result.mll_proximity is True
        assert result.allowed is True


# ── Consistency cap ─────────────────────────────────────────────────────


class TestConsistencyCap:
    def test_consistency_cap_fires(self, cb):
        """daily_pnl >= 50 % of cumulative → CONSISTENCY_CAP fires (combine mode)."""
        # cumulative = 1000, cap = 50% → 500. daily_pnl = 500 should fire.
        result = _check_normal(cb, daily_pnl=500.0, cumulative_pnl=1000.0)
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.CONSISTENCY_CAP in types

    def test_consistency_cap_below_threshold(self, cb):
        """daily_pnl < 50 % of cumulative → no cap breach."""
        result = _check_normal(cb, daily_pnl=400.0, cumulative_pnl=1000.0)
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.CONSISTENCY_CAP not in types

    def test_consistency_cap_zero_cumulative(self, cb):
        """If cumulative_pnl <= 0, consistency cap should not fire."""
        result = _check_normal(cb, daily_pnl=200.0, cumulative_pnl=0.0)
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.CONSISTENCY_CAP not in types


# ── Multiple simultaneous breakers ──────────────────────────────────────


class TestMultipleBreakers:
    def test_loss_and_crisis_together(self, cb):
        """Both DAILY_LOSS and CRISIS_REGIME can fire simultaneously."""
        result = _check_normal(
            cb,
            daily_pnl=-300.0,
            current_regime=RegimeState.CRISIS,
        )
        assert result.allowed is False
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.DAILY_LOSS in types
        assert BreakerType.CRISIS_REGIME in types
        assert len(result.active_breakers) >= 2

    def test_trade_count_and_profit_together(self, cb):
        """TRADE_COUNT and DAILY_PROFIT can fire simultaneously."""
        result = _check_normal(
            cb,
            daily_pnl=1200.0,
            daily_trade_count=3,
            active_strategy="VWAP",
        )
        types = [e.breaker for e in result.active_breakers]
        assert BreakerType.DAILY_PROFIT in types
        assert BreakerType.TRADE_COUNT in types


# ── Reset ───────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_event_log(self, cb):
        """After firing breakers, reset() should clear the internal event log."""
        _check_normal(cb, daily_pnl=-300.0)
        assert len(cb.events) > 0

        cb.reset()
        assert len(cb.events) == 0

    def test_reset_does_not_affect_future_checks(self, cb):
        """After reset, normal conditions should pass cleanly."""
        _check_normal(cb, daily_pnl=-300.0)
        cb.reset()

        result = _check_normal(cb)
        assert result.allowed is True


# ── BreakerCheckResult API ──────────────────────────────────────────────


class TestBreakerCheckResultAPI:
    def test_reasons_property(self, cb):
        result = _check_normal(cb, daily_pnl=-250.0)
        assert len(result.reasons) > 0
        assert all(isinstance(r, str) for r in result.reasons)

    def test_breaker_event_fields(self, cb):
        result = _check_normal(cb, daily_pnl=-250.0)
        event = result.active_breakers[0]
        assert isinstance(event, BreakerEvent)
        assert event.breaker == BreakerType.DAILY_LOSS
        assert event.timestamp  # non-empty string
        assert event.message  # non-empty string


class TestCombinePreTradeGuards:
    def test_pass_state_reached_halts(self, cb):
        result = _check_normal(
            cb,
            cumulative_pnl=3_100.0,
            daily_pnl=100.0,
            current_best_day_pnl=1_200.0,
        )
        assert result.allowed is False
        assert "PASS_STATE_REACHED" in result.reasons

    def test_mll_headroom_too_low_halts(self, cb):
        result = _check_normal(
            cb,
            cumulative_pnl=-1_850.0,
            account_balance=48_150.0,
            account_high_water_mark=50_000.0,
            projected_trade_risk=20.0,
        )
        assert result.allowed is False
        assert "MLL_HEADROOM_TOO_LOW" in result.reasons

    def test_consistency_cap_risk_halts(self, cb):
        result = _check_normal(
            cb,
            daily_pnl=1_495.0,
            cumulative_pnl=2_000.0,
            current_best_day_pnl=1_450.0,
            projected_trade_risk=10.0,
        )
        assert result.allowed is False
        assert "CONSISTENCY_CAP_RISK" in result.reasons

    def test_daily_loss_budget_low_halts(self, cb):
        result = _check_normal(
            cb,
            daily_pnl=-230.0,
            projected_trade_risk=20.0,
        )
        assert result.allowed is False
        assert "DAILY_LOSS_BUDGET_LOW" in result.reasons
