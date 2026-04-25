"""
tests/test_position_sizer.py — Tests for risk/position_sizer.py PositionSizer.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from risk.position_sizer import PositionSizer


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sizer():
    """Default sizer with $20 risk per trade."""
    return PositionSizer(risk_per_trade=20)


@pytest.fixture
def max_sizer():
    """Sizer configured at maximum risk ($40)."""
    return PositionSizer(risk_per_trade=40)


# ── Core sizing logic ──────────────────────────────────────────────────


class TestCoreSizing:
    def test_8_tick_stop(self, sizer):
        """$20 risk / 2-point stop ($10/contract) = 2 contracts."""
        # 8 ticks = 2 points (0.25 per tick × 8)
        contracts = sizer.calculate(stop_distance_points=2.0)
        assert contracts == 2

    def test_4_point_stop(self, sizer):
        """$20 risk / 4-point stop ($20/contract) = 1 contract."""
        contracts = sizer.calculate(stop_distance_points=4.0)
        assert contracts == 1

    def test_max_risk_4_point_stop(self, max_sizer):
        """$40 risk / 4-point stop ($20/contract) = 2 contracts."""
        contracts = max_sizer.calculate(stop_distance_points=4.0)
        assert contracts == 2

    def test_small_stop_capped_by_max_risk(self, max_sizer):
        """Very small stop shouldn't exceed RISK_PER_TRADE_MAX in total exposure."""
        # 1-point stop → $5/contract → $40/$5 = 8 contracts
        # But 8 × $5 = $40 which equals max, so 8 is allowed
        contracts = max_sizer.calculate(stop_distance_points=1.0)
        # Total exposure = contracts × 1.0 × 5.0 must be <= 40
        assert contracts * 1.0 * 5.0 <= 40.0
        assert contracts >= 1

    def test_large_stop_rejects(self, sizer):
        """A very large stop relative to risk cannot force 1 contract."""
        contracts = sizer.calculate(stop_distance_points=100.0)
        assert contracts == 0


# ── Stop distance edge cases ───────────────────────────────────────────


class TestStopDistanceEdgeCases:
    def test_zero_stop_rejects(self, sizer):
        """stop_distance <= 0 is invalid and rejects sizing."""
        assert sizer.calculate(stop_distance_points=0.0) == 0

    def test_negative_stop_rejects(self, sizer):
        """Negative stop distance is invalid and rejects sizing."""
        assert sizer.calculate(stop_distance_points=-5.0) == 0


# ── MLL proximity ──────────────────────────────────────────────────────


class TestMLLProximity:
    def test_mll_proximity_uses_min_risk(self, max_sizer):
        """When mll_proximity=True, use $20 min risk instead of configured $40."""
        # 4-point stop: $20/contract
        # Without proximity: $40/$20 = 2 contracts
        # With proximity: $20/$20 = 1 contract
        contracts_normal = max_sizer.calculate(stop_distance_points=4.0, mll_proximity=False)
        contracts_prox = max_sizer.calculate(stop_distance_points=4.0, mll_proximity=True)

        assert contracts_normal == 2
        assert contracts_prox == 1

    def test_mll_proximity_with_small_stop(self, max_sizer):
        """Even with small stop, proximity forces $20 risk."""
        # 2-point stop: $10/contract
        # Proximity: $20/$10 = 2 contracts
        contracts = max_sizer.calculate(stop_distance_points=2.0, mll_proximity=True)
        assert contracts == 2

    def test_mll_proximity_on_default_sizer(self, sizer):
        """On a $20 sizer, proximity doesn't change anything (already at min)."""
        contracts_normal = sizer.calculate(stop_distance_points=2.0, mll_proximity=False)
        contracts_prox = sizer.calculate(stop_distance_points=2.0, mll_proximity=True)
        assert contracts_normal == contracts_prox


# ── Minimum-contract risk gate ─────────────────────────────────────────


class TestMinimumContractRiskGate:
    def test_normal_stop_allows_1_contract(self, sizer):
        """$20 budget / 4-point MES stop ($20/contract) = 1 contract."""
        result = sizer.calculate_with_risk_gate(stop_distance_points=4.0)

        assert result.allowed is True
        assert result.quantity == 1
        assert result.risk_per_contract == pytest.approx(20.0)
        assert result.rejection_reason == ""

    def test_wide_stop_rejects_trade(self, sizer):
        """A 100-point MES stop risks $500 for one contract, above a $20 budget."""
        result = sizer.calculate_with_risk_gate(stop_distance_points=100.0)

        assert result.allowed is False
        assert result.quantity == 0
        assert "MIN_CONTRACT_RISK_EXCEEDS_TRADE_RISK" in result.rejection_reason

    def test_low_daily_loss_headroom_rejects_trade(self, sizer):
        """A normal 1-contract stop is rejected when daily loss budget is too low."""
        result = sizer.calculate_with_risk_gate(
            stop_distance_points=4.0,
            remaining_daily_loss_budget=10.0,
        )

        assert result.allowed is False
        assert result.quantity == 0
        assert "MIN_CONTRACT_RISK_EXCEEDS_DAILY_LOSS_BUDGET" in result.rejection_reason

    def test_low_mll_headroom_rejects_trade(self, sizer):
        """A normal 1-contract stop is rejected when MLL headroom is too low."""
        result = sizer.calculate_with_risk_gate(
            stop_distance_points=4.0,
            remaining_mll_headroom=10.0,
        )

        assert result.allowed is False
        assert result.quantity == 0
        assert "MIN_CONTRACT_RISK_EXCEEDS_MLL_HEADROOM" in result.rejection_reason

    def test_projected_trade_risk_rejects_against_mll_fraction(self, sizer):
        """The MLL gate uses projected sized-trade risk, not only one-contract risk."""
        result = sizer.calculate_with_risk_gate(
            stop_distance_points=1.0,
            remaining_mll_headroom=100.0,
            mll_headroom_safety_fraction=0.10,
        )

        assert result.allowed is False
        assert result.quantity == 0
        assert result.projected_trade_risk == pytest.approx(20.0)
        assert "PROJECTED_TRADE_RISK_EXCEEDS_MLL_HEADROOM" in result.rejection_reason

    def test_rejected_quantity_is_not_negative(self, sizer):
        for stop in [-10.0, -0.25, 0.0, 100.0]:
            contracts = sizer.calculate(stop_distance_points=stop)
            assert contracts == 0

    def test_dollars_per_point_is_configurable(self):
        """Non-MES point value changes minimum-contract risk calculation."""
        sizer = PositionSizer(risk_per_trade=20, dollars_per_point=10.0)
        result = sizer.calculate_with_risk_gate(stop_distance_points=2.0)

        assert result.allowed is True
        assert result.quantity == 1
        assert result.risk_per_contract == pytest.approx(20.0)


# ── Input validation ────────────────────────────────────────────────────


class TestInputValidation:
    def test_risk_below_min_raises(self):
        """risk_per_trade < 20 → ValueError."""
        with pytest.raises(ValueError, match="risk_per_trade"):
            PositionSizer(risk_per_trade=10)

    def test_risk_above_max_raises(self):
        """risk_per_trade > 40 → ValueError."""
        with pytest.raises(ValueError, match="risk_per_trade"):
            PositionSizer(risk_per_trade=50)

    def test_risk_at_min_ok(self):
        """risk_per_trade = 20 should not raise."""
        sizer = PositionSizer(risk_per_trade=20)
        assert sizer.risk_per_trade == 20

    def test_risk_at_max_ok(self):
        """risk_per_trade = 40 should not raise."""
        sizer = PositionSizer(risk_per_trade=40)
        assert sizer.risk_per_trade == 40


# ── Return type ─────────────────────────────────────────────────────────


class TestReturnType:
    def test_returns_int(self, sizer):
        result = sizer.calculate(stop_distance_points=2.0)
        assert isinstance(result, int)
