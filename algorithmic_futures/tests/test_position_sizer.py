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

    def test_large_stop_returns_1(self, sizer):
        """A very large stop relative to risk → floor to 1 contract."""
        contracts = sizer.calculate(stop_distance_points=100.0)
        assert contracts == 1


# ── Stop distance edge cases ───────────────────────────────────────────


class TestStopDistanceEdgeCases:
    def test_zero_stop_returns_1(self, sizer):
        """stop_distance <= 0 → safe default of 1 contract."""
        assert sizer.calculate(stop_distance_points=0.0) == 1

    def test_negative_stop_returns_1(self, sizer):
        """Negative stop distance → safe default of 1 contract."""
        assert sizer.calculate(stop_distance_points=-5.0) == 1


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


# ── Never zero or negative ─────────────────────────────────────────────


class TestMinimumContracts:
    def test_never_returns_zero(self, sizer):
        """Contract count is always >= 1."""
        for stop in [0.0, -1.0, 0.01, 100.0, 1000.0]:
            contracts = sizer.calculate(stop_distance_points=stop)
            assert contracts >= 1, f"Got {contracts} for stop={stop}"

    def test_never_returns_negative(self, sizer):
        for stop in [-10.0, -0.25, 0.0]:
            contracts = sizer.calculate(stop_distance_points=stop)
            assert contracts > 0


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
