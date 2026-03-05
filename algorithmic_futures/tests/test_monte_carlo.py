"""
tests/test_monte_carlo.py — Tests for risk/monte_carlo.py MonteCarloValidator.
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from risk.monte_carlo import MonteCarloValidator, MonteCarloResult


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def validator():
    """Default validator with standard config."""
    return MonteCarloValidator(
        starting_capital=2000,
        ruin_boundary=0,
        target_boundary=5000,
        n_simulations=10_000,
        max_trades=200,
    )


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """100 % win / 100 % loss boundary scenarios."""

    def test_100_percent_loss_rate(self, validator):
        """Every trade is a loss → guaranteed ruin."""
        result = validator.run(win_rate=0.0, avg_win=30.0, avg_loss=-22.0, seed=42)

        assert isinstance(result, MonteCarloResult)
        assert result.ruin_probability == 1.0
        assert result.accepted is False
        assert len(result.rejection_reasons) > 0

    def test_100_percent_win_rate(self, validator):
        """Every trade is a win → guaranteed target hit, zero ruin."""
        result = validator.run(win_rate=1.0, avg_win=30.0, avg_loss=-22.0, seed=42)

        assert result.target_probability == 1.0
        assert result.ruin_probability == 0.0
        assert result.accepted is True
        assert len(result.rejection_reasons) == 0


# ── Target parameters ──────────────────────────────────────────────────


class TestTargetParameters:
    """Strategy-realistic parameters that should pass validation."""

    def test_target_params_pass_thresholds(self, validator):
        """55 % WR, $30 avg win, -$22 avg loss should satisfy all MC gates."""
        result = validator.run(win_rate=0.55, avg_win=30.0, avg_loss=-22.0, seed=123)

        # Ruin probability < 15 %
        assert result.ruin_probability < 0.15, (
            f"Ruin prob {result.ruin_probability:.2%} exceeds 15 %"
        )
        # Max drawdown p95 < $1200
        assert result.max_drawdown_p95 < 1200.0, (
            f"Drawdown p95 ${result.max_drawdown_p95:,.0f} exceeds $1,200"
        )

    def test_result_summary_contains_status(self, validator):
        """MonteCarloResult.summary() should include ACCEPTED or REJECTED."""
        result = validator.run(win_rate=0.55, avg_win=30.0, avg_loss=-22.0, seed=1)
        summary = result.summary()
        assert "ACCEPTED" in summary or "REJECTED" in summary


# ── Input validation ────────────────────────────────────────────────────


class TestInputValidation:
    """Invalid inputs must raise ValueError."""

    def test_win_rate_above_1(self, validator):
        with pytest.raises(ValueError, match="win_rate"):
            validator.run(win_rate=1.5, avg_win=30.0, avg_loss=-22.0)

    def test_win_rate_below_0(self, validator):
        with pytest.raises(ValueError, match="win_rate"):
            validator.run(win_rate=-0.1, avg_win=30.0, avg_loss=-22.0)

    def test_avg_win_negative(self, validator):
        with pytest.raises(ValueError, match="avg_win"):
            validator.run(win_rate=0.5, avg_win=-10.0, avg_loss=-22.0)

    def test_avg_win_zero(self, validator):
        with pytest.raises(ValueError, match="avg_win"):
            validator.run(win_rate=0.5, avg_win=0.0, avg_loss=-22.0)

    def test_avg_loss_positive(self, validator):
        with pytest.raises(ValueError, match="avg_loss"):
            validator.run(win_rate=0.5, avg_win=30.0, avg_loss=10.0)

    def test_avg_loss_zero(self, validator):
        with pytest.raises(ValueError, match="avg_loss"):
            validator.run(win_rate=0.5, avg_win=30.0, avg_loss=0.0)


# ── Reproducibility ────────────────────────────────────────────────────


class TestReproducibility:
    """Same seed must produce identical results."""

    def test_same_seed_same_result(self, validator):
        r1 = validator.run(win_rate=0.55, avg_win=30.0, avg_loss=-22.0, seed=999)
        r2 = validator.run(win_rate=0.55, avg_win=30.0, avg_loss=-22.0, seed=999)

        assert r1.ruin_probability == r2.ruin_probability
        assert r1.target_probability == r2.target_probability
        assert r1.max_drawdown_p95 == r2.max_drawdown_p95
        assert r1.max_losing_streak_p95 == r2.max_losing_streak_p95
        assert r1.accepted == r2.accepted
        assert r1.rejection_reasons == r2.rejection_reasons

    def test_different_seeds_vary(self, validator):
        """Different seeds should (almost certainly) produce different ruin probs."""
        r1 = validator.run(win_rate=0.55, avg_win=30.0, avg_loss=-22.0, seed=1)
        r2 = validator.run(win_rate=0.55, avg_win=30.0, avg_loss=-22.0, seed=2)

        # Not a strict guarantee, but with 10k sims they'll differ
        differs = (
            r1.ruin_probability != r2.ruin_probability
            or r1.max_drawdown_p95 != r2.max_drawdown_p95
        )
        assert differs, "Two different seeds produced identical results — extremely unlikely"


# ── Result structure ────────────────────────────────────────────────────


class TestResultStructure:
    """MonteCarloResult fields are well-formed."""

    def test_n_simulations_matches(self, validator):
        result = validator.run(win_rate=0.55, avg_win=30.0, avg_loss=-22.0, seed=42)
        assert result.n_simulations == 10_000

    def test_probabilities_in_range(self, validator):
        result = validator.run(win_rate=0.55, avg_win=30.0, avg_loss=-22.0, seed=42)
        assert 0.0 <= result.ruin_probability <= 1.0
        assert 0.0 <= result.target_probability <= 1.0

    def test_drawdown_non_negative(self, validator):
        result = validator.run(win_rate=0.55, avg_win=30.0, avg_loss=-22.0, seed=42)
        assert result.max_drawdown_p95 >= 0.0

    def test_frozen_dataclass(self, validator):
        result = validator.run(win_rate=0.55, avg_win=30.0, avg_loss=-22.0, seed=42)
        with pytest.raises(AttributeError):
            result.accepted = True  # type: ignore[misc]
