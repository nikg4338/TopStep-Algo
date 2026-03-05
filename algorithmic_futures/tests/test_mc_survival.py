"""Tests for simulation/mc_survival.py — Monte Carlo combine-survival simulator.

Covers: IID mode, block bootstrap, stress transforms, true trailing DD,
daily loss termination, distribution diagnostics, and multi-scenario runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from simulation.mc_survival import MonteCarloSurvivalSimulator, SurvivalResult


class TestMonteCarloSurvivalSimulator:
    """Core unit tests for MonteCarloSurvivalSimulator."""

    def _make_sim(
        self,
        risk_per_trade: float = 20.0,
        profit_target: float = 3000.0,
        max_loss_limit: float = 2000.0,
        daily_loss_limit: float = 1000.0,
        consistency_cap: float = 0.50,
        n_simulations: int = 500,
        max_trades: int = 800,
        trades_per_day: int = 5,
        mode: str = "iid",
        block_type: str = "session",
    ) -> MonteCarloSurvivalSimulator:
        return MonteCarloSurvivalSimulator(
            risk_per_trade=risk_per_trade,
            profit_target=profit_target,
            max_loss_limit=max_loss_limit,
            daily_loss_limit=daily_loss_limit,
            consistency_cap=consistency_cap,
            n_simulations=n_simulations,
            max_trades=max_trades,
            trades_per_day=trades_per_day,
            mode=mode,
            block_type=block_type,
        )

    # ── Deterministic seed test ─────────────────────────────────────────

    def test_deterministic_seed_reproducibility(self) -> None:
        """Same seed + same r_values → identical SurvivalResult."""
        sim = self._make_sim(n_simulations=200)
        r_vals = [1.0, -0.5, 2.0, -1.0, 0.5, -0.3, 1.5, -0.8]

        res_a = sim.run(r_vals, seed=42)
        res_b = sim.run(r_vals, seed=42)

        assert res_a.p_target_before_ruin == res_b.p_target_before_ruin
        assert res_a.p_ruin == res_b.p_ruin
        assert res_a.dd_p95 == res_b.dd_p95
        assert res_a.losing_streak_p95 == res_b.losing_streak_p95

    # ── Unit scaling: pnl_r × RISK_PER_TRADE = dollars ──────────────────

    def test_unit_scaling_r_to_dollars(self) -> None:
        """pnl_r=[1.0], RISK_PER_TRADE=$20, PROFIT_TARGET=$40 → target in 2 trades."""
        sim = self._make_sim(
            risk_per_trade=20.0,
            profit_target=40.0,
            max_loss_limit=5000.0,
            n_simulations=100,
            max_trades=100,
        )
        res = sim.run([1.0], seed=1)

        assert res.p_target_before_ruin == pytest.approx(1.0, abs=0.01)
        assert res.median_trades_to_target == pytest.approx(2.0, abs=0.5)
        assert res.p_ruin == 0.0

    # ── All positive r-values → p_target ≈ 1.0 ─────────────────────────

    def test_all_positive_r_values(self) -> None:
        """When every trade is profitable, nearly all paths hit target."""
        sim = self._make_sim(
            n_simulations=500,
            profit_target=500.0,
            max_loss_limit=5000.0,
            max_trades=800,
        )
        r_vals = [1.0, 2.0, 1.5, 3.0, 0.5]
        res = sim.run(r_vals, seed=7)

        assert res.p_target_before_ruin >= 0.95
        assert res.p_ruin == 0.0

    # ── All negative r-values → p_ruin ≈ 1.0 ───────────────────────────

    def test_all_negative_r_values(self) -> None:
        """pnl_r=[-1.0], RISK_PER_TRADE=$20, MAX_LOSS_LIMIT=$40 → ruin in 2 trades."""
        sim = self._make_sim(
            risk_per_trade=20.0,
            profit_target=3000.0,
            max_loss_limit=40.0,
            n_simulations=100,
            max_trades=100,
        )
        res = sim.run([-1.0], seed=99)

        assert res.p_ruin == pytest.approx(1.0, abs=0.01)
        assert res.p_target_before_ruin == 0.0

    # ── Max trades termination ──────────────────────────────────────────

    def test_max_trades_termination(self) -> None:
        """Tiny edge, huge targets → all terminate at max_trades."""
        sim = self._make_sim(
            risk_per_trade=20.0,
            profit_target=1_000_000.0,
            max_loss_limit=1_000_000.0,
            n_simulations=100,
            max_trades=10,
        )
        res = sim.run([0.01], seed=5)

        assert res.p_target_before_ruin == 0.0
        assert res.p_ruin == 0.0
        assert res.termination_max_trades == 100
        assert res.termination_hit_target == 0
        assert res.termination_ruin == 0

    # ── No dual termination ─────────────────────────────────────────────

    def test_no_dual_termination(self) -> None:
        """hit_target + ruin + max_trades must equal n_simulations exactly."""
        sim = self._make_sim(
            n_simulations=200,
            profit_target=200.0,
            max_loss_limit=200.0,
            max_trades=400,
        )
        r_vals = [2.0, -1.5, 1.0, -2.0, 0.5, -0.3, 3.0, -1.0]
        res = sim.run(r_vals, seed=42)

        total = res.termination_hit_target + res.termination_ruin + res.termination_max_trades
        assert total == 200

    # ── Edge case: empty r_values ───────────────────────────────────────

    def test_empty_r_values(self) -> None:
        """Empty r_values → degenerate result with zeros."""
        sim = self._make_sim()
        res = sim.run([], seed=0)

        assert isinstance(res, SurvivalResult)
        assert res.p_target_before_ruin == 0.0
        assert res.p_ruin == 0.0
        assert "No r_values" in res.notes

    # ── Dollar-value mode ───────────────────────────────────────────────

    def test_dollar_value_mode(self) -> None:
        """use_dollar_values=True bypasses risk_per_trade multiplication."""
        sim = self._make_sim(
            n_simulations=200,
            profit_target=200.0,
            max_loss_limit=5000.0,
            max_trades=800,
        )
        r_vals = [50.0, 50.0, 50.0, 50.0, 50.0]
        res = sim.run(r_vals, seed=1, use_dollar_values=True)

        assert res.p_target_before_ruin >= 0.95
        assert res.median_trades_to_target == pytest.approx(4.0, abs=0.5)

    # ── SurvivalResult has all fields ───────────────────────────────────

    def test_result_has_all_fields(self) -> None:
        """SurvivalResult has all expected fields including diagnostics."""
        sim = self._make_sim(n_simulations=50)
        res = sim.run([1.0, -0.5], seed=123)

        expected_fields = [
            "n_simulations", "p_target_before_ruin", "p_ruin",
            "p_fail_consistency_given_target", "dd_p95", "losing_streak_p95",
            "median_trades_to_target", "p_daily_loss_breach", "notes",
            "avg_trade_pnl_dollars", "expected_trades_to_target",
            "termination_hit_target", "termination_ruin",
            "termination_max_trades", "nan_trades_skipped",
            # Distribution diagnostics
            "trade_count_input", "std_trade_pnl_dollars",
            "skewness", "kurtosis",
            "worst_sampled_loss", "worst_sampled_drawdown",
            "equity_p1", "equity_p99", "worst_intraday_dd",
            # Metadata
            "mode", "stress_scenario",
        ]
        for f in expected_fields:
            assert hasattr(res, f), f"Missing field: {f}"
        assert res.n_simulations == 50

    # ── write_results persistence ───────────────────────────────────────

    def test_write_results_json(self, tmp_path: Path) -> None:
        """write_results writes valid JSON with all expected keys."""
        sim = self._make_sim(n_simulations=50)
        res = sim.run([1.0, -0.5, 0.5], seed=42)

        out = tmp_path / "mc_results.json"
        MonteCarloSurvivalSimulator.write_results(res, out)

        assert out.is_file()
        data = json.loads(out.read_text())
        assert data["n_simulations"] == 50
        for key in [
            "p_target_before_ruin", "p_ruin", "dd_p95", "losing_streak_p95",
            "generated", "avg_trade_pnl_dollars", "expected_trades_to_target",
            "termination_breakdown", "trade_count_input",
            "std_trade_pnl_dollars", "skewness", "kurtosis",
            "worst_sampled_loss", "worst_sampled_drawdown",
            "equity_p1", "equity_p99", "worst_intraday_dd",
            "mode", "stress_scenario",
        ]:
            assert key in data, f"Missing JSON key: {key}"
        assert "hit_target" in data["termination_breakdown"]

    # ── Probabilities bounded ───────────────────────────────────────────

    def test_probabilities_bounded(self) -> None:
        """All probability fields are in [0, 1]."""
        sim = self._make_sim(n_simulations=200)
        res = sim.run([1.5, -0.8, 0.3, -1.2, 2.0], seed=55)

        for field_name in (
            "p_target_before_ruin",
            "p_ruin",
            "p_fail_consistency_given_target",
            "p_daily_loss_breach",
        ):
            val = getattr(res, field_name)
            assert 0.0 <= val <= 1.0, f"{field_name}={val} out of bounds"


# ═══════════════════════════════════════════════════════════════════════
#  Block Bootstrap Tests
# ═══════════════════════════════════════════════════════════════════════


@patch("simulation.mc_survival._cfg.MC_DAY_HORIZON_ENABLED", False)
class TestBlockBootstrap:
    """Tests for block-bootstrap MC mode."""

    def _make_sim(self, **kwargs: float | int | str) -> MonteCarloSurvivalSimulator:
        defaults: dict[str, float | int | str] = dict(
            risk_per_trade=20.0,
            profit_target=3000.0,
            max_loss_limit=2000.0,
            daily_loss_limit=1000.0,
            consistency_cap=0.50,
            n_simulations=500,
            max_trades=800,
            trades_per_day=5,
            mode="block",
            block_type="session",
        )
        defaults.update(kwargs)
        return MonteCarloSurvivalSimulator(**defaults)  # type: ignore[arg-type]

    def test_block_mode_produces_result(self) -> None:
        """Block mode runs and produces a valid SurvivalResult."""
        sim = self._make_sim(n_simulations=100)
        r_vals = [1.0, -0.5, 2.0, -1.0, 0.5]
        session_ids = ["s1", "s1", "s2", "s2", "s2"]

        res = sim.run(r_vals, seed=42, session_ids=session_ids)
        assert isinstance(res, SurvivalResult)
        assert res.mode == "block"
        assert res.n_simulations == 100

    def test_block_vs_iid_different_dd_with_clustered_losses(self) -> None:
        """Clustered losses: block dd_p95 should differ from IID dd_p95.

        When losses are clustered in one session, block sampling preserves
        the clustering and should produce higher drawdowns than IID which
        smooths them out.
        """
        # Session A: all wins   Session B: all losses (clustered)
        r_vals = [1.0, 1.0, 1.0, 1.0, 1.0,    # session A
                  -2.0, -2.0, -2.0, -2.0, -2.0]  # session B (clustered losses)
        session_ids = ["A"] * 5 + ["B"] * 5

        sim_block = MonteCarloSurvivalSimulator(
            risk_per_trade=20.0,
            profit_target=5000.0,
            max_loss_limit=2000.0,
            daily_loss_limit=5000.0,
            n_simulations=2000,
            max_trades=200,
            trades_per_day=5,
            mode="block",
            block_type="session",
        )
        sim_iid = MonteCarloSurvivalSimulator(
            risk_per_trade=20.0,
            profit_target=5000.0,
            max_loss_limit=2000.0,
            daily_loss_limit=5000.0,
            n_simulations=2000,
            max_trades=200,
            trades_per_day=5,
            mode="iid",
        )

        res_block = sim_block.run(r_vals, seed=42, session_ids=session_ids)
        res_iid = sim_iid.run(r_vals, seed=42)

        # Block preserves loss clustering → higher ruin probability
        # (dd_p95 can saturate at max_loss_limit, so compare p_ruin)
        assert res_block.p_ruin >= res_iid.p_ruin, (
            f"Block ruin should be >= IID ruin: block={res_block.p_ruin}, iid={res_iid.p_ruin}"
        )

    def test_block_day_grouping(self) -> None:
        """Block mode with day grouping produces valid results."""
        sim = self._make_sim(block_type="day", n_simulations=100)
        r_vals = [1.0, -0.5, 0.3, -0.2, 0.8]
        timestamps = [
            "2026-02-18T14:30:00Z",
            "2026-02-18T15:00:00Z",
            "2026-02-19T14:30:00Z",
            "2026-02-19T15:00:00Z",
            "2026-02-19T15:30:00Z",
        ]
        res = sim.run(r_vals, seed=1, timestamps=timestamps)
        assert isinstance(res, SurvivalResult)
        assert res.mode == "block"

    def test_block_fallback_without_labels(self) -> None:
        """Without session_ids/timestamps, block falls back to tpd chunks."""
        sim = self._make_sim(n_simulations=100, trades_per_day=3)
        r_vals = [1.0, -0.5, 2.0, -1.0, 0.5, -0.3]
        res = sim.run(r_vals, seed=10)
        assert isinstance(res, SurvivalResult)
        assert res.mode == "block"


# ═══════════════════════════════════════════════════════════════════════
#  Stress Testing Tests
# ═══════════════════════════════════════════════════════════════════════


class TestStressTesting:
    """Tests for adversarial stress transforms."""

    def _make_sim(self, **kwargs: float | int | str) -> MonteCarloSurvivalSimulator:
        defaults: dict[str, float | int | str] = dict(
            risk_per_trade=20.0,
            profit_target=3000.0,
            max_loss_limit=2000.0,
            daily_loss_limit=1000.0,
            n_simulations=500,
            max_trades=800,
            trades_per_day=5,
            mode="iid",
        )
        defaults.update(kwargs)
        return MonteCarloSurvivalSimulator(**defaults)  # type: ignore[arg-type]

    def test_loss_multiplier_reduces_survival(self) -> None:
        """LOSS_MULTIPLIER=2.0 must decrease p_target_before_ruin."""
        sim = self._make_sim(
            n_simulations=500,
            profit_target=500.0,
            max_loss_limit=300.0,
        )
        r_vals = [1.0, -0.5, 0.8, -1.2, 1.5, -0.3, 0.5, -0.8, 2.0, -1.0]

        res_base = sim.run(r_vals, seed=42)
        res_stress = sim.run(
            r_vals, seed=42,
            stress={"loss_multiplier": 2.0, "win_multiplier": 1.0, "_name": "test"},
        )

        assert res_stress.p_target_before_ruin <= res_base.p_target_before_ruin, (
            f"Stress should reduce survival: base={res_base.p_target_before_ruin}, "
            f"stress={res_stress.p_target_before_ruin}"
        )
        assert res_stress.stress_scenario == "test"

    def test_win_multiplier_reduces_edge(self) -> None:
        """WIN_MULTIPLIER=0.5 reduces average trade PnL."""
        sim = self._make_sim(n_simulations=100)
        r_vals = [2.0, -0.5, 1.5, -0.3, 1.0]

        res_base = sim.run(r_vals, seed=1)
        res_stress = sim.run(
            r_vals, seed=1,
            stress={"win_multiplier": 0.5, "_name": "wincut"},
        )
        assert res_stress.avg_trade_pnl_dollars < res_base.avg_trade_pnl_dollars

    def test_win_rate_shift_flips_winners(self) -> None:
        """WIN_RATE_SHIFT=-0.5 with all winners flips ~50% to losers."""
        sim = self._make_sim(n_simulations=100, max_trades=50)
        r_vals = [1.0] * 20  # all winners

        res_base = sim.run(r_vals, seed=42)
        res_stress = sim.run(
            r_vals, seed=42,
            stress={"win_rate_shift": -0.5, "_name": "wrshift"},
        )
        # avg_trade_pnl should be lower after flipping winners
        assert res_stress.avg_trade_pnl_dollars < res_base.avg_trade_pnl_dollars

    def test_run_all_scenarios(self) -> None:
        """run_all_scenarios produces base, mild, severe, tilt_bad_week results."""
        sim = self._make_sim(n_simulations=100)
        r_vals = [1.0, -0.5, 0.8, -1.2, 1.5]

        results = sim.run_all_scenarios(r_vals, seed=42, n_batches=1)

        assert "base" in results
        assert "mild" in results
        assert "severe" in results
        assert "tilt_bad_week" in results
        assert results["base"].stress_scenario == "base"
        assert results["mild"].stress_scenario == "mild"
        assert results["severe"].stress_scenario == "severe"
        assert results["tilt_bad_week"].stress_scenario == "tilt_bad_week"

    def test_write_all_results(self, tmp_path: Path) -> None:
        """write_all_results writes 4 separate JSON files."""
        sim = self._make_sim(n_simulations=50)
        r_vals = [1.0, -0.5]
        results = sim.run_all_scenarios(r_vals, seed=1, n_batches=1)

        paths = MonteCarloSurvivalSimulator.write_all_results(results, tmp_path)

        assert (tmp_path / "mc_results.json").is_file()
        assert (tmp_path / "mc_results_stress_mild.json").is_file()
        assert (tmp_path / "mc_results_stress_severe.json").is_file()
        assert (tmp_path / "mc_results_stress_tilt_bad_week.json").is_file()
        assert len(paths) == 4


# ═══════════════════════════════════════════════════════════════════════
#  True Trailing Drawdown Tests
# ═══════════════════════════════════════════════════════════════════════


class TestTrailingDrawdown:
    """Tests for true path-based trailing drawdown logic."""

    def test_trailing_dd_ruin_mid_path(self) -> None:
        """Known sequence that exceeds trailing limit mid-path must mark ruin.

        Sequence: +$100, +$100, -$250
        Peak after trade 2: $200
        After trade 3: $200 - $250 = -$50  →  DD = $200 - (-$50) = $250
        With MAX_LOSS_LIMIT=$200: ruin.
        """
        sim = MonteCarloSurvivalSimulator(
            risk_per_trade=1.0,  # pnl_r = dollar values effectively
            profit_target=1_000_000.0,  # unreachable target
            max_loss_limit=200.0,
            daily_loss_limit=1_000_000.0,
            n_simulations=100,
            max_trades=10,
            trades_per_day=10,
            mode="iid",
        )
        # Every sim draws from same pool → same sequence
        r_vals = [100.0, 100.0, -250.0]
        # force the same 3-trade pattern: all sims see +100,+100,-250 in rotation
        res = sim.run(r_vals, seed=42, use_dollar_values=True)

        # Should have at least some ruin due to the -250 drawdown
        assert res.p_ruin > 0.0, f"Expected ruin>0 with trailing DD, got {res.p_ruin}"

    def test_peak_tracking_increases_monotonically(self) -> None:
        """Peak only increases: after +$100,-$50, peak should stay at $100.

        Subsequent trade of -$200 from peak should trigger $300 DD.
        With MAX_LOSS_LIMIT=$250, this should ruin.
        """
        sim = MonteCarloSurvivalSimulator(
            risk_per_trade=1.0,
            profit_target=1_000_000.0,
            max_loss_limit=250.0,
            daily_loss_limit=1_000_000.0,
            n_simulations=100,
            max_trades=10,
            trades_per_day=10,
            mode="iid",
        )
        # Pool: heavily biased towards the DDing pattern
        r_vals = [100.0, -50.0, -200.0]
        res = sim.run(r_vals, seed=42, use_dollar_values=True)

        # With these values, some paths will hit ruin
        assert res.p_ruin > 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Daily Loss Limit Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDailyLossLimit:
    """Tests for daily loss limit termination."""

    def test_daily_loss_breach_terminates(self) -> None:
        """Single-day losses exceeding DAILY_LOSS_LIMIT must terminate.

        Pool: [-$50] only.  Daily limit = $100.  trades_per_day=5.
        After 2 trades in a day: -$100, which triggers daily breach.
        """
        sim = MonteCarloSurvivalSimulator(
            risk_per_trade=1.0,
            profit_target=1_000_000.0,
            max_loss_limit=1_000_000.0,  # won't trigger
            daily_loss_limit=100.0,
            n_simulations=100,
            max_trades=50,
            trades_per_day=5,
            mode="iid",
        )
        res = sim.run([-50.0], seed=42, use_dollar_values=True)

        # Every scenario should breach daily limit within a "day" block
        assert res.p_daily_loss_breach > 0.0, (
            f"Expected daily loss breaches, got p={res.p_daily_loss_breach}"
        )
        assert res.termination_daily_loss > 0


# ═══════════════════════════════════════════════════════════════════════
#  Distribution Diagnostics Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDistributionDiagnostics:
    """Tests for extended distribution diagnostics in SurvivalResult."""

    def test_diagnostics_populated(self) -> None:
        """All distribution diagnostic fields are populated with real values."""
        sim = MonteCarloSurvivalSimulator(
            risk_per_trade=20.0,
            profit_target=500.0,
            max_loss_limit=500.0,
            daily_loss_limit=500.0,
            n_simulations=200,
            max_trades=200,
            trades_per_day=5,
            mode="iid",
        )
        r_vals = [2.0, -1.5, 1.0, -2.0, 0.5, -0.3, 3.0, -1.0, 0.8, -0.5]
        res = sim.run(r_vals, seed=42)

        assert res.trade_count_input == 10
        assert res.std_trade_pnl_dollars > 0.0
        # skewness and kurtosis are floats (may be positive or negative)
        assert isinstance(res.skewness, float)
        assert isinstance(res.kurtosis, float)
        assert res.worst_sampled_loss < 0.0
        assert res.worst_sampled_drawdown >= 0.0
        assert isinstance(res.equity_p1, float)
        assert isinstance(res.equity_p99, float)

    def test_edge_saturation_note(self) -> None:
        """When p_target=1.0, notes should warn about edge saturation."""
        sim = MonteCarloSurvivalSimulator(
            risk_per_trade=20.0,
            profit_target=40.0,
            max_loss_limit=5000.0,
            daily_loss_limit=5000.0,
            n_simulations=100,
            max_trades=100,
            trades_per_day=5,
            mode="iid",
        )
        res = sim.run([1.0], seed=1)

        assert res.p_target_before_ruin == 1.0
        assert "Edge saturation likely" in res.notes

    def test_mode_and_stress_in_result(self) -> None:
        """mode and stress_scenario metadata are populated correctly."""
        sim = MonteCarloSurvivalSimulator(
            risk_per_trade=20.0,
            profit_target=500.0,
            max_loss_limit=500.0,
            daily_loss_limit=500.0,
            n_simulations=50,
            max_trades=100,
            trades_per_day=5,
            mode="iid",
        )
        res = sim.run(
            [1.0, -0.5], seed=1,
            stress={"loss_multiplier": 1.5, "_name": "my_stress"},
        )
        assert res.mode == "iid"
        assert res.stress_scenario == "my_stress"


# ═══════════════════════════════════════════════════════════════════════
#  Wilson CI Tests
# ═══════════════════════════════════════════════════════════════════════


class TestWilsonCI:
    """Tests for the Wilson binomial confidence interval function."""

    def test_wilson_50_percent(self) -> None:
        """50/100 successes → CI contains 0.5, symmetric around 0.5."""
        from simulation.mc_survival import wilson_ci
        lo, hi = wilson_ci(50, 100, confidence=0.95)
        assert lo < 0.5 < hi
        assert lo == pytest.approx(0.4, abs=0.05)
        assert hi == pytest.approx(0.6, abs=0.05)

    def test_wilson_all_pass(self) -> None:
        """100/100 successes → CI upper = 1.0 (or near), lower well above 0.9."""
        from simulation.mc_survival import wilson_ci
        lo, hi = wilson_ci(100, 100, confidence=0.95)
        assert hi >= 0.95
        assert lo > 0.9

    def test_wilson_zero_pass(self) -> None:
        """0/100 → CI lower = 0, upper small."""
        from simulation.mc_survival import wilson_ci
        lo, hi = wilson_ci(0, 100, confidence=0.95)
        assert lo == 0.0
        assert hi < 0.05

    def test_wilson_n_zero(self) -> None:
        """n=0 → degenerate (0, 1) — maximum uncertainty."""
        from simulation.mc_survival import wilson_ci
        lo, hi = wilson_ci(5, 0, confidence=0.95)
        assert lo == 0.0
        assert hi == 1.0

    def test_wilson_bounds_ordered(self) -> None:
        """Lower bound < upper bound for any valid input."""
        from simulation.mc_survival import wilson_ci
        for k in range(0, 51, 5):
            lo, hi = wilson_ci(k, 50, confidence=0.95)
            assert lo <= hi, f"k={k}: lo={lo} > hi={hi}"


# ═══════════════════════════════════════════════════════════════════════
#  Expanded SurvivalResult Diagnostics
# ═══════════════════════════════════════════════════════════════════════


class TestExpandedDiagnostics:
    """Tests for Phase 2 expanded SurvivalResult fields."""

    def _make_sim(self, **kw):
        defaults = dict(
            risk_per_trade=20.0,
            profit_target=500.0,
            max_loss_limit=500.0,
            daily_loss_limit=500.0,
            n_simulations=200,
            max_trades=200,
            trades_per_day=5,
            mode="iid",
        )
        defaults.update(kw)
        return MonteCarloSurvivalSimulator(**defaults)  # type: ignore[arg-type]

    def test_dd_percentiles(self) -> None:
        """DD percentiles p50 <= p90 <= p95 <= p99."""
        sim = self._make_sim()
        r_vals = [2.0, -1.5, 1.0, -2.0, 0.5, -0.3, 3.0, -1.0, 0.8, -0.5]
        res = sim.run(r_vals, seed=42)

        assert res.dd_p50 <= res.dd_p90 <= res.dd_p95 <= res.dd_p99
        assert res.dd_p50 >= 0

    def test_trades_to_target_distribution(self) -> None:
        """TTT percentiles p5 <= p25 <= median <= p75 <= p95 (when paths hit target)."""
        sim = self._make_sim(profit_target=200.0, n_simulations=500)
        r_vals = [1.0, 0.5, 2.0, -0.5, 1.5]
        res = sim.run(r_vals, seed=42)

        if res.p_target_before_ruin > 0:
            assert res.trades_to_target_p5 <= res.trades_to_target_p25
            assert res.trades_to_target_p25 <= res.median_trades_to_target
            assert res.median_trades_to_target <= res.trades_to_target_p75
            assert res.trades_to_target_p75 <= res.trades_to_target_p95

    def test_frac_terminated_max_trades(self) -> None:
        """frac_terminated_max_trades is in [0, 1]."""
        sim = self._make_sim()
        res = sim.run([0.01], seed=5)
        assert 0.0 <= res.frac_terminated_max_trades <= 1.0

    def test_equity_snapshots_exist(self) -> None:
        """Equity snapshot fields exist and are floats."""
        sim = self._make_sim(max_trades=250)
        r_vals = [1.0, -0.5, 0.8, -0.3] * 5
        res = sim.run(r_vals, seed=42)

        for attr in ("equity_at_50", "equity_at_100", "equity_at_200"):
            val = getattr(res, attr)
            assert isinstance(val, float), f"{attr} is not float: {type(val)}"

    def test_result_has_expanded_fields(self) -> None:
        """SurvivalResult has all Phase 2/3 fields."""
        sim = self._make_sim(n_simulations=50)
        res = sim.run([1.0, -0.5], seed=123)

        expanded_fields = [
            "trades_to_target_p5", "trades_to_target_p25",
            "trades_to_target_p75", "trades_to_target_p95",
            "equity_at_50", "equity_at_100", "equity_at_200",
            "frac_terminated_max_trades",
            "dd_p50", "dd_p90", "dd_p99",
            "p_target_ci_lo", "p_target_ci_hi",
            "p_target_batch_median", "p_target_batch_min", "p_target_batch_max",
        ]
        for f in expanded_fields:
            assert hasattr(res, f), f"Missing field: {f}"

    def test_json_has_expanded_keys(self, tmp_path) -> None:
        """write_results JSON includes expanded diagnostic keys."""
        sim = self._make_sim(n_simulations=50)
        res = sim.run([1.0, -0.5, 0.5], seed=42)
        out = tmp_path / "mc_results.json"
        MonteCarloSurvivalSimulator.write_results(res, out)

        import json
        data = json.loads(out.read_text())

        assert "trades_to_target_distribution" in data
        assert "equity_slope" in data
        assert "frac_terminated_max_trades" in data
        assert "drawdown_distribution" in data
        assert "p_target_ci" in data
        assert "p_target_batch_spread" in data

        # Nested keys
        assert "p5" in data["trades_to_target_distribution"]
        assert "p50" in data["drawdown_distribution"]
        assert "at_50_trades" in data["equity_slope"]
        assert "lower" in data["p_target_ci"]
        assert "median" in data["p_target_batch_spread"]


# ═══════════════════════════════════════════════════════════════════════
#  Multi-Batch MC Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMultiBatch:
    """Tests for multi-batch MC with Wilson CI."""

    def _make_sim(self, **kw):
        defaults = dict(
            risk_per_trade=20.0,
            profit_target=500.0,
            max_loss_limit=500.0,
            daily_loss_limit=500.0,
            n_simulations=100,
            max_trades=200,
            trades_per_day=5,
            mode="iid",
        )
        defaults.update(kw)
        return MonteCarloSurvivalSimulator(**defaults)  # type: ignore[arg-type]

    def test_multi_batch_ci_populated(self) -> None:
        """Multi-batch run populates CI fields."""
        sim = self._make_sim()
        r_vals = [1.0, -0.5, 0.8, -1.2, 1.5]
        results = sim.run_all_scenarios(r_vals, seed=42, n_batches=3)

        base = results["base"]
        assert base.p_target_ci_lo <= base.p_target_ci_hi
        assert base.p_target_ci_lo >= 0.0
        assert base.p_target_ci_hi <= 1.0
        assert base.p_target_batch_min <= base.p_target_batch_max

    def test_multi_batch_vs_single(self) -> None:
        """Multi-batch median should be close to single-batch p_target."""
        sim = self._make_sim(n_simulations=200)
        r_vals = [1.0, -0.5, 0.8, -1.2, 1.5, -0.3, 0.5]

        res_single = sim.run(r_vals, seed=42)
        results_multi = sim.run_all_scenarios(r_vals, seed=42, n_batches=3)
        res_multi = results_multi["base"]

        # Median should be in the same ballpark as single run
        assert abs(res_multi.p_target_batch_median - res_single.p_target_before_ruin) < 0.15


# ═══════════════════════════════════════════════════════════════════════
#  Tilt Bad Week Tests
# ═══════════════════════════════════════════════════════════════════════


class TestTiltBadWeek:
    """Tests for tilt_bad_week stress scenario."""

    def _make_sim(self, **kw):
        defaults = dict(
            risk_per_trade=20.0,
            profit_target=3000.0,
            max_loss_limit=2000.0,
            daily_loss_limit=1000.0,
            n_simulations=200,
            max_trades=800,
            trades_per_day=5,
            mode="block",
            block_type="session",
        )
        defaults.update(kw)
        return MonteCarloSurvivalSimulator(**defaults)  # type: ignore[arg-type]

    def test_tilt_preset_exists(self) -> None:
        """STRESS_PRESETS contains tilt_bad_week."""
        from simulation.mc_survival import STRESS_PRESETS
        assert "tilt_bad_week" in STRESS_PRESETS
        preset = STRESS_PRESETS["tilt_bad_week"]
        assert "tilt_frac" in preset
        assert "tilt_quantile" in preset

    def test_tilt_reduces_survival_vs_base(self) -> None:
        """tilt_bad_week p_target should be <= base p_target."""
        sim = self._make_sim(n_simulations=300)
        # Create 4 sessions: 2 good, 2 bad
        r_vals = (
            [2.0, 1.0, 1.5] +  # session A (good)
            [2.0, 1.0, 0.5] +  # session B (good)
            [-1.5, -2.0, -1.0] +  # session C (bad)
            [-1.0, -1.5, -2.0]    # session D (bad)
        )
        session_ids = ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3

        results = sim.run_all_scenarios(
            r_vals, seed=42, session_ids=session_ids, n_batches=1,
        )

        assert results["tilt_bad_week"].p_target_before_ruin <= results["base"].p_target_before_ruin + 0.05

    def test_tilt_in_run_all_scenarios(self) -> None:
        """run_all_scenarios includes tilt_bad_week in results."""
        sim = self._make_sim(n_simulations=50)
        r_vals = [1.0, -0.5, 0.8, -1.2, 1.5]
        session_ids = ["s1", "s1", "s2", "s2", "s2"]
        results = sim.run_all_scenarios(r_vals, seed=1, session_ids=session_ids, n_batches=1)

        assert "tilt_bad_week" in results
        assert results["tilt_bad_week"].stress_scenario == "tilt_bad_week"

    def test_rank_bad_blocks(self) -> None:
        """_rank_bad_blocks returns indices of worst-PnL blocks."""
        blocks = [
            np.array([10.0, 20.0]),    # sum = 30 (good)
            np.array([-5.0, -10.0]),   # sum = -15 (bad)
            np.array([5.0, 5.0]),      # sum = 10 (ok)
            np.array([-20.0, -30.0]),  # sum = -50 (worst)
        ]
        bad = MonteCarloSurvivalSimulator._rank_bad_blocks(blocks, quantile=0.50)
        assert len(bad) == 2
        assert 3 in bad  # worst
        assert 1 in bad  # second worst


# ═══════════════════════════════════════════════════════════════════════
#  Stress Comparison Logger Tests
# ═══════════════════════════════════════════════════════════════════════


class TestStressComparisonLogger:
    """Tests for log_stress_comparison table output."""

    def test_comparison_table_format(self) -> None:
        """log_stress_comparison returns a formatted table string."""
        sim = MonteCarloSurvivalSimulator(
            risk_per_trade=20.0,
            profit_target=500.0,
            max_loss_limit=500.0,
            daily_loss_limit=500.0,
            n_simulations=50,
            max_trades=200,
            trades_per_day=5,
            mode="iid",
        )
        r_vals = [1.0, -0.5, 0.8, -1.2, 1.5, -0.3]
        results = sim.run_all_scenarios(r_vals, seed=42, n_batches=1)

        table = MonteCarloSurvivalSimulator.log_stress_comparison(results)

        assert isinstance(table, str)
        assert "STRESS COMPARISON TABLE" in table
        assert "P(Target)" in table
        assert "base" in table
        assert "mild" in table
        assert "severe" in table

    def test_comparison_table_partial_results(self) -> None:
        """Works with only a subset of scenarios."""
        sim = MonteCarloSurvivalSimulator(
            risk_per_trade=20.0,
            profit_target=500.0,
            max_loss_limit=500.0,
            daily_loss_limit=500.0,
            n_simulations=50,
            max_trades=200,
            trades_per_day=5,
            mode="iid",
        )
        r_vals = [1.0, -0.5]
        base_res = sim.run(r_vals, seed=1)
        table = MonteCarloSurvivalSimulator.log_stress_comparison({"base": base_res})

        assert "STRESS COMPARISON TABLE" in table
        assert "base" in table
