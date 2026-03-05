"""
tests/test_promotion_gate.py — Tests for the promotion gate module.

Covers all-pass, session failure, low approval rate, summary output,
regression detection, and missing-scorecard degradation.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path

import pytest

from validation.promotion_gate import (
    GateThresholds,
    GateCheck,
    PromotionGate,
    PromotionGateResult,
    run_promotion_gate,
)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_run_dir(
    tmp_path,
    *,
    name: str = "run_current",
    manifest: dict | None = None,
    scorecard: dict | None = None,
    mc_profile: dict | None = None,
    aggregate_metrics: dict | None = None,
    mc_profile_root: dict | None = None,
    mc_results: dict | None = None,
    mc_results_mild: dict | None = None,
    mc_results_severe: dict | None = None,
):
    """Build a minimal run directory with optional artifact JSON files.

    Parameters
    ----------
    manifest : dict
        Written as ``manifest.json`` at root.
    scorecard : dict
        Written as ``scorecard/aggregate_metrics.json``.
    mc_profile : dict
        Written as ``mc_profile/replay_derived_profile.json`` (legacy).
    aggregate_metrics : dict
        Written as ``aggregate_metrics.json`` at run root (new).
    mc_profile_root : dict
        Written as ``mc_profile.json`` at run root (new).
    mc_results : dict
        Written as ``mc_results.json`` at run root (new).
    mc_results_mild : dict
        Written as ``mc_results_stress_mild.json`` at run root.
    mc_results_severe : dict
        Written as ``mc_results_stress_severe.json`` at run root.
    """
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)

    if manifest is not None:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    if scorecard is not None:
        sc_dir = run_dir / "scorecard"
        sc_dir.mkdir(parents=True, exist_ok=True)
        (sc_dir / "aggregate_metrics.json").write_text(
            json.dumps(scorecard), encoding="utf-8"
        )

    if mc_profile is not None:
        mc_dir = run_dir / "mc_profile"
        mc_dir.mkdir(parents=True, exist_ok=True)
        (mc_dir / "replay_derived_profile.json").write_text(
            json.dumps(mc_profile), encoding="utf-8"
        )

    if aggregate_metrics is not None:
        (run_dir / "aggregate_metrics.json").write_text(
            json.dumps(aggregate_metrics), encoding="utf-8"
        )

    if mc_profile_root is not None:
        (run_dir / "mc_profile.json").write_text(
            json.dumps(mc_profile_root), encoding="utf-8"
        )

    if mc_results is not None:
        (run_dir / "mc_results.json").write_text(
            json.dumps(mc_results), encoding="utf-8"
        )

    if mc_results_mild is not None:
        (run_dir / "mc_results_stress_mild.json").write_text(
            json.dumps(mc_results_mild), encoding="utf-8"
        )

    if mc_results_severe is not None:
        (run_dir / "mc_results_stress_severe.json").write_text(
            json.dumps(mc_results_severe), encoding="utf-8"
        )

    return run_dir


def _all_pass_manifest(n_sessions: int = 5) -> dict:
    """Return a manifest where every session has success=True."""
    return {
        "run_id": "test_run_001",
        "pack_id": "pack_a",
        "sessions": [
            {"session_id": f"s{i}", "success": True}
            for i in range(1, n_sessions + 1)
        ],
    }


def _good_scorecard() -> dict:
    return {
        "approval_rate": 0.75,
        "expectancy_r": 0.35,
    }


def _good_mc_profile() -> dict:
    import config
    return {
        "p95_max_drawdown": 600.0,
        "sample_size_trades": config.MC_PROFILE_MIN_TRADE_COUNT + 10
    }


def _good_aggregate_metrics() -> dict:
    """Return aggregate_metrics that pass all gate checks."""
    import config
    return {
        "trade_count_total": config.MC_PROFILE_MIN_TRADE_COUNT + 10,
        "win_rate": 0.55,
        "avg_r": 0.3,
        "expectancy_r": 0.3,
        "readiness": True,
    }


def _good_mc_profile_root() -> dict:
    """Return mc_profile.json content that passes gate checks."""
    return {
        "trade_count": 250,
        "win_rate": 0.55,
        "avg_r": 0.3,
    }


def _good_mc_results() -> dict:
    """Return mc_results.json content that passes all MC gate checks."""
    return {
        "n_simulations": 10000,
        "p_target_before_ruin": 0.75,   # > MC_TARGET_THRESHOLD (0.60)
        "p_ruin": 0.08,                 # < MC_RUIN_THRESHOLD (0.15)
        "dd_p95": 900.0,                # < PROMOTION_MC_MAX_DD_P95 (1200)
        "losing_streak_p95": 5.0,       # < MC_LOSING_STREAK_P95_MAX (8)
        "median_trades_to_target": 120.0,
        "p_daily_loss_breach": 0.02,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════


def test_gate_all_pass(tmp_path):
    """All sessions succeed, metrics within bounds → overall_pass=True."""
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(5),
        scorecard=_good_scorecard(),
        mc_profile=_good_mc_profile(),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    thresholds = GateThresholds(
        min_session_success_rate=0.9,
        min_approval_rate=0.5,
        max_approval_rate=0.95,
        min_expectancy_r=0.1,
        max_p95_drawdown=1000.0,
    )

    gate = PromotionGate(str(run_dir), thresholds=thresholds)
    result = gate.evaluate()

    assert isinstance(result, PromotionGateResult)
    assert result.overall_pass is True

    # Every individual check should pass
    for chk in result.checks:
        assert chk.passed, f"Check '{chk.name}' unexpectedly failed: {chk.notes}"

    assert result.run_id == "test_run_001"


def test_gate_session_failure(tmp_path):
    """Some sessions success=False, rate drops below threshold → fail."""
    manifest = {
        "run_id": "test_run_fail",
        "pack_id": "pack_b",
        "sessions": [
            {"session_id": "s1", "success": True},
            {"session_id": "s2", "success": False},
            {"session_id": "s3", "success": True},
            {"session_id": "s4", "success": False},
            {"session_id": "s5", "success": False},
        ],
    }
    run_dir = _make_run_dir(
        tmp_path,
        manifest=manifest,
        scorecard=_good_scorecard(),
        mc_profile=_good_mc_profile(),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    thresholds = GateThresholds(min_session_success_rate=0.9)
    gate = PromotionGate(str(run_dir), thresholds=thresholds)
    result = gate.evaluate()

    assert result.overall_pass is False

    session_check = next(c for c in result.checks if c.name == "session_success_rate")
    assert session_check.passed is False
    # 2 out of 5 = 0.4
    assert float(session_check.current_value) == pytest.approx(0.4, abs=1e-4)


def test_gate_low_approval_rate(tmp_path):
    """Scorecard approval_rate below min_approval_rate → fail."""
    # Put the low approval_rate in the aggregate_metrics (which the gate now prefers)
    low_ar_metrics = _good_aggregate_metrics()
    low_ar_metrics["approval_rate"] = 0.25
    low_ar_metrics["expectancy_r"] = 0.3

    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=low_ar_metrics,
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    thresholds = GateThresholds(
        min_session_success_rate=0.9,
        min_approval_rate=0.5,
    )
    gate = PromotionGate(str(run_dir), thresholds=thresholds)
    result = gate.evaluate()

    assert result.overall_pass is False

    ar_check = next(c for c in result.checks if c.name == "approval_rate")
    assert ar_check.passed is False
    assert float(ar_check.current_value) == pytest.approx(0.25, abs=1e-4)


def test_gate_insufficient_trades(tmp_path):
    """Gate fails if aggregate_metrics trade count is below threshold."""
    import config
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        scorecard=_good_scorecard(),
        mc_profile=_good_mc_profile(),
        aggregate_metrics={"trade_count_total": config.MC_PROFILE_MIN_TRADE_COUNT - 1, "readiness": False},
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    thresholds = GateThresholds(
        min_session_success_rate=0.9,
        min_approval_rate=0.5,
    )
    gate = PromotionGate(str(run_dir), thresholds=thresholds)
    result = gate.evaluate()

    assert result.overall_pass is False

    tc_check = next(c for c in result.checks if c.name == "min_trade_count")
    assert tc_check.passed is False
    assert tc_check.notes == "insufficient trades"

def test_writes_summary_md(tmp_path):
    """write_summary produces a Markdown file with expected headings."""
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        scorecard=_good_scorecard(),
        mc_profile=_good_mc_profile(),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()
    md_path = gate.write_summary(result)

    assert os.path.isfile(md_path)

    content = open(md_path, "r", encoding="utf-8").read()
    assert "# Promotion Gate Summary" in content
    assert "## Gate Checks" in content
    assert "## Notes" in content
    assert "PASS" in content or "FAIL" in content


def test_regression_detection(tmp_path):
    """Baseline has better approval_rate → regression flagged, overall_pass=False."""
    # Baseline run — higher approval_rate
    baseline_dir = _make_run_dir(
        tmp_path,
        name="run_baseline",
        manifest=_all_pass_manifest(3),
        scorecard={"approval_rate": 0.85, "expectancy_r": 0.5},
        mc_profile=_good_mc_profile(),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    # Current run — lower approval_rate (drop of 0.35 > tolerance 0.05)
    current_dir = _make_run_dir(
        tmp_path,
        name="run_current",
        manifest=_all_pass_manifest(3),
        scorecard={"approval_rate": 0.45, "expectancy_r": 0.5},
        mc_profile=_good_mc_profile(),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    thresholds = GateThresholds(
        min_session_success_rate=0.9,
        min_approval_rate=0.3,        # low enough that 0.45 passes the bounds check
        regression_tolerance=0.05,
    )
    gate = PromotionGate(
        str(current_dir),
        thresholds=thresholds,
        compare_to_dir=str(baseline_dir),
    )
    result = gate.evaluate()

    assert result.overall_pass is False

    reg_check = next(
        c for c in result.checks if c.name == "regression:approval_rate"
    )
    assert reg_check.passed is False
    assert "decreased" in reg_check.notes.lower() or ">" in reg_check.notes


def test_no_scorecard_still_works(tmp_path):
    """No scorecard directory → approval_rate and expectancy checks degrade gracefully.
    
    Note: aggregate_metrics.json at run root is used as the scorecard fallback.
    When aggregate_metrics doesn't include approval_rate, the check degrades.
    """
    # Provide aggregate_metrics WITHOUT approval_rate or expectancy_r
    agg_no_optional = {
        "trade_count_total": 20,
        "win_rate": 0.55,
        "avg_r": 0.3,
        "readiness": True,
    }

    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        # scorecard omitted
        aggregate_metrics=agg_no_optional,
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    thresholds = GateThresholds(
        min_session_success_rate=0.9,
        min_approval_rate=0.5,
        min_expectancy_r=0.1,
        max_p95_drawdown=1000.0,
    )
    gate = PromotionGate(str(run_dir), thresholds=thresholds)
    result = gate.evaluate()

    # Approval_rate not in aggregate_metrics → check skipped
    ar_check = next(c for c in result.checks if c.name == "approval_rate")
    assert ar_check.passed is True
    assert "skipped" in ar_check.notes.lower() or "not in" in ar_check.notes.lower()

    # expectancy_r not in aggregate_metrics → check skipped
    exp_check = next(c for c in result.checks if c.name == "expectancy_r")
    assert exp_check.passed is True
    assert "skipped" in exp_check.notes.lower() or "not in" in exp_check.notes.lower()

    # Overall should still pass because degraded checks pass
    assert result.overall_pass is True


# ═══════════════════════════════════════════════════════════════════════
#  New MC-results-driven gate checks
# ═══════════════════════════════════════════════════════════════════════


def test_gate_fail_mc_drawdown_too_high(tmp_path):
    """mc_results dd_p95 violates PROMOTION_MC_MAX_DD_P95 → gate fails."""
    bad_mc_results = _good_mc_results()
    bad_mc_results["dd_p95"] = 1500.0  # > 1200 threshold

    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=bad_mc_results,
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    assert result.overall_pass is False

    dd_check = next(c for c in result.checks if c.name == "mc_dd_p95")
    assert dd_check.passed is False
    assert float(dd_check.current_value) == pytest.approx(1500.0, abs=1)


def test_gate_fail_mc_target_prob_too_low(tmp_path):
    """mc_results p_target_before_ruin below MC_TARGET_THRESHOLD → fail."""
    bad_mc_results = _good_mc_results()
    bad_mc_results["p_target_before_ruin"] = 0.40  # < 0.60 threshold

    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=bad_mc_results,
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    assert result.overall_pass is False

    target_check = next(c for c in result.checks if c.name == "mc_target_prob")
    assert target_check.passed is False


def test_gate_fail_mc_ruin_prob_too_high(tmp_path):
    """mc_results p_ruin exceeds MC_RUIN_THRESHOLD → fail."""
    bad_mc_results = _good_mc_results()
    bad_mc_results["p_ruin"] = 0.25  # > 0.15 threshold

    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=bad_mc_results,
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    assert result.overall_pass is False

    ruin_check = next(c for c in result.checks if c.name == "mc_ruin_prob")
    assert ruin_check.passed is False


def test_gate_missing_mc_results_fails_required(tmp_path):
    """Missing mc_results.json → required_artifacts check fails."""
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        # mc_results deliberately omitted
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    assert result.overall_pass is False

    req_check = next(c for c in result.checks if c.name == "required_artifacts")
    assert req_check.passed is False
    assert "mc_results.json" in req_check.notes


def test_gate_writes_gate_result_json(tmp_path):
    """evaluate() writes gate_result.json at run root."""
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    gate_file = run_dir / "gate_result.json"
    assert gate_file.is_file()

    data = json.loads(gate_file.read_text())
    assert "overall_pass" in data
    assert "checks" in data
    assert isinstance(data["checks"], list)
    assert data["overall_pass"] == result.overall_pass


def test_gate_all_mc_checks_pass(tmp_path):
    """Synthetic mc_results that pass all thresholds → gate passes."""
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(5),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    assert result.overall_pass is True

    # Verify specific MC checks
    mc_checks = [c for c in result.checks if c.name.startswith("mc_")]
    for chk in mc_checks:
        assert chk.passed, f"MC check '{chk.name}' failed: {chk.notes}"


# ═══════════════════════════════════════════════════════════════════════
#  Stress scenario gate checks
# ═══════════════════════════════════════════════════════════════════════


def _good_mc_results_mild() -> dict:
    """mc_results_stress_mild.json that passes mild target threshold (>=0.50)."""
    return {
        "n_simulations": 10000,
        "p_target_before_ruin": 0.60,
        "p_ruin": 0.10,
        "dd_p95": 1000.0,
        "stress_scenario": "mild",
    }


def _good_mc_results_severe() -> dict:
    """mc_results_stress_severe.json that passes severe thresholds."""
    return {
        "n_simulations": 10000,
        "p_target_before_ruin": 0.40,   # >= 0.30
        "p_ruin": 0.12,                 # <= 0.15
        "dd_p95": 1100.0,
        "stress_scenario": "severe",
    }


def test_gate_stress_mild_target_pass(tmp_path):
    """Mild stress p_target above threshold → check passes."""
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
        mc_results_mild=_good_mc_results_mild(),
        mc_results_severe=_good_mc_results_severe(),
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    chk = next(c for c in result.checks if c.name == "stress_mild_target_prob")
    assert chk.passed is True


def test_gate_stress_mild_target_fail(tmp_path):
    """Mild stress p_target below 0.50 → check fails."""
    bad_mild = _good_mc_results_mild()
    bad_mild["p_target_before_ruin"] = 0.35  # < 0.50

    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
        mc_results_mild=bad_mild,
        mc_results_severe=_good_mc_results_severe(),
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    assert result.overall_pass is False

    chk = next(c for c in result.checks if c.name == "stress_mild_target_prob")
    assert chk.passed is False
    assert "median" in chk.notes.lower() or "<" in chk.notes


def test_gate_stress_severe_target_fail(tmp_path):
    """Severe stress p_target below 0.30 → check fails."""
    bad_severe = _good_mc_results_severe()
    bad_severe["p_target_before_ruin"] = 0.20  # < 0.30

    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
        mc_results_mild=_good_mc_results_mild(),
        mc_results_severe=bad_severe,
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    assert result.overall_pass is False

    chk = next(c for c in result.checks if c.name == "stress_severe_target_prob")
    assert chk.passed is False


def test_gate_stress_severe_ruin_fail(tmp_path):
    """Severe stress p_ruin above MC_RUIN_THRESHOLD → check fails."""
    bad_severe = _good_mc_results_severe()
    bad_severe["p_ruin"] = 0.25  # > 0.15

    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
        mc_results_mild=_good_mc_results_mild(),
        mc_results_severe=bad_severe,
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    assert result.overall_pass is False

    chk = next(c for c in result.checks if c.name == "stress_severe_ruin_prob")
    assert chk.passed is False
    assert "ruin" in chk.notes.lower()


def test_gate_stress_files_missing_skipped(tmp_path):
    """Missing stress files → stress checks pass (skipped gracefully)."""
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
        # stress files deliberately omitted
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    # All stress checks skipped → pass
    for name in ("stress_mild_target_prob", "stress_severe_target_prob",
                 "stress_severe_ruin_prob", "stress_tilt_target_prob"):
        chk = next(c for c in result.checks if c.name == name)
        assert chk.passed is True
        assert "skipped" in chk.notes.lower()


def test_gate_mc_trade_count_warning_low(tmp_path):
    """Trade count below VALIDATION_MIN_TRADES_FOR_MC → advisory warning (still passes)."""
    low_agg = _good_aggregate_metrics()
    low_agg["trade_count_total"] = 50  # well below 200

    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=low_agg,
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    chk = next(c for c in result.checks if c.name == "mc_trade_count_warning")
    assert chk.passed is True  # advisory — never fails
    assert "WARNING" in chk.notes or "warning" in chk.notes.lower()


def test_gate_mc_trade_count_sufficient(tmp_path):
    """Trade count above VALIDATION_MIN_TRADES_FOR_MC → no warning."""
    high_trade_agg = _good_aggregate_metrics()
    high_trade_agg["trade_count_total"] = 250  # above 200 threshold

    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=high_trade_agg,
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()

    chk = next(c for c in result.checks if c.name == "mc_trade_count_warning")
    assert chk.passed is True
    assert chk.notes == ""


# ═══════════════════════════════════════════════════════════════════════
#  CI-Aware Gate Tests
# ═══════════════════════════════════════════════════════════════════════


def _mc_results_with_ci(
    p_target: float = 0.60,
    ci_lo: float = 0.55,
    ci_hi: float = 0.65,
    batch_median: float = 0.60,
    batch_min: float = 0.55,
    batch_max: float = 0.65,
    scenario: str = "mild",
) -> dict:
    """Build an mc_results dict with CI and batch spread fields."""
    return {
        "n_simulations": 10000,
        "p_target_before_ruin": p_target,
        "p_ruin": 0.10,
        "dd_p95": 1000.0,
        "stress_scenario": scenario,
        "p_target_ci": {
            "lower": ci_lo,
            "upper": ci_hi,
        },
        "p_target_batch_spread": {
            "median": batch_median,
            "min": batch_min,
            "max": batch_max,
        },
    }


def test_gate_ci_mild_pass_both_tiers(tmp_path):
    """Mild stress CI: both tiers pass → gate passes."""
    mild = _mc_results_with_ci(
        p_target=0.65, ci_lo=0.55, ci_hi=0.75,
        batch_median=0.65, batch_min=0.60, batch_max=0.70,
        scenario="mild",
    )
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
        mc_results_mild=mild,
        mc_results_severe=_good_mc_results_severe(),
    )
    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()
    chk = next(c for c in result.checks if c.name == "stress_mild_target_prob")
    assert chk.passed is True
    assert "CI=" in chk.notes


def test_gate_ci_severe_floor_fail(tmp_path):
    """Severe stress CI lower < floor → gate fails even if median passes."""
    severe = _mc_results_with_ci(
        p_target=0.35, ci_lo=0.15, ci_hi=0.45,
        batch_median=0.35, batch_min=0.30, batch_max=0.40,
        scenario="severe",
    )
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
        mc_results_mild=_good_mc_results_mild(),
        mc_results_severe=severe,
    )
    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()
    chk = next(c for c in result.checks if c.name == "stress_severe_target_prob")
    assert chk.passed is False
    assert "CI lower" in chk.notes


def test_gate_ci_severe_median_fail(tmp_path):
    """Severe stress batch median < threshold → gate fails even if CI ok."""
    severe = _mc_results_with_ci(
        p_target=0.25, ci_lo=0.22, ci_hi=0.30,
        batch_median=0.25, batch_min=0.22, batch_max=0.30,
        scenario="severe",
    )
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
        mc_results_mild=_good_mc_results_mild(),
        mc_results_severe=severe,
    )
    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()
    chk = next(c for c in result.checks if c.name == "stress_severe_target_prob")
    assert chk.passed is False
    assert "median" in chk.notes


def test_gate_tilt_advisory_always_passes(tmp_path):
    """Tilt gate is advisory — always passes regardless of p_target."""
    tilt_data = {
        "n_simulations": 10000,
        "p_target_before_ruin": 0.10,  # terrible, but advisory
        "p_ruin": 0.50,
        "dd_p95": 1500.0,
        "stress_scenario": "tilt_bad_week",
        "p_target_ci": {"lower": 0.05, "upper": 0.15},
    }
    run_dir = _make_run_dir(
        tmp_path,
        manifest=_all_pass_manifest(3),
        aggregate_metrics=_good_aggregate_metrics(),
        mc_profile_root=_good_mc_profile_root(),
        mc_results=_good_mc_results(),
    )
    # Write tilt file manually
    (run_dir / "mc_results_stress_tilt_bad_week.json").write_text(
        json.dumps(tilt_data), encoding="utf-8"
    )

    gate = PromotionGate(str(run_dir))
    result = gate.evaluate()
    chk = next(c for c in result.checks if c.name == "stress_tilt_target_prob")
    assert chk.passed is True  # advisory
    assert "p_target=0.1000" in chk.notes
    assert "CI=" in chk.notes