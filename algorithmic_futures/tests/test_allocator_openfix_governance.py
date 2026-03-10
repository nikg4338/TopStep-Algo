"""Tests for allocator-openfix governance verdict semantics."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.candidate_openfix import evaluate_candidate_verdict


def _candidate_summary(**overrides):
    summary = {
        "orb_routed_sessions": 10,
        "allocator_rows": [{} for _ in range(5)],
        "sessions_total": 5,
        "ruin_probability": 0.01,
        "dd_p95": 900.0,
        "daily_loss_breach_count": 0,
        "consistency_rule_breach_count": 0,
        "target_probability": 0.70,
    }
    summary.update(overrides)
    return summary


def _live_checks(**overrides):
    checks = {
        "no_lookahead_evidence": True,
        "eod_flatten": True,
        "daily_loss_halt": True,
        "daily_profit_halt": True,
        "no_duplicate_orders": True,
        "no_stale_orders": True,
    }
    checks.update(overrides)
    return checks


def test_low_ptarget_cannot_yield_promotion_pass():
    verdict = evaluate_candidate_verdict(
        _candidate_summary(target_probability=0.20),
        target_threshold=0.60,
        live_integrity=_live_checks(),
        robustness_rows=[{"p_ruin": 0.01, "dd_p95": 950.0}],
    )
    assert verdict.engineering_verdict == "PASS"
    assert verdict.promotion_verdict == "HOLD"
    assert "target-hit probability" in verdict.reason


def test_good_integrity_but_weak_ptarget_yields_hold():
    verdict = evaluate_candidate_verdict(
        _candidate_summary(target_probability=0.40),
        target_threshold=0.60,
        live_integrity=_live_checks(),
        robustness_rows=[{"p_ruin": 0.01, "dd_p95": 950.0}],
    )
    assert verdict.engineering_verdict == "PASS"
    assert verdict.promotion_verdict == "HOLD"


def test_strong_candidate_can_yield_promotion_pass():
    verdict = evaluate_candidate_verdict(
        _candidate_summary(target_probability=0.75),
        target_threshold=0.60,
        live_integrity=_live_checks(),
        robustness_rows=[{"p_ruin": 0.01, "dd_p95": 950.0}],
    )
    assert verdict.engineering_verdict == "PASS"
    assert verdict.promotion_verdict == "PASS"


def test_hard_integrity_failure_yields_fail():
    verdict = evaluate_candidate_verdict(
        _candidate_summary(dd_p95=1500.0, target_probability=0.90),
        target_threshold=0.60,
        live_integrity=_live_checks(eod_flatten=False),
        robustness_rows=[{"p_ruin": 0.01, "dd_p95": 1500.0}],
    )
    assert verdict.engineering_verdict == "FAIL"
    assert verdict.promotion_verdict == "FAIL"
