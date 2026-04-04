from __future__ import annotations

import argparse

from run_v3_sizing_calibration import (
    _make_sizing_config,
    build_promotion_ranking,
    build_tightening_ranking,
    resolve_sizing_arms,
)


def _make_args(**overrides: object) -> argparse.Namespace:
    base = {
        "arms": None,
        "dynamic_v3_tractions": None,
        "dynamic_v3_giveback": 25.0,
        "fixed_arms": ["fixed_1c", "fixed_2c"],
        "include_orb_start_arm": False,
        "orb_start_traction": 75.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_sizing_arms_returns_explicit_override() -> None:
    args = _make_args(arms=["fixed_1c", "dynamic_v3_60_25"])
    assert resolve_sizing_arms(args) == ["fixed_1c", "dynamic_v3_60_25"]


def test_resolve_sizing_arms_builds_requested_traction_sweep() -> None:
    args = _make_args(dynamic_v3_tractions=[40.0, 50.0, 60.0, 75.0, 100.0])
    assert resolve_sizing_arms(args) == [
        "fixed_1c",
        "fixed_2c",
        "dynamic_v3_40_25",
        "dynamic_v3_50_25",
        "dynamic_v3_60_25",
        "dynamic_v3_75_25",
        "dynamic_v3_100_25",
    ]


def test_resolve_sizing_arms_can_append_generated_orb_start_arm() -> None:
    args = _make_args(
        dynamic_v3_tractions=[50.0, 75.0],
        include_orb_start_arm=True,
        orb_start_traction=60.0,
    )
    assert resolve_sizing_arms(args) == [
        "fixed_1c",
        "fixed_2c",
        "dynamic_v3_50_25",
        "dynamic_v3_75_25",
        "dynamic_v3_60_25_orb_start",
    ]


def test_make_sizing_config_parses_generated_dynamic_v3_arm() -> None:
    cfg = _make_sizing_config("dynamic_v3_60_25")
    assert cfg.policy == "dynamic_v3"
    assert cfg.v3_earned_traction == 60.0
    assert cfg.v3_giveback_floor == 25.0
    assert cfg.v3_orb_upsize_allowed is False


def test_make_sizing_config_parses_generated_orb_start_arm() -> None:
    cfg = _make_sizing_config("dynamic_v3_60_25_orb_start")
    assert cfg.policy == "dynamic_v3"
    assert cfg.v3_earned_traction == 60.0
    assert cfg.v3_giveback_floor == 25.0
    assert cfg.v3_orb_upsize_allowed is True


def test_build_promotion_ranking_prefers_gate_passing_arm() -> None:
    aggregate = {
        "fixed_1c": {
            "p_target_before_ruin": {"mean": 0.55},
            "p_ruin": {"mean": 0.08},
            "dd_p95": {"mean": 900.0},
            "losing_streak_p95": {"mean": 5.0},
            "equity_p50": {"mean": 1400.0},
            "trade_count": {"mean": 40.0},
        },
        "dynamic_v3_60_25": {
            "p_target_before_ruin": {"mean": 0.63},
            "p_ruin": {"mean": 0.09},
            "dd_p95": {"mean": 950.0},
            "losing_streak_p95": {"mean": 5.5},
            "equity_p50": {"mean": 1500.0},
            "trade_count": {"mean": 42.0},
        },
    }

    ranking = build_promotion_ranking(aggregate)

    assert ranking[0]["arm"] == "dynamic_v3_60_25"
    assert ranking[0]["promotion_pass"] is True
    assert ranking[1]["arm"] == "fixed_1c"
    assert "mc_target_prob" in ranking[1]["failed_checks"]


def test_build_tightening_ranking_prioritizes_promotion_progress_without_risk_regression() -> None:
    aggregate = {
        "dynamic_v3_75_25": {
            "p_target_before_ruin": {"mean": 0.20},
            "p_ruin": {"mean": 0.08},
            "dd_p95": {"mean": 1250.0},
            "losing_streak_p95": {"mean": 8.5},
            "equity_p50": {"mean": 1500.0},
            "trade_count": {"mean": 42.0},
        },
        "dynamic_v3_60_25": {
            "p_target_before_ruin": {"mean": 0.24},
            "p_ruin": {"mean": 0.07},
            "dd_p95": {"mean": 1180.0},
            "losing_streak_p95": {"mean": 7.0},
            "equity_p50": {"mean": 1580.0},
            "trade_count": {"mean": 43.0},
        },
        "dynamic_v3_50_25": {
            "p_target_before_ruin": {"mean": 0.26},
            "p_ruin": {"mean": 0.09},
            "dd_p95": {"mean": 1275.0},
            "losing_streak_p95": {"mean": 8.0},
            "equity_p50": {"mean": 1600.0},
            "trade_count": {"mean": 45.0},
        },
    }

    ranking = build_tightening_ranking(aggregate, "dynamic_v3_75_25")

    assert ranking[0]["arm"] == "dynamic_v3_60_25"
    assert ranking[0]["classification"] == "promotion_progress"
    assert ranking[0]["clears_dd_gate"] is True
    assert ranking[0]["clears_losing_streak_gate"] is True
    assert ranking[1]["arm"] == "dynamic_v3_50_25"
    assert ranking[1]["classification"] == "risk_regression"


def test_build_tightening_ranking_raises_for_missing_reference_arm() -> None:
    aggregate = {
        "fixed_1c": {
            "p_target_before_ruin": {"mean": 0.10},
            "p_ruin": {"mean": 0.05},
            "dd_p95": {"mean": 900.0},
            "losing_streak_p95": {"mean": 5.0},
            "equity_p50": {"mean": 1200.0},
            "trade_count": {"mean": 30.0},
        }
    }

    try:
        build_tightening_ranking(aggregate, "dynamic_v3_75_25")
    except ValueError as exc:
        assert "reference arm" in str(exc)
    else:
        raise AssertionError("expected missing reference arm to raise ValueError")