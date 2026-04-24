from __future__ import annotations

from validation.open_proxy_route_calibration import (
    build_candidate_grid,
    build_route_tightening_ranking,
    derive_route_quality_metrics,
)


def test_build_candidate_grid_creates_expected_route_candidates() -> None:
    candidates = build_candidate_grid(
        persist_bars=[1, 2],
        low_atr_persistences=[2],
        high_impulse_persistences=[1],
        medium_impulse_min_atrs=[8.0],
        medium_impulse_max_atrs=[15.0],
        medium_impulse_mins=[0.9],
        medium_impulse_maxs=[1.8, 2.0],
        medium_impulse_min_persistences=[2],
    )

    assert [candidate.label for candidate in candidates] == [
        "op_p1_low2_hi1_m0p9-1p8_atr8-15_mp2",
        "op_p1_low2_hi1_m0p9-2_atr8-15_mp2",
        "op_p2_low2_hi1_m0p9-1p8_atr8-15_mp2",
        "op_p2_low2_hi1_m0p9-2_atr8-15_mp2",
    ]


def test_derive_route_quality_metrics_summarizes_orb_quality() -> None:
    summary = {
        "allocator_rows": [
            {"route": "orb", "session_pnl_dollars": 100.0, "confidence_score": 0.80},
            {"route": "orb", "session_pnl_dollars": -25.0, "confidence_score": 0.40},
            {"route": "mr", "session_pnl_dollars": 10.0, "confidence_score": 0.10},
        ],
        "final_equity": 50.0,
        "dd_p95": 1100.0,
        "ruin_probability": 0.10,
    }

    metrics = derive_route_quality_metrics(summary)

    assert metrics["orb_route_rate"] == 0.666667
    assert metrics["orb_win_rate"] == 0.5
    assert metrics["false_positive_orb_count"] == 1
    assert metrics["false_positive_orb_rate"] == 0.5
    assert metrics["avg_orb_session_pnl"] == 37.5
    assert metrics["route_quality_status"] == "watch"


def test_build_route_tightening_ranking_prefers_route_quality_progress() -> None:
    reference = {
        "label": "reference",
        "false_positive_orb_rate": 0.40,
        "dd_p95": 1300.0,
        "target_probability": 0.10,
        "ruin_probability": 0.20,
        "orb_win_rate": 0.45,
        "final_equity": 1000.0,
    }
    rows = [
        {
            "label": "narrow_band",
            "false_positive_orb_rate": 0.25,
            "dd_p95": 1200.0,
            "target_probability": 0.12,
            "ruin_probability": 0.18,
            "orb_win_rate": 0.60,
            "final_equity": 1100.0,
        },
        {
            "label": "wider_band",
            "false_positive_orb_rate": 0.50,
            "dd_p95": 1350.0,
            "target_probability": 0.09,
            "ruin_probability": 0.22,
            "orb_win_rate": 0.40,
            "final_equity": 900.0,
        },
    ]

    ranking = build_route_tightening_ranking(reference, rows)

    assert ranking[0]["label"] == "narrow_band"
    assert ranking[0]["classification"] == "route_quality_progress"
    assert ranking[1]["label"] == "wider_band"
    assert ranking[1]["classification"] == "route_regression"