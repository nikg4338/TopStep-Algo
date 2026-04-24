from __future__ import annotations

from validation.mr_edge_iteration import build_candidate_grid, build_edge_ranking


def test_build_candidate_grid_varies_sigma_and_soft_range_k() -> None:
    candidates = build_candidate_grid(
        sigma_entries=[1.25, 1.3],
        soft_range_impulse_ks=[1.0],
        cooldown_bars=[1],
        first_outside_modes=["on"],
        dedupe_modes=["on"],
        reclaim_mode="off",
        attempt_cap_mode="on",
        regime_mode="on",
    )

    assert [candidate.label for candidate in candidates] == [
        "mr_edge_sigma1p25_k1_fo1_ded1_cd1",
        "mr_edge_sigma1p3_k1_fo1_ded1_cd1",
    ]
    assert all(candidate.mr_first_outside_enabled for candidate in candidates)
    assert all(candidate.mr_dedupe_enabled for candidate in candidates)


def test_build_edge_ranking_prefers_edge_progress_over_risk_only() -> None:
    rows = [
        {
            "label": "mr_edge_sigma1p3_k1p2_fo1_ded1_cd1",
            "expectancy_r": 0.05,
            "p_target_before_ruin": 0.10,
            "p_ruin": 0.10,
            "dd_p95": 1200.0,
            "trade_count_total": 30,
            "failed_checks": [],
            "avg_r": 0.05,
        },
        {
            "label": "mr_edge_sigma1p25_k1p1_fo1_ded1_cd1",
            "expectancy_r": 0.08,
            "p_target_before_ruin": 0.14,
            "p_ruin": 0.08,
            "dd_p95": 1150.0,
            "trade_count_total": 32,
            "failed_checks": [],
            "avg_r": 0.08,
        },
        {
            "label": "mr_edge_sigma1p35_k1p3_fo1_ded1_cd1",
            "expectancy_r": 0.04,
            "p_target_before_ruin": 0.09,
            "p_ruin": 0.07,
            "dd_p95": 1100.0,
            "trade_count_total": 28,
            "failed_checks": [],
            "avg_r": 0.04,
        },
    ]

    reference, ranking = build_edge_ranking(
        rows,
        reference_label="mr_edge_sigma1p3_k1p2_fo1_ded1_cd1",
    )

    assert reference is not None
    assert reference["label"] == "mr_edge_sigma1p3_k1p2_fo1_ded1_cd1"
    assert ranking[0]["label"] == "mr_edge_sigma1p25_k1p1_fo1_ded1_cd1"
    assert ranking[0]["edge_classification"] == "edge_progress"
    assert ranking[1]["label"] == "mr_edge_sigma1p35_k1p3_fo1_ded1_cd1"
    assert ranking[1]["edge_classification"] == "risk_only"