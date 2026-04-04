from __future__ import annotations

from validation.mr_approval_calibration import build_candidate_grid, rank_candidate_rows


def test_build_candidate_grid_creates_expected_labels() -> None:
    candidates = build_candidate_grid(
        sigma_entries=[1.2],
        reclaim_modes=["off", "soft"],
        cooldown_bars=[0, 1],
        first_outside_modes=["off"],
        dedupe_modes=["off"],
        attempt_cap_modes=["on"],
        regime_modes=["on"],
        soft_range_impulse_k=1.2,
    )

    labels = [candidate.label for candidate in candidates]
    assert labels == [
        "mr_sigma1p2_off_cd0_fo0_ded0_cap1_reg1",
        "mr_sigma1p2_off_cd1_fo0_ded0_cap1_reg1",
        "mr_sigma1p2_soft_cd0_fo0_ded0_cap1_reg1",
        "mr_sigma1p2_soft_cd1_fo0_ded0_cap1_reg1",
    ]


def test_rank_candidate_rows_prefers_promotion_like_pass() -> None:
    rows = [
        {
            "label": "candidate_a",
            "promotion_like_pass": False,
            "failed_checks": ["mc_target_prob"],
            "p_target_before_ruin": 0.55,
            "expectancy_r": 0.20,
            "p_ruin": 0.07,
            "dd_p95": 800.0,
            "losing_streak_p95": 4.0,
            "approval_rate": 0.20,
            "trade_count_total": 100,
        },
        {
            "label": "candidate_b",
            "promotion_like_pass": True,
            "failed_checks": [],
            "p_target_before_ruin": 0.61,
            "expectancy_r": 0.18,
            "p_ruin": 0.08,
            "dd_p95": 900.0,
            "losing_streak_p95": 5.0,
            "approval_rate": 0.18,
            "trade_count_total": 90,
        },
    ]

    ranked = rank_candidate_rows(rows)

    assert ranked[0]["label"] == "candidate_b"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["label"] == "candidate_a"