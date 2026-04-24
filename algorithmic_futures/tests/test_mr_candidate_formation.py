from __future__ import annotations

from validation.mr_candidate_formation import build_candidate_formation_ranking


def test_build_candidate_formation_ranking_prefers_positive_edge_and_yield() -> None:
    rows = [
        {
            "label": "fo1_ded1_cd1",
            "approval_rate": 0.22,
            "expectancy_r": 0.08,
            "avg_r": 0.08,
            "p_target_before_ruin": 0.24,
            "p_ruin": 0.12,
            "dd_p95": 1180.0,
            "trade_count_total": 18,
            "drop_ledger_total": {
                "z_cross_events": 30,
                "candidates_formed": 20,
                "cooldown_rejects": 2,
                "dedupe_rejects": 1,
                "eligible_session_bars": 120,
                "trades": 18,
            },
        },
        {
            "label": "fo0_ded0_cd0",
            "approval_rate": 0.35,
            "expectancy_r": -0.02,
            "avg_r": -0.02,
            "p_target_before_ruin": 0.10,
            "p_ruin": 0.28,
            "dd_p95": 1450.0,
            "trade_count_total": 24,
            "drop_ledger_total": {
                "z_cross_events": 28,
                "candidates_formed": 24,
                "cooldown_rejects": 0,
                "dedupe_rejects": 0,
                "eligible_session_bars": 120,
                "trades": 24,
            },
        },
        {
            "label": "fo1_ded1_cd2",
            "approval_rate": 0.12,
            "expectancy_r": 0.01,
            "avg_r": 0.01,
            "p_target_before_ruin": 0.05,
            "p_ruin": 0.18,
            "dd_p95": 1220.0,
            "trade_count_total": 8,
            "drop_ledger_total": {
                "z_cross_events": 30,
                "candidates_formed": 10,
                "cooldown_rejects": 12,
                "dedupe_rejects": 8,
                "eligible_session_bars": 120,
                "trades": 8,
            },
        },
    ]

    ranking = build_candidate_formation_ranking(rows)

    assert ranking[0]["label"] == "fo1_ded1_cd1"
    assert ranking[0]["formation_classification"] == "formation_progress"
    assert ranking[1]["label"] == "fo0_ded0_cd0"
    assert ranking[1]["formation_classification"] == "negative_edge"
    assert ranking[2]["label"] == "fo1_ded1_cd2"
    assert ranking[2]["formation_classification"] == "over_suppressed"