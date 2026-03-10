"""Tests for forward-shadow tracker aggregation and schema handling."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.candidate_openfix import build_forward_shadow_rows, summarize_forward_shadow


def test_forward_shadow_rows_include_expected_schema():
    candidate_summary = {
        "preset_name": "mainline_combine_v1_1_allocator_openfix",
        "per_session_route": {"session_20260303": "orb", "session_20260304": "mr"},
        "session_pnl": {"session_20260303": 100.0, "session_20260304": -50.0},
        "allocator_rows": [
            {"session_id": "session_20260303", "date": "20260303", "confidence_score": 0.8, "width_atr": 2.3, "impulse": 1.1, "persistence": 1, "notes": "trend"},
            {"session_id": "session_20260304", "date": "20260304", "confidence_score": 0.2, "notes": "range"},
        ],
    }
    baseline_summary = {"session_pnl": {"session_20260303": 80.0, "session_20260304": -25.0}}
    rows = build_forward_shadow_rows(candidate_summary, baseline_summary=baseline_summary)
    assert len(rows) == 2
    assert rows[0]["date"] == "20260303"
    assert rows[0]["preset"] == "mainline_combine_v1_1_allocator_openfix"
    assert "cumulative_equity" in rows[0]
    assert "false_positive_orb" in rows[0]


def test_forward_shadow_cumulative_progress_aggregates_correctly():
    candidate_summary = {
        "preset_name": "mainline_combine_v1_1_allocator_openfix",
        "per_session_route": {"session_20260303": "orb", "session_20260304": "mr"},
        "session_pnl": {"session_20260303": 100.0, "session_20260304": -40.0},
        "allocator_rows": [],
    }
    rows = build_forward_shadow_rows(candidate_summary)
    assert rows[0]["cumulative_equity"] == 100.0
    assert rows[1]["cumulative_equity"] == 60.0
    assert rows[1]["cumulative_drawdown"] == 40.0


def test_forward_shadow_handles_missing_optional_allocator_fields():
    candidate_summary = {
        "preset_name": "mainline_combine_v1_1_allocator_openfix",
        "per_session_route": {"session_20260303": "orb"},
        "session_pnl": {"session_20260303": 25.0},
        "allocator_rows": [{"session_id": "session_20260303"}],
    }
    rows = build_forward_shadow_rows(candidate_summary)
    assert rows[0]["route_confidence"] == ""
    assert rows[0]["opening_range_width"] == ""
    assert rows[0]["notes"] == ""


def test_forward_shadow_summary_backward_compatible_without_debug_columns():
    rows = [
        {
            "date": "20260303",
            "session_id": "session_20260303",
            "preset": "mainline_combine_v1_1_allocator_openfix",
            "route": "orb",
            "session_pnl": 100.0,
            "win_flag": True,
            "daily_rule_clean": True,
            "consistency_rule_clean": True,
            "false_positive_orb": "",
            "cumulative_equity": 100.0,
            "cumulative_target_progress": 0.0333,
            "cumulative_drawdown": 0.0,
            "notes": "",
        }
    ]
    summary = summarize_forward_shadow(rows)
    assert summary["sessions_processed"] == 1
    assert summary["orb_routed_sessions"] == 1
    assert summary["status"] in {"stable", "watch", "degrading"}
