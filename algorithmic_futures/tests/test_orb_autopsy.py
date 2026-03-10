"""Tests for ORB autopsy dataset and report helpers."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.orb_autopsy import (
    assign_orb_label,
    build_orb_autopsy_rows,
    candidate_failure_hypotheses,
    grouped_numeric_summary,
    summarize_orb_dataset,
)


def test_assign_orb_label_uses_baseline_delta_for_false_positive():
    assert assign_orb_label(route="orb", session_pnl=-50.0, baseline_session_pnl=10.0, false_positive=True) == "bad_orb"
    assert assign_orb_label(route="orb", session_pnl=20.0, baseline_session_pnl=10.0, false_positive=False) == "good_orb"
    assert assign_orb_label(route="orb", session_pnl=0.0, baseline_session_pnl=0.0, false_positive=False) == "neutral_orb"


def test_build_orb_autopsy_rows_handles_missing_optional_columns():
    candidate_summary = {
        "pack_id": "historical_holdout_20d",
        "per_session_route": {"session_20251103": "orb"},
        "session_pnl": {"session_20251103": 25.0},
        "allocator_rows": [{"session_id": "session_20251103", "date": "20251103"}],
    }
    rows = build_orb_autopsy_rows(candidate_summary, window_label="holdout", candidate_run_id="run_1")
    assert len(rows) == 1
    assert rows[0]["close_location"] == ""
    assert rows[0]["one_sidedness"] == ""
    assert rows[0]["label"] == "good_orb"


def test_build_orb_autopsy_rows_includes_baseline_delta_fields():
    candidate_summary = {
        "pack_id": "historical_holdout_20d",
        "per_session_route": {"session_20251103": "orb"},
        "session_pnl": {"session_20251103": -77.5},
        "allocator_rows": [{"session_id": "session_20251103", "date": "20251103", "confidence_score": 0.8}],
    }
    baseline_summary = {
        "per_session_route": {"session_20251103": "mr"},
        "session_pnl": {"session_20251103": 425.03},
    }
    rows = build_orb_autopsy_rows(
        candidate_summary,
        window_label="holdout",
        candidate_run_id="run_1",
        baseline_summary=baseline_summary,
        baseline_run_id="run_base",
    )
    assert rows[0]["baseline_session_pnl"] == 425.03
    assert rows[0]["delta_vs_baseline"] == -502.53
    assert rows[0]["false_positive"] is True
    assert rows[0]["label"] == "bad_orb"


def test_grouped_numeric_summary_emits_expected_schema():
    rows = [
        {"label": "good_orb", "route_confidence": 0.8, "opening_range_width": 20.0, "atr": 10.0, "width_atr": 2.0, "impulse": 1.2, "persistence": 1, "session_pnl": 100.0, "baseline_session_pnl": 50.0, "delta_vs_baseline": 50.0, "cumulative_equity": 100.0, "cumulative_drawdown": 0.0},
        {"label": "bad_orb", "route_confidence": 0.3, "opening_range_width": 30.0, "atr": 12.0, "width_atr": 2.5, "impulse": 0.4, "persistence": 0, "session_pnl": -80.0, "baseline_session_pnl": 20.0, "delta_vs_baseline": -100.0, "cumulative_equity": 20.0, "cumulative_drawdown": 80.0},
    ]
    summary = grouped_numeric_summary(rows)
    assert any(row["group"] == "good_orb" and row["feature"] == "route_confidence" for row in summary)


def test_candidate_failure_hypotheses_surface_simple_separator():
    rows = [
        {"label": "good_orb", "route_confidence": 0.8, "opening_range_width": 20.0, "atr": 10.0, "width_atr": 2.0, "impulse": 1.5, "persistence": 1, "one_sidedness": 1.4, "false_positive": False, "window_label": "recent"},
        {"label": "bad_orb", "route_confidence": 0.4, "opening_range_width": 30.0, "atr": 10.0, "width_atr": 3.0, "impulse": 0.3, "persistence": 0, "one_sidedness": 0.2, "false_positive": True, "window_label": "holdout"},
    ]
    hypotheses = candidate_failure_hypotheses(rows)
    assert hypotheses


def test_summarize_orb_dataset_accepts_csv_like_boolean_strings():
    summary = summarize_orb_dataset(
        [
            {"label": "bad_orb", "window_label": "holdout", "false_positive": "True", "baseline_available": "True"},
            {"label": "good_orb", "window_label": "recent", "false_positive": "False", "baseline_available": "True"},
        ]
    )
    assert summary["false_positive_count"] == 1
    assert summary["baseline_available_count"] == 2
