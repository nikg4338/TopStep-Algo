"""Tests for Databento cache inspection helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.cache_inventory import list_cached_sessions, parse_cached_session_file, summarize_cached_sessions


def test_parse_cached_session_file_round_trips_expected_fields(tmp_path):
    path = tmp_path / "20251103_143000__20251103_210000.parquet"
    path.write_text("", encoding="utf-8")

    session = parse_cached_session_file(path)

    assert session.session_id == "session_20251103"
    assert session.date == "20251103"
    assert session.start == "2025-11-03T14:30:00Z"
    assert session.end == "2025-11-03T21:00:00Z"


def test_list_cached_sessions_sorts_by_date(tmp_path):
    files = [
        tmp_path / "20251104_143000__20251104_210000.parquet",
        tmp_path / "20251103_143000__20251103_210000.parquet",
    ]
    for path in files:
        path.write_text("", encoding="utf-8")

    sessions = list_cached_sessions(cache_dir=tmp_path)

    assert [session.session_id for session in sessions] == ["session_20251103", "session_20251104"]


def test_summarize_cached_sessions_reports_obvious_gap(tmp_path):
    for name in [
        "20251103_143000__20251103_210000.parquet",
        "20251105_143000__20251105_210000.parquet",
    ]:
        (tmp_path / name).write_text("", encoding="utf-8")

    summary = summarize_cached_sessions(cache_dir=tmp_path)

    assert summary["session_count"] == 2
    assert summary["earliest_session"] == "session_20251103"
    assert summary["latest_session"] == "session_20251105"
    assert summary["gap_count"] == 1
    assert summary["gap_session_ids"] == ["session_20251104"]