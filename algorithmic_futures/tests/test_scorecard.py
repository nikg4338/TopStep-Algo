"""
tests/test_scorecard.py — Tests for the scorecard aggregator.

Covers aggregate computation, approval rate, rejection merging,
missing file handling, baseline comparison, and summary output.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
from pathlib import Path

import pytest

from validation.scorecard import ScorecardAggregator, run_scorecard


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _create_mock_session(
    sessions_dir: Path,
    session_id: str,
    n_candidates: int = 5,
    n_approved: int = 3,
    rejection_reasons: list[str] | None = None,
    trades: list[dict] | None = None,
) -> Path:
    """Create mock session artifacts inside ``sessions_dir / session_id``.

    Writes signals.csv and session_summary.json.  Optionally writes
    trades.csv when *trades* is provided.

    Parameters
    ----------
    sessions_dir:
        Parent ``sessions/`` directory inside the run root.
    session_id:
        Unique session identifier (used as sub-directory name).
    n_candidates:
        Total number of candidate signals.
    n_approved:
        Number of approved signals (must be <= n_candidates).
    rejection_reasons:
        Rejection reasons for rejected signals.  Cycled if shorter
        than the number of rejections.
    trades:
        Optional list of trade dicts to write as trades.csv.

    Returns
    -------
    Path
        Path to the session directory.
    """
    session_dir = sessions_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    n_rejected = n_candidates - n_approved
    if rejection_reasons is None:
        rejection_reasons = ["COOLDOWN"]

    # -- session_summary.json ------------------------------------------------
    rej_breakdown: dict[str, int] = {}
    for i in range(n_rejected):
        reason = rejection_reasons[i % len(rejection_reasons)]
        rej_breakdown[reason] = rej_breakdown.get(reason, 0) + 1

    summary = {
        "session_id": session_id,
        "total_candidates": n_candidates,
        "total_approved": n_approved,
        "total_rejected": n_rejected,
        "rejection_breakdown": rej_breakdown,
    }
    (session_dir / "session_summary.json").write_text(json.dumps(summary, indent=2))

    # -- signals.csv ---------------------------------------------------------
    fieldnames = [
        "timestamp", "regime", "signal_type", "side",
        "candidate_price", "approved", "rejection_reason",
        "band_level", "vwap", "sigma_value", "session_id",
    ]
    with (session_dir / "signals.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_candidates):
            approved = i < n_approved
            reason = "" if approved else rejection_reasons[(i - n_approved) % len(rejection_reasons)]
            writer.writerow({
                "timestamp": f"2026-02-18T14:{30 + i:02d}:00Z",
                "regime": "range",
                "signal_type": "MR",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "candidate_price": 5900.0 + i,
                "approved": str(approved),
                "rejection_reason": reason,
                "band_level": 2.5,
                "vwap": 5900.0,
                "sigma_value": 2.0,
                "session_id": session_id,
            })

    # -- trades.csv (optional) -----------------------------------------------
    if trades is not None:
        trade_fields = [
            "entry_time", "exit_time", "side", "entry_price", "exit_price",
            "pnl_dollars", "pnl_r", "exit_reason", "mae_points", "mfe_points",
            "hold_minutes",
        ]
        with (session_dir / "trades.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=trade_fields)
            writer.writeheader()
            for t in trades:
                writer.writerow(t)

    return session_dir


def _create_run_dir(
    tmp_path: Path,
    run_name: str,
    sessions_data: list[dict],
) -> Path:
    """Create a full run directory with manifest.json + sessions/ subdirectory.

    Expected layout::

        run_dir/
          manifest.json
          sessions/
            session_id_1/
              signals.csv
              session_summary.json
              trades.csv (optional)
            session_id_2/
              ...
    """
    run_dir = tmp_path / run_name
    sessions_dir = run_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    manifest_sessions = []
    for sd in sessions_data:
        sid = sd["session_id"]
        _create_mock_session(
            sessions_dir,
            sid,
            n_candidates=sd.get("n_candidates", 5),
            n_approved=sd.get("n_approved", 3),
            rejection_reasons=sd.get("rejection_reasons"),
            trades=sd.get("trades"),
        )
        manifest_sessions.append({
            "session_id": sid,
            "success": sd.get("success", True),
            "category": sd.get("category", "default"),
        })

    manifest = {
        "run_id": run_name,
        "pack_id": "test_pack",
        "sessions": manifest_sessions,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return run_dir


# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAggregateBasic:
    """Create 2 sessions, run aggregator, verify total candidates/approved/rejected."""

    def test_aggregate_basic(self, tmp_path):
        run_dir = _create_run_dir(tmp_path, "run_basic", [
            {"session_id": "s1", "n_candidates": 5, "n_approved": 3},
            {"session_id": "s2", "n_candidates": 4, "n_approved": 2},
        ])

        agg = ScorecardAggregator(str(run_dir))
        result = agg.generate()

        assert result["total_candidates"] == 9
        assert result["total_approved"] == 5
        assert result["total_rejected"] == 4
        assert result["total_sessions_run"] == 2
        assert result["total_sessions_succeeded"] == 2
        assert result["total_sessions_failed"] == 0


class TestApprovalRateComputation:
    """Verify aggregate_approval_rate = approved / total."""

    def test_approval_rate_computation(self, tmp_path):
        run_dir = _create_run_dir(tmp_path, "run_ar", [
            {"session_id": "s1", "n_candidates": 10, "n_approved": 6},
        ])

        agg = ScorecardAggregator(str(run_dir))
        result = agg.generate()

        expected_rate = 6 / 10
        assert result["aggregate_approval_rate"] == pytest.approx(expected_rate)


class TestRejectionBreakdownMerge:
    """Two sessions with different rejection reasons merge correctly."""

    def test_rejection_breakdown_merge(self, tmp_path):
        run_dir = _create_run_dir(tmp_path, "run_rej", [
            {
                "session_id": "s1",
                "n_candidates": 5,
                "n_approved": 2,
                "rejection_reasons": ["COOLDOWN", "MAX_LONG_ATTEMPTS"],
            },
            {
                "session_id": "s2",
                "n_candidates": 4,
                "n_approved": 1,
                "rejection_reasons": ["SESSION_CUTOFF", "COOLDOWN"],
            },
        ])

        agg = ScorecardAggregator(str(run_dir))
        result = agg.generate()

        breakdown = result["aggregate_rejection_breakdown"]
        # s1: 3 rejected cycling [COOLDOWN, MAX_LONG_ATTEMPTS]
        #     → COOLDOWN(2), MAX_LONG_ATTEMPTS(1)
        # s2: 3 rejected cycling [SESSION_CUTOFF, COOLDOWN]
        #     → SESSION_CUTOFF(2), COOLDOWN(1)
        # Merged: COOLDOWN=3, MAX_LONG_ATTEMPTS=1, SESSION_CUTOFF=2
        assert breakdown["COOLDOWN"] == 3
        assert breakdown["MAX_LONG_ATTEMPTS"] == 1
        assert breakdown["SESSION_CUTOFF"] == 2


class TestHandlesMissingSignalsCsv:
    """Session directory exists but no signals.csv — should not crash."""

    def test_handles_missing_signals_csv(self, tmp_path):
        run_dir = tmp_path / "run_nosig"
        sessions_dir = run_dir / "sessions"
        session_dir = sessions_dir / "s1"
        session_dir.mkdir(parents=True)

        # Write session_summary.json but NO signals.csv
        summary = {
            "session_id": "s1",
            "total_candidates": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "rejection_breakdown": {},
        }
        (session_dir / "session_summary.json").write_text(json.dumps(summary))

        manifest = {
            "run_id": "run_nosig",
            "pack_id": "test_pack",
            "sessions": [{"session_id": "s1", "success": True, "category": "default"}],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        agg = ScorecardAggregator(str(run_dir))
        result = agg.generate()

        # Should not crash; totals can be zero
        assert result["total_candidates"] == 0
        assert result["total_approved"] == 0
        assert result["total_rejected"] == 0


class TestHandlesMissingTradesCsv:
    """No trades.csv present — trade_metrics should be absent from result."""

    def test_handles_missing_trades_csv(self, tmp_path):
        run_dir = _create_run_dir(tmp_path, "run_notrades", [
            {"session_id": "s1", "n_candidates": 5, "n_approved": 3},
        ])

        agg = ScorecardAggregator(str(run_dir))
        result = agg.generate()

        # trade_metrics key should not exist when no trades.csv is present
        assert "trade_metrics" not in result


class TestBaselineComparison:
    """Compare a second run to a first run; verify baseline_comparison key with deltas."""

    def test_baseline_comparison(self, tmp_path):
        # First run (baseline) — generate its scorecard so
        # aggregate_metrics.json is written to run_a/scorecard/
        run_a = _create_run_dir(tmp_path, "run_a", [
            {"session_id": "s1", "n_candidates": 10, "n_approved": 7},
        ])
        ScorecardAggregator(str(run_a)).generate()

        # Second run — compare to run_a
        run_b = _create_run_dir(tmp_path, "run_b", [
            {"session_id": "s1", "n_candidates": 10, "n_approved": 5},
        ])

        agg = ScorecardAggregator(str(run_b), compare_to_dir=str(run_a))
        result = agg.generate()

        assert "baseline_comparison" in result
        deltas = result["baseline_comparison"]

        # run_b approval rate (0.5) vs run_a approval rate (0.7) → negative delta
        ar_delta = deltas["aggregate_approval_rate"]
        assert ar_delta["current"] == pytest.approx(0.5)
        assert ar_delta["previous"] == pytest.approx(0.7)
        assert ar_delta["delta"] == pytest.approx(-0.2)

        # total_approved: 5 vs 7 → delta = -2
        assert deltas["total_approved"]["delta"] == pytest.approx(-2.0)


class TestWritesSummaryMd:
    """Verify summary.md is created and contains expected markdown headers."""

    def test_writes_summary_md(self, tmp_path):
        run_dir = _create_run_dir(tmp_path, "run_md", [
            {"session_id": "s1", "n_candidates": 5, "n_approved": 3},
        ])

        agg = ScorecardAggregator(str(run_dir))
        agg.generate()

        md_path = run_dir / "scorecard" / "summary.md"
        assert md_path.exists(), "summary.md should be written by generate()"

        content = md_path.read_text()
        assert "# Validation Scorecard" in content
        assert "## Aggregate Metrics" in content
        assert "## Per-Session Summary" in content
        assert "## Rejection Breakdown" in content
        assert "Approval Rate" in content
