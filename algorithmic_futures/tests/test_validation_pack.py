"""
tests/test_validation_pack.py — Tests for the validation pack runner.

Covers pack loading, session entry defaults, manifest creation,
directory structure creation, and failure handling.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import patch

import pytest

from validation.validation_pack import (
    load_pack,
    SessionEntry,
    SessionResult,
    ValidationPack,
    ValidationRunManifest,
    ValidationPackRunner,
)


# ═══════════════════════════════════════════════════════════════════════
#  Pack loading
# ═══════════════════════════════════════════════════════════════════════


class TestLoadPack:
    def test_load_pack_builtin(self):
        """load_pack('baseline_v1') returns a valid ValidationPack."""
        pack = load_pack("baseline_v1")
        assert isinstance(pack, ValidationPack)
        assert pack.pack_id == "baseline_v1"
        assert len(pack.sessions) > 0
        for s in pack.sessions:
            assert isinstance(s, SessionEntry)
            assert s.start is not None
            assert s.end is not None

    def test_load_pack_unknown(self):
        """load_pack('nonexistent') raises ValueError."""
        with pytest.raises(ValueError, match="nonexistent"):
            load_pack("nonexistent")


# ═══════════════════════════════════════════════════════════════════════
#  Dataclass defaults
# ═══════════════════════════════════════════════════════════════════════


class TestSessionEntry:
    def test_session_entry_defaults(self):
        """SessionEntry dataclass has correct default values."""
        entry = SessionEntry(
            session_id="s1",
            start="2026-02-18T14:30:00Z",
            end="2026-02-18T16:00:00Z",
            category="range",
        )
        assert entry.session_id == "s1"
        assert entry.start == "2026-02-18T14:30:00Z"
        assert entry.end == "2026-02-18T16:00:00Z"
        assert entry.category == "range"
        # Defaults
        assert entry.symbol == "MES.c.0"
        assert entry.tags == []
        assert entry.notes == ""


# ═══════════════════════════════════════════════════════════════════════
#  Manifest creation
# ═══════════════════════════════════════════════════════════════════════


class TestManifestCreation:
    def test_manifest_creation(self):
        """ValidationRunManifest can be created and fields are populated."""
        results = [
            SessionResult(
                session_id="s1",
                success=True,
                category="range",
                runtime_seconds=1.23,
            ),
            SessionResult(
                session_id="s2",
                success=False,
                category="trend",
                error_message="timeout",
                runtime_seconds=4.56,
            ),
        ]
        manifest = ValidationRunManifest(
            run_id="run_001",
            pack_id="baseline_v1",
            timestamp="2026-02-21T10:00:00",
            config_hash="abc123",
            sessions=results,
            total_runtime_seconds=5.79,
            notes="test run",
        )
        assert manifest.run_id == "run_001"
        assert manifest.pack_id == "baseline_v1"
        assert manifest.timestamp == "2026-02-21T10:00:00"
        assert manifest.config_hash == "abc123"
        assert len(manifest.sessions) == 2
        assert manifest.sessions[0].session_id == "s1"
        assert manifest.sessions[0].success is True
        assert manifest.sessions[1].success is False
        assert manifest.sessions[1].error_message == "timeout"
        assert manifest.total_runtime_seconds == 5.79
        assert manifest.notes == "test run"


# ═══════════════════════════════════════════════════════════════════════
#  Runner integration
# ═══════════════════════════════════════════════════════════════════════


def _make_minimal_pack() -> ValidationPack:
    """Create a minimal pack with 2 sessions for testing."""
    return ValidationPack(
        pack_id="test_pack",
        description="Minimal test pack",
        sessions=[
            SessionEntry(
                session_id="sess_a",
                start="2026-02-18T14:30:00Z",
                end="2026-02-18T15:00:00Z",
                category="range",
            ),
            SessionEntry(
                session_id="sess_b",
                start="2026-02-19T14:30:00Z",
                end="2026-02-19T15:00:00Z",
                category="trend",
            ),
        ],
    )


class TestRunnerIntegration:
    @patch("replay_debug.run_debug_replay", return_value=0)
    @patch("dotenv.load_dotenv")
    @patch("simulation.mr_exit_simulator.DatabentoReplayProvider")
    def test_runner_calls_exit_sim(self, mock_provider, mock_dotenv, mock_replay, tmp_path):
        """Runner calls exit sim and writes artifacts to the session folder."""
        pack = _make_minimal_pack()
        runner = ValidationPackRunner(
            pack=pack,
            artifacts_root=str(tmp_path),
            continue_on_error=True,
        )

        # We need to mock the replay to actually write a signals.csv so the exit sim runs
        def fake_replay(args):
            # args.save_path is empty, but config.ARTIFACTS_DIR is set to sessions_dir
            import config
            from pathlib import Path
            session_dir = Path(config.ARTIFACTS_DIR) / args.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Write a fake signals.csv
            import csv
            from tests.test_mr_exit_simulator import SIGNAL_CSV_COLUMNS, _signal
            csv_path = session_dir / "signals.csv"
            with csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=SIGNAL_CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(_signal(session_id=args.session_id))
            return 0

        mock_replay.side_effect = fake_replay

        # Mock the provider to just return immediately without fetching
        mock_provider.return_value.replay_trades.return_value = None

        manifest = runner.run()

        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        
        # Check session A
        sess_a_dir = run_dir / "sessions" / "sess_a"
        assert (sess_a_dir / "signals.csv").exists()
        assert (sess_a_dir / "trades.csv").exists()
        assert (sess_a_dir / "exit_sim_diagnostics.json").exists()
        assert (sess_a_dir / "exit_sim_called.txt").exists()

        # Check manifest references
        assert manifest.sessions[0].artifact_dir == str(sess_a_dir)

class TestRunnerCreatesDirs:
    @patch("replay_debug.run_debug_replay", return_value=0)
    @patch("dotenv.load_dotenv")
    def test_runner_creates_dirs(self, mock_dotenv, mock_replay, tmp_path):
        """Runner creates the expected directory structure and writes manifest."""
        pack = _make_minimal_pack()
        runner = ValidationPackRunner(
            pack=pack,
            artifacts_root=str(tmp_path),
            continue_on_error=True,
        )
        manifest = runner.run()

        # The runner creates a run_dir = artifacts_root / run_id
        # run_id starts with the pack_id
        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        assert run_dir.name.startswith("test_pack_")

        # manifest.json should exist in the run directory
        manifest_path = run_dir / "manifest.json"
        assert manifest_path.exists()

        with manifest_path.open() as fh:
            data = json.load(fh)
        assert data["pack_id"] == "test_pack"
        assert len(data["sessions"]) == 2

        # Session directories should exist under run_dir/sessions/
        sessions_dir = run_dir / "sessions"
        assert sessions_dir.exists()

        # Both sessions should have succeeded
        assert manifest.sessions[0].success is True
        assert manifest.sessions[1].success is True

        # run_debug_replay should have been called twice
        assert mock_replay.call_count == 2


class TestRunnerHandlesFailure:
    @patch("replay_debug.run_debug_replay")
    @patch("dotenv.load_dotenv")
    def test_runner_handles_failure(self, mock_dotenv, mock_replay, tmp_path):
        """If one session raises, manifest records failure; others continue."""
        # First call succeeds, second raises
        mock_replay.side_effect = [0, RuntimeError("Databento timeout")]

        pack = _make_minimal_pack()
        runner = ValidationPackRunner(
            pack=pack,
            artifacts_root=str(tmp_path),
            continue_on_error=True,
        )
        manifest = runner.run()

        # Both sessions should be recorded
        assert len(manifest.sessions) == 2

        # First session succeeded
        assert manifest.sessions[0].success is True
        assert manifest.sessions[0].session_id == "sess_a"

        # Second session failed
        assert manifest.sessions[1].success is False
        assert manifest.sessions[1].session_id == "sess_b"
        assert "Databento timeout" in manifest.sessions[1].error_message

        # Manifest file should still be written
        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        manifest_path = run_dirs[0] / "manifest.json"
        assert manifest_path.exists()

        with manifest_path.open() as fh:
            data = json.load(fh)

        sessions = data["sessions"]
        assert sessions[0]["success"] is True
        assert sessions[1]["success"] is False
        assert "Databento timeout" in sessions[1]["error_message"]
