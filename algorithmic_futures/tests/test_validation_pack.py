"""
tests/test_validation_pack.py — Tests for the validation pack runner.

Covers pack loading, session entry defaults, manifest creation,
directory structure creation, and failure handling.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import csv
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

    def test_load_pack_historical_holdout_generated(self):
        """Historical holdout pack resolves to a generated reproducible window."""
        pack = load_pack("historical_holdout_20d")
        assert isinstance(pack, ValidationPack)
        assert pack.pack_id == "historical_holdout_20d"
        assert len(pack.sessions) == 20
        assert pack.sessions[0].session_id == "session_20251103"
        assert pack.sessions[-1].session_id == "session_20251128"

    def test_load_pack_route_sensitivity_session_id_pack(self):
        """Route sensitivity pack resolves to the curated threshold-sensitive sessions."""
        pack = load_pack("route_sensitivity_16")
        assert isinstance(pack, ValidationPack)
        assert pack.pack_id == "route_sensitivity_16"
        assert len(pack.sessions) == 16
        assert pack.sessions[0].session_id == "session_20251204"
        assert pack.sessions[-1].session_id == "session_20260112"
        session_ids = [session.session_id for session in pack.sessions]
        assert "session_20260209" in session_ids
        assert "session_20260212" in session_ids


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

    def test_build_replay_args_enables_batch_fast_mode_flags(self) -> None:
        session = SessionEntry(
            session_id="sess_fast",
            start="2026-02-18T14:30:00Z",
            end="2026-02-18T21:00:00Z",
            category="range",
        )

        runner = ValidationPackRunner(
            pack=_make_minimal_pack(),
            batch_fast_mode=True,
        )

        args = runner._build_replay_args(
            session,
            runner.batch_fast_mode,
            runner.mr_reclaim_mode,
            runner.mr_sigma_entry,
            runner.mr_soft_impulse_k,
            runner.mr_dedupe_enabled,
            runner.mr_attempt_cap_enabled,
            runner.mr_cooldown_bars,
            runner.mr_first_outside_enabled,
            runner.mr_touch_latch_reset_buffer,
            runner.mr_dedupe_window_bars,
            runner.mr_dedupe_min_delta_z,
            runner.mr_regime_enabled,
            runner.engine_mode,
            runner.allocator_policy,
            runner.allocator_v1_adx_threshold,
            runner.allocator_v2_trend_open_threshold,
            runner.allocator_v2_rising_threshold,
            runner.allocator_v2_rising_bars,
            runner.allocator_v2_range_threshold,
            runner.allocator_v2_range_bars,
            runner.alloc_openproxy_or_width_atr,
            runner.alloc_openproxy_impulse_atr,
            runner.alloc_openproxy_persist_bars,
            runner.alloc_openproxy_require_break,
            runner.alloc_openproxy_enable_orb_selectivity_refinement,
            runner.alloc_openproxy_low_atr_threshold,
            runner.alloc_openproxy_min_persistence_in_low_atr,
            runner.alloc_openproxy_high_impulse_threshold,
            runner.alloc_openproxy_min_persistence_when_high_impulse,
            runner.alloc_openproxy_medium_impulse_weak_persistence_filter_enabled,
            runner.alloc_openproxy_medium_impulse_decay_filter_enabled,
            runner.alloc_openproxy_medium_impulse_min_atr,
            runner.alloc_openproxy_medium_impulse_max_atr,
            runner.alloc_openproxy_medium_impulse_min,
            runner.alloc_openproxy_medium_impulse_max,
            runner.alloc_openproxy_medium_impulse_min_persistence,
            runner.orb_enabled,
            runner.orb_trigger_mode,
            runner.orb_pullback_confirm_bars,
            runner.orb_pullback_max_bars,
            runner.orb_pullback_tolerance_pts,
            runner.orb_pullback_entry_mode,
        )

        assert args.no_show is True
        assert args.no_dashboard is True
        assert args.no_report is False

    @patch("dotenv.load_dotenv")
    @patch("simulation.mr_exit_simulator.DatabentoReplayProvider")
    @patch("replay_debug.run_debug_replay")
    def test_runner_generates_scorecard_and_enriches_aggregate(
        self,
        mock_replay,
        mock_provider,
        mock_dotenv,
        tmp_path,
    ):
        """Runner writes scorecard outputs and backfills gate metrics to aggregate_metrics.json."""
        pack = ValidationPack(
            pack_id="scorecard_pack",
            description="Single-session pack for scorecard staging",
            sessions=[
                SessionEntry(
                    session_id="score_sess",
                    start="2026-02-18T14:30:00Z",
                    end="2026-02-18T15:00:00Z",
                    category="range",
                )
            ],
        )

        def fake_replay(args):
            import config
            from pathlib import Path

            session_dir = Path(config.ARTIFACTS_DIR) / args.session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            with (session_dir / "signals.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "approved",
                        "signal_type",
                        "rejection_reason",
                        "regime",
                        "side",
                        "sigma_points",
                        "z_score",
                        "band_level",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "approved": "true",
                        "signal_type": "MR",
                        "rejection_reason": "",
                        "regime": "range",
                        "side": "LONG",
                        "sigma_points": "1.4",
                        "z_score": "1.2",
                        "band_level": "2.0",
                    }
                )

            with (session_dir / "trades.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "trade_id",
                        "session_id",
                        "pnl_dollars",
                        "pnl_r",
                        "entry_timestamp",
                        "exit_timestamp",
                        "mae_points",
                        "mfe_points",
                        "hold_minutes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "trade_id": "score_sess_t1",
                        "session_id": args.session_id,
                        "pnl_dollars": "25.0",
                        "pnl_r": "0.5",
                        "entry_timestamp": "2026-02-18T14:50:00+00:00",
                        "exit_timestamp": "2026-02-18T15:00:00+00:00",
                        "mae_points": "0.5",
                        "mfe_points": "1.5",
                        "hold_minutes": "10",
                    }
                )

            return 0

        mock_replay.side_effect = fake_replay
        mock_provider.return_value.replay_trades.return_value = None

        runner = ValidationPackRunner(
            pack=pack,
            artifacts_root=str(tmp_path),
            continue_on_error=True,
        )
        runner.run()

        run_dir = next(tmp_path.iterdir())
        scorecard_metrics_path = run_dir / "scorecard" / "aggregate_metrics.json"
        aggregate_metrics_path = run_dir / "aggregate_metrics.json"

        assert scorecard_metrics_path.exists()
        assert aggregate_metrics_path.exists()

        scorecard_metrics = json.loads(scorecard_metrics_path.read_text(encoding="utf-8"))
        aggregate_metrics = json.loads(aggregate_metrics_path.read_text(encoding="utf-8"))

        assert scorecard_metrics["approval_rate"] == pytest.approx(1.0)
        assert aggregate_metrics["approval_rate"] == pytest.approx(1.0)


class TestAllocatorDebugArtifact:
    @patch("dotenv.load_dotenv")
    @patch("simulation.mr_exit_simulator.DatabentoReplayProvider")
    @patch("replay_debug.run_debug_replay")
    def test_runner_stages_allocator_debug_csv(self, mock_replay, mock_provider, mock_dotenv, tmp_path):
        """Allocator-enabled runs stage one allocator_debug.csv row per session."""
        pack = _make_minimal_pack()

        def fake_replay(args):
            import config
            from pathlib import Path
            session_dir = Path(config.ARTIFACTS_DIR) / args.session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            with (session_dir / "signals.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["approved", "signal_type"])
                writer.writeheader()
                writer.writerow({"approved": "true", "signal_type": "MR"})

            with (session_dir / "trades.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["trade_id", "session_id", "pnl_dollars", "pnl_r", "entry_timestamp", "exit_timestamp"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "trade_id": f"{args.session_id}_t1",
                        "session_id": args.session_id,
                        "pnl_dollars": "25.0",
                        "pnl_r": "0.5",
                        "entry_timestamp": "2026-02-18T14:50:00+00:00",
                        "exit_timestamp": "2026-02-18T15:00:00+00:00",
                    }
                )

            summary = {
                "session_id": args.session_id,
                "total_candidates": 1,
                "total_approved": 1,
                "total_rejected": 0,
                "rejection_breakdown": {},
                "regime_distribution": {"range": 1},
                "rejection_counters": {"rejected_by_daily_loss_governor": 0, "rejected_by_profit_cap": 0},
                "orb_funnel": {
                    "engine_mode": "both",
                    "allocator_policy": args.allocator_policy,
                    "allocator_decision": "mr",
                    "allocator_reason": "OPEN_PROXY_RANGE_SELECTIVITY_V3_MEDIUM_IMPULSE_WEAK_PERSISTENCE impulse_atr=1.00 band=[0.90,2.40) persistence=0",
                    "open_proxy_diagnostics": {
                        "or_high": 105.0,
                        "or_low": 100.0,
                        "opening_range_width_pts": 5.0,
                        "opening_range_width_atr": 2.5,
                        "first_3bar_directional_impulse": 1.0,
                        "signed_imbalance": 0.8,
                        "breakout_persistence": True,
                        "breakout_direction": "UP",
                        "persist_bars_observed": 1,
                        "trigger_width": True,
                        "trigger_impulse": True,
                        "trigger_persist": True,
                        "atr_at_decision": 2.0,
                        "selectivity_refinement_enabled": True,
                        "selectivity_low_atr_caution": False,
                        "selectivity_high_impulse_caution": False,
                        "selectivity_orb_blocked": True,
                        "selectivity_block_reason": "OPEN_PROXY_RANGE_SELECTIVITY_V3_MEDIUM_IMPULSE_WEAK_PERSISTENCE impulse_atr=1.00 band=[0.90,2.40) persistence=0",
                        "pre_selectivity_decision": "orb",
                        "selectivity_medium_impulse_weak_persistence_caution": True,
                        "selectivity_v3_orb_blocked": True,
                        "selectivity_v3_block_reason": "OPEN_PROXY_RANGE_SELECTIVITY_V3_MEDIUM_IMPULSE_WEAK_PERSISTENCE impulse_atr=1.00 band=[0.90,2.40) persistence=0",
                        "pre_v3_selectivity_decision": "orb",
                        "post_v3_selectivity_decision": "mr",
                    },
                },
            }
            (session_dir / "session_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return 0

        mock_replay.side_effect = fake_replay
        mock_provider.return_value.replay_trades.return_value = None

        runner = ValidationPackRunner(
            pack=pack,
            artifacts_root=str(tmp_path),
            allocator_policy="open_proxy_v1",
            engine_mode="both",
        )
        runner.run()

        run_dir = next(tmp_path.iterdir())
        allocator_debug = run_dir / "allocator_debug.csv"
        assert allocator_debug.exists()

        with allocator_debug.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        assert rows[0]["allocator_policy"] == "open_proxy_v1"
        assert rows[0]["route"] == "mr"
        assert rows[0]["width_atr"] == "2.5"
        assert rows[0]["selectivity_refinement_enabled"] == "True"
        assert rows[0]["pre_selectivity_decision"] == "orb"
        assert rows[0]["selectivity_medium_impulse_weak_persistence_caution"] == "True"
        assert rows[0]["selectivity_v3_orb_blocked"] == "True"
        assert rows[0]["selectivity_block_reason"] == (
            "OPEN_PROXY_RANGE_SELECTIVITY_V3_MEDIUM_IMPULSE_WEAK_PERSISTENCE "
            "impulse_atr=1.00 band=[0.90,2.40) persistence=0"
        )
        assert rows[0]["post_v3_selectivity_decision"] == "mr"
        assert rows[0]["notes"] == rows[0]["selectivity_block_reason"]


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
