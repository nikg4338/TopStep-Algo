"""
tests/test_replay_profile_bridge.py — Tests for the replay profile bridge.

Covers profile building from trades, low-sample warnings, JSON output keys,
all-wins, all-losses, and empty-trades edge cases.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from dataclasses import fields as dc_fields

import pytest

from validation.replay_profile_bridge import ReplayDerivedProfile, ReplayProfileBridge


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

CSV_HEADER = (
    "trade_id,session_id,signal_timestamp,side,entry_timestamp,entry_price,"
    "stop_price,target_price,exit_timestamp,exit_price,exit_reason,"
    "pnl_points,pnl_dollars,pnl_r,mae_points,mfe_points,hold_minutes,"
    "regime_at_entry,sigma_band_level"
)


def _trade_row(
    trade_id: str = "t1",
    session_id: str = "s1",
    side: str = "BUY",
    entry_price: float = 5000.0,
    exit_price: float = 5010.0,
    pnl_points: float = 10.0,
    pnl_dollars: float = 125.0,
    pnl_r: float = 1.5,
    mae_points: float = 2.0,
    mfe_points: float = 12.0,
    hold_minutes: float = 8.0,
    sigma_band_level: float = 1.2,
    exit_reason: str = "target",
) -> str:
    """Return a single CSV data row with sensible defaults."""
    return (
        f"{trade_id},{session_id},2026-02-18T14:30:00Z,{side},"
        f"2026-02-18T14:31:00Z,{entry_price},4990.0,5020.0,"
        f"2026-02-18T14:39:00Z,{exit_price},{exit_reason},"
        f"{pnl_points},{pnl_dollars},{pnl_r},{mae_points},{mfe_points},"
        f"{hold_minutes},trending,{sigma_band_level}"
    )


def _write_trades_csv(session_dir, rows: list[str]) -> None:
    """Write a trades.csv with header + data rows into *session_dir*."""
    session_dir.mkdir(parents=True, exist_ok=True)
    csv_path = session_dir / "trades.csv"
    csv_path.write_text(
        "\n".join([CSV_HEADER] + rows) + "\n",
        encoding="utf-8",
    )


def _make_run_dir(tmp_path, sessions: dict[str, list[str]], manifest=None):
    """Build a minimal run directory with trade CSVs and optional manifest.

    Parameters
    ----------
    sessions : dict[str, list[str]]
        Mapping of session_id → list of CSV data rows.
    manifest : dict | None
        If provided, written as manifest.json.
    """
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()

    if manifest is not None:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    for session_id, rows in sessions.items():
        _write_trades_csv(run_dir / "sessions" / session_id, rows)

    return run_dir


# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════


def test_build_profile_from_trades(tmp_path):
    """Two sessions with multiple trades → correct aggregated statistics."""
    sessions = {
        "sess1": [
            _trade_row(trade_id="t1", pnl_dollars=100.0, pnl_r=1.0),
            _trade_row(trade_id="t2", pnl_dollars=-50.0, pnl_r=-0.5),
            _trade_row(trade_id="t3", pnl_dollars=200.0, pnl_r=2.0),
        ],
        "sess2": [
            _trade_row(trade_id="t4", pnl_dollars=-80.0, pnl_r=-0.8),
            _trade_row(trade_id="t5", pnl_dollars=150.0, pnl_r=1.5),
        ],
    }
    run_dir = _make_run_dir(tmp_path, sessions)
    bridge = ReplayProfileBridge(str(run_dir))
    profile = bridge.build_profile()

    assert isinstance(profile, ReplayDerivedProfile)
    assert profile.sample_size_trades == 5
    assert profile.sample_size_sessions == 2
    assert profile.sessions_in_sample == 2

    # 3 wins out of 5
    assert profile.win_rate == pytest.approx(0.6, abs=1e-4)

    # avg_win = (100 + 200 + 150) / 3 = 150.0
    assert profile.avg_win_dollars == pytest.approx(150.0, abs=0.01)

    # avg_loss = (-50 + -80) / 2 = -65.0
    assert profile.avg_loss_dollars == pytest.approx(-65.0, abs=0.01)

    # expectancy = mean of all pnl_dollars = (100-50+200-80+150)/5 = 64.0
    assert profile.expectancy_dollars == pytest.approx(64.0, abs=0.01)

    # payoff_ratio = |150 / -65| ≈ 2.3077
    assert profile.payoff_ratio == pytest.approx(2.3077, abs=0.01)

    assert profile.source == "replay_derived"
    assert profile.recommended_mc_horizon_trades > 0


def test_low_sample_warning(tmp_path):
    """Fewer trades than min_trade_count → notes contain WARNING."""
    sessions = {
        "sess1": [
            _trade_row(trade_id="t1", pnl_dollars=100.0, pnl_r=1.0),
            _trade_row(trade_id="t2", pnl_dollars=-50.0, pnl_r=-0.5),
        ],
    }
    run_dir = _make_run_dir(tmp_path, sessions)
    bridge = ReplayProfileBridge(str(run_dir), min_trade_count=10)
    profile = bridge.build_profile()

    assert profile.sample_size_trades == 2
    assert "WARNING" in profile.notes
    assert "below minimum" in profile.notes.lower() or "sample size" in profile.notes.lower()


def test_profile_json_keys(tmp_path):
    """Written JSON has keys matching every dataclass field."""
    sessions = {
        "sess1": [
            _trade_row(trade_id="t1", pnl_dollars=100.0, pnl_r=1.0),
            _trade_row(trade_id="t2", pnl_dollars=-50.0, pnl_r=-0.5),
        ],
    }
    run_dir = _make_run_dir(tmp_path, sessions)
    bridge = ReplayProfileBridge(str(run_dir), min_trade_count=1)
    profile = bridge.build_profile()
    json_path = bridge.write_profile(profile)

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    expected_keys = {f.name for f in dc_fields(ReplayDerivedProfile)}
    assert set(data.keys()) == expected_keys


def test_all_wins(tmp_path):
    """All pnl_dollars > 0 → avg_loss_dollars==0.0, notes warn about no losses."""
    sessions = {
        "sess1": [
            _trade_row(trade_id="t1", pnl_dollars=100.0, pnl_r=1.0),
            _trade_row(trade_id="t2", pnl_dollars=200.0, pnl_r=2.0),
            _trade_row(trade_id="t3", pnl_dollars=50.0, pnl_r=0.5),
        ],
    }
    run_dir = _make_run_dir(tmp_path, sessions)
    bridge = ReplayProfileBridge(str(run_dir), min_trade_count=1)
    profile = bridge.build_profile()

    assert profile.win_rate == pytest.approx(1.0, abs=1e-6)
    assert profile.avg_loss_dollars == 0.0
    assert profile.avg_loss_r == 0.0
    assert "no losing trades" in profile.notes.lower() or "no loss" in profile.notes.lower()


def test_all_losses(tmp_path):
    """All pnl_dollars <= 0 → avg_win_dollars==0.0, notes warn about no wins."""
    sessions = {
        "sess1": [
            _trade_row(trade_id="t1", pnl_dollars=-100.0, pnl_r=-1.0),
            _trade_row(trade_id="t2", pnl_dollars=-200.0, pnl_r=-2.0),
            _trade_row(trade_id="t3", pnl_dollars=-50.0, pnl_r=-0.5),
        ],
    }
    run_dir = _make_run_dir(tmp_path, sessions)
    bridge = ReplayProfileBridge(str(run_dir), min_trade_count=1)
    profile = bridge.build_profile()

    assert profile.win_rate == pytest.approx(0.0, abs=1e-6)
    assert profile.avg_win_dollars == 0.0
    assert profile.avg_win_r == 0.0
    assert "no winning trades" in profile.notes.lower() or "no win" in profile.notes.lower()


def test_empty_trades(tmp_path):
    """No trades.csv files at all → zeroed profile with warning."""
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    # Create sessions dir but no CSV files
    (run_dir / "sessions" / "sess_empty").mkdir(parents=True)

    bridge = ReplayProfileBridge(str(run_dir))
    profile = bridge.build_profile()

    assert profile.sample_size_trades == 0
    assert profile.sample_size_sessions == 0
    assert profile.win_rate == 0.0
    assert "WARNING" in profile.notes


# ═══════════════════════════════════════════════════════════════════════
#  Aggregate-first bridge tests (new mc_profile.json output)
# ═══════════════════════════════════════════════════════════════════════


def _write_aggregate_csv(run_dir, rows: list[str]) -> None:
    """Write aggregate_trades.csv at the run root."""
    csv_path = run_dir / "aggregate_trades.csv"
    csv_path.write_text(
        "\n".join([CSV_HEADER] + rows) + "\n",
        encoding="utf-8",
    )


def test_aggregate_trades_preferred(tmp_path):
    """When aggregate_trades.csv exists, it is used over session scanning."""
    run_dir = _make_run_dir(
        tmp_path,
        sessions={
            "sess1": [_trade_row(trade_id="t1", pnl_dollars=100.0, pnl_r=1.0)],
        },
    )

    # Write aggregate_trades.csv with 3 rows (more than the 1 in sessions)
    _write_aggregate_csv(run_dir, [
        _trade_row(trade_id="a1", pnl_dollars=200.0, pnl_r=2.0),
        _trade_row(trade_id="a2", pnl_dollars=-50.0, pnl_r=-0.5),
        _trade_row(trade_id="a3", pnl_dollars=150.0, pnl_r=1.5),
    ])

    bridge = ReplayProfileBridge(str(run_dir), min_trade_count=1)
    profile = bridge.build_profile_from_aggregate()

    # Should pick up the 3 aggregate rows, not the 1 session row
    assert profile.sample_size_trades == 3


def test_aggregate_writes_mc_profile_json(tmp_path):
    """build_profile_from_aggregate writes mc_profile.json at run root."""
    run_dir = _make_run_dir(
        tmp_path,
        sessions={
            "sess1": [
                _trade_row(trade_id="t1", pnl_dollars=100.0, pnl_r=1.0),
                _trade_row(trade_id="t2", pnl_dollars=-40.0, pnl_r=-0.4),
            ],
        },
    )

    bridge = ReplayProfileBridge(str(run_dir), min_trade_count=1)
    bridge.build_profile_from_aggregate()

    mc_path = run_dir / "mc_profile.json"
    assert mc_path.is_file()

    data = json.loads(mc_path.read_text())
    assert "trade_count" in data
    assert "win_rate" in data
    assert "quantiles" in data
    assert "expectancy_dollars" in data


def test_fallback_to_sessions_when_no_aggregate(tmp_path):
    """When no aggregate_trades.csv exists, sessions are scanned."""
    sessions = {
        "sess1": [
            _trade_row(trade_id="t1", pnl_dollars=100.0, pnl_r=1.0),
            _trade_row(trade_id="t2", pnl_dollars=-50.0, pnl_r=-0.5),
        ],
        "sess2": [
            _trade_row(trade_id="t3", pnl_dollars=75.0, pnl_r=0.75),
        ],
    }
    run_dir = _make_run_dir(tmp_path, sessions=sessions)

    bridge = ReplayProfileBridge(str(run_dir), min_trade_count=1)
    profile = bridge.build_profile_from_aggregate()

    assert profile.sample_size_trades == 3
    # mc_profile.json should still be written
    assert (run_dir / "mc_profile.json").is_file()


def test_no_trades_aggregate_path(tmp_path):
    """No trades anywhere → zeroed profile, mc_profile.json still written."""
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    (run_dir / "sessions" / "empty").mkdir(parents=True)

    bridge = ReplayProfileBridge(str(run_dir))
    profile = bridge.build_profile_from_aggregate()

    assert profile.sample_size_trades == 0
    assert profile.win_rate == 0.0
    # mc_profile.json should still exist
    assert (run_dir / "mc_profile.json").is_file()
