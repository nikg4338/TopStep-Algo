"""Tests for validation/trade_aggregator.py — trade aggregation stage."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from validation.trade_aggregator import aggregate_trades


# ── Fixtures / Helpers ──────────────────────────────────────────────────


def _write_trades_csv(
    session_dir: Path,
    rows: list[dict[str, str]],
) -> Path:
    """Write a trades.csv with given rows.  Empty list → header-only."""
    session_dir.mkdir(parents=True, exist_ok=True)
    csv_path = session_dir / "trades.csv"
    fieldnames = [
        "trade_id", "session_id", "signal_timestamp", "side",
        "entry_timestamp", "entry_price", "stop_price", "target_price",
        "exit_timestamp", "exit_price", "exit_reason",
        "pnl_points", "pnl_dollars", "pnl_r",
        "mae_points", "mfe_points", "hold_minutes", "hold_bars",
        "regime_at_entry", "sigma_band_level",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def _make_trade(
    trade_id: str = "t1",
    session_id: str = "sess_a",
    side: str = "SELL",
    pnl_r: float = 1.0,
    pnl_dollars: float = 20.0,
    pnl_points: float = 4.0,
    exit_reason: str = "target",
    hold_bars: int = 4,
) -> dict[str, str]:
    return {
        "trade_id": trade_id,
        "session_id": session_id,
        "signal_timestamp": "2026-02-20T15:40:00+00:00",
        "side": side,
        "entry_timestamp": "2026-02-20T15:45:00+00:00",
        "entry_price": "6910.0",
        "stop_price": "6927.0",
        "target_price": "6895.0",
        "exit_timestamp": "2026-02-20T16:00:00+00:00",
        "exit_price": "6895.0",
        "exit_reason": exit_reason,
        "pnl_points": str(pnl_points),
        "pnl_dollars": str(pnl_dollars),
        "pnl_r": str(pnl_r),
        "mae_points": "2.0",
        "mfe_points": "15.0",
        "hold_minutes": "15.0",
        "hold_bars": str(hold_bars),
        "regime_at_entry": "range",
        "sigma_band_level": "1.5",
    }


def _build_run_root(
    tmp_path: Path,
    session_trade_counts: dict[str, int],
) -> Path:
    """Create a run_root with sessions and trades.csv files."""
    run_root = tmp_path / "test_run"
    sessions_dir = run_root / "sessions"

    # Write a minimal manifest.json
    manifest = {
        "run_id": "test_run",
        "sessions": [
            {"session_id": sid, "success": True, "category": "trend"}
            for sid in session_trade_counts
        ],
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(json.dumps(manifest))

    for sid, n_trades in session_trade_counts.items():
        sess_dir = sessions_dir / sid
        rows = [
            _make_trade(trade_id=f"{sid}_t{i}", session_id=sid, pnl_r=(1.0 if i % 2 == 0 else -0.5))
            for i in range(n_trades)
        ]
        _write_trades_csv(sess_dir, rows)

    return run_root


# ── Tests ───────────────────────────────────────────────────────────────


class TestTradeAggregator:
    """Unit tests for the trade aggregation stage."""

    def test_basic_aggregation_3_sessions(self, tmp_path: Path) -> None:
        """3 sessions (4 + 8 + 0 rows) → aggregate_trades.csv with 12 rows."""
        run_root = _build_run_root(tmp_path, {
            "sess_a": 4,
            "sess_b": 8,
            "sess_c": 0,
        })
        metrics = aggregate_trades(run_root, min_trade_count=10)

        # aggregate_trades.csv exists with 12 data rows
        agg_csv = run_root / "aggregate_trades.csv"
        assert agg_csv.is_file()
        with agg_csv.open() as fh:
            reader = list(csv.DictReader(fh))
        assert len(reader) == 12

        # Metrics correctness
        assert metrics["trade_count_total"] == 12
        assert metrics["sessions_total"] == 3
        assert metrics["sessions_with_trades"] == 2
        assert metrics["readiness"] is True  # 12 >= 10
        assert "win_rate" in metrics
        assert "losing_streak_max" in metrics
        assert "p5_r" in metrics
        assert "p50_r" in metrics
        assert "p95_r" in metrics
        assert "mae_p50" in metrics
        assert "mae_p90" in metrics
        assert "mae_p95" in metrics
        assert "early_stop_rate" in metrics

    def test_no_error_on_empty_csv(self, tmp_path: Path) -> None:
        """Aggregator handles header-only trades.csv without error."""
        run_root = _build_run_root(tmp_path, {"empty_sess": 0})
        metrics = aggregate_trades(run_root, min_trade_count=1)
        assert metrics["trade_count_total"] == 0
        assert metrics["readiness"] is False
        assert metrics["readiness_reason"] == "insufficient_trades"

    def test_manifest_patched(self, tmp_path: Path) -> None:
        """aggregate artifacts are added to manifest.json."""
        run_root = _build_run_root(tmp_path, {"sess_a": 5})
        aggregate_trades(run_root)
        manifest = json.loads((run_root / "manifest.json").read_text())
        assert "aggregate_artifacts" in manifest
        assert "aggregate_trades_csv" in manifest["aggregate_artifacts"]
        assert "aggregate_metrics_json" in manifest["aggregate_artifacts"]

    def test_aggregate_metrics_json_written(self, tmp_path: Path) -> None:
        """aggregate_metrics.json is created alongside aggregate_trades.csv."""
        run_root = _build_run_root(tmp_path, {"sess_a": 3, "sess_b": 7})
        aggregate_trades(run_root)
        metrics_path = run_root / "aggregate_metrics.json"
        assert metrics_path.is_file()
        data = json.loads(metrics_path.read_text())
        assert data["trade_count_total"] == 10

    def test_readiness_false_below_threshold(self, tmp_path: Path) -> None:
        """readiness=false when trade count is below min_trade_count."""
        run_root = _build_run_root(tmp_path, {"sess_a": 3})
        metrics = aggregate_trades(run_root, min_trade_count=10)
        assert metrics["readiness"] is False
        assert metrics["readiness_reason"] == "insufficient_trades"

    def test_losing_streak_calculation(self, tmp_path: Path) -> None:
        """losing_streak_max correctly counts consecutive losses."""
        run_root = tmp_path / "streak_run"
        sessions_dir = run_root / "sessions" / "s1"
        # All losing trades
        rows = [_make_trade(trade_id=f"t{i}", pnl_r=-0.5) for i in range(5)]
        _write_trades_csv(sessions_dir, rows)
        (run_root / "manifest.json").write_text(json.dumps({
            "run_id": "streak_run", "sessions": [{"session_id": "s1", "success": True}],
        }))
        metrics = aggregate_trades(run_root, min_trade_count=1)
        assert metrics["losing_streak_max"] == 5

    def test_early_stop_rate_uses_stop_within_n_bars(self, tmp_path: Path) -> None:
        """early_stop_rate counts stop exits with hold_bars <= N."""
        run_root = tmp_path / "early_stop_run"
        sessions_dir = run_root / "sessions" / "s1"
        rows = [
            _make_trade(trade_id="t1", session_id="s1", pnl_r=-1.0, exit_reason="stop", hold_bars=2),
            _make_trade(trade_id="t2", session_id="s1", pnl_r=-0.5, exit_reason="stop", hold_bars=5),
            _make_trade(trade_id="t3", session_id="s1", pnl_r=1.0, exit_reason="target", hold_bars=3),
        ]
        _write_trades_csv(sessions_dir, rows)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "manifest.json").write_text(json.dumps({
            "run_id": "early_stop_run", "sessions": [{"session_id": "s1", "success": True}],
        }))

        metrics = aggregate_trades(run_root, min_trade_count=1, early_stop_n_bars=3)
        assert metrics["early_stop_n_bars"] == 3
        assert metrics["early_stop_count"] == 1
        assert metrics["early_stop_rate"] == pytest.approx(1 / 3, abs=1e-4)
