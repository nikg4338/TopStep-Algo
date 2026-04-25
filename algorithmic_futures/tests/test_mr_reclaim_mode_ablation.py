"""
tests/test_mr_reclaim_mode_ablation.py — helper tests for reclaim ablation runner.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.run_mr_reclaim_mode_ablation import (
    ReclaimModeMetrics,
    average_trade_pnl,
    extract_mode_metrics,
    metrics_to_markdown,
    sum_candidate_pool,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sum_candidate_pool_reads_session_summaries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "sessions" / "s1" / "session_summary.json",
        {"gate_funnel": {"candidates_total": 3}},
    )
    _write_json(
        run_dir / "sessions" / "s2" / "session_summary.json",
        {"gate_funnel": {"candidates_total": 7}},
    )

    assert sum_candidate_pool(run_dir) == 10


def test_average_trade_pnl_reads_aggregate_trades(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "aggregate_trades.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pnl_dollars"])
        writer.writeheader()
        writer.writerows([{"pnl_dollars": "10"}, {"pnl_dollars": "-5"}, {"pnl_dollars": "25"}])

    assert average_trade_pnl(run_dir) == pytest.approx(10.0)


def test_extract_mode_metrics_collects_required_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "sessions" / "s1" / "session_summary.json",
        {"gate_funnel": {"candidates_total": 4}},
    )
    _write_json(
        run_dir / "aggregate_metrics.json",
        {
            "trade_count_total": 2,
            "sessions_total": 4,
            "sessions_with_trades": 1,
            "win_rate": 0.5,
            "avg_r": 0.1,
            "avg_win_r": 1.0,
            "avg_loss_r": -0.8,
        },
    )
    _write_json(
        run_dir / "mc_results.json",
        {
            "p_target_before_ruin": 0.62,
            "p_ruin": 0.08,
            "dd_p95": 900,
            "losing_streak_p95": 6,
        },
    )
    _write_json(run_dir / "mc_results_stress_mild.json", {"p_target_before_ruin": 0.52})
    _write_json(run_dir / "mc_results_stress_severe.json", {"p_target_before_ruin": 0.31})
    _write_json(run_dir / "gate_result.json", {"overall_pass": True})
    with (run_dir / "aggregate_trades.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pnl_dollars"])
        writer.writeheader()
        writer.writerows([{"pnl_dollars": "12"}, {"pnl_dollars": "-4"}])

    metrics = extract_mode_metrics("touch", run_dir)

    assert metrics.reclaim_mode == "touch"
    assert metrics.candidate_pool_size == 4
    assert metrics.approved_trades == 2
    assert metrics.trades_per_session == pytest.approx(0.5)
    assert metrics.avg_trade_pnl == pytest.approx(4.0)
    assert metrics.p_target == pytest.approx(0.62)
    assert metrics.stress_severe_target_probability == pytest.approx(0.31)
    assert metrics.promotion_gate_result == "PASS"


def test_metrics_to_markdown_contains_all_modes() -> None:
    rows = [
        ReclaimModeMetrics(
            reclaim_mode="on",
            run_dir="run",
            candidate_pool_size=1,
            approved_trades=1,
            sessions_total=1,
            sessions_with_trades=1,
            trades_per_session=1.0,
            win_rate=1.0,
            avg_r=0.5,
            avg_win_r=0.5,
            avg_loss_r=0.0,
            avg_trade_pnl=10.0,
            p_target=0.7,
            p_ruin=0.1,
            dd_p95=500.0,
            losing_streak_p95=3.0,
            stress_mild_target_probability=0.6,
            stress_severe_target_probability=0.4,
            promotion_gate_result="PASS",
        )
    ]

    markdown = metrics_to_markdown(rows, "Title")

    assert "# Title" in markdown
    assert "| on |" in markdown
    assert "P(Target)" in markdown
    assert "Severe P(Target)" in markdown
