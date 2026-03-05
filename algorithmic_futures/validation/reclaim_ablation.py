"""Compare reclaim ON/OFF validation runs with a fixed metric set.

Usage:
    python -m validation.reclaim_ablation \
      --run-on artifacts/validation_runs/pilot_20d_YYYYMMDD_HHMMSS \
      --run-off artifacts/validation_runs/pilot_20d_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sum_candidates(run_dir: Path) -> int:
    total = 0
    sessions = run_dir / "sessions"
    if not sessions.is_dir():
        return 0
    for summary_path in sessions.glob("*/session_summary.json"):
        payload = _load_json(summary_path)
        gate_funnel = payload.get("gate_funnel", {}) or {}
        total += int(gate_funnel.get("candidates_total", 0) or 0)
    return total


def _extract(run_dir: Path) -> dict[str, Any]:
    aggregate = _load_json(run_dir / "aggregate_metrics.json")
    mc_base = _load_json(run_dir / "mc_results.json")

    sessions_total = int(aggregate.get("sessions_total", 0) or 0)
    trade_count = int(aggregate.get("trade_count_total", 0) or 0)
    trades_per_day = (trade_count / sessions_total) if sessions_total > 0 else 0.0

    return {
        "run_id": run_dir.name,
        "candidate_count": _sum_candidates(run_dir),
        "trades_per_day": round(trades_per_day, 4),
        "avg_r": float(aggregate.get("avg_r", 0.0) or 0.0),
        "win_rate": float(aggregate.get("win_rate", 0.0) or 0.0),
        "avg_win_r": float(aggregate.get("avg_win_r", 0.0) or 0.0),
        "avg_loss_r": float(aggregate.get("avg_loss_r", 0.0) or 0.0),
        "mae_p50": float(aggregate.get("mae_p50", 0.0) or 0.0),
        "mae_p90": float(aggregate.get("mae_p90", 0.0) or 0.0),
        "mae_p95": float(aggregate.get("mae_p95", 0.0) or 0.0),
        "early_stop_rate": float(aggregate.get("early_stop_rate", 0.0) or 0.0),
        "dd_p95": float(mc_base.get("dd_p95", 0.0) or 0.0),
        "daily_loss_breach_rate": float(mc_base.get("p_daily_loss_breach", 0.0) or 0.0),
    }


def _print_row(label: str, data: dict[str, Any]) -> None:
    print(
        f"{label:>8} | "
        f"candidates={data['candidate_count']:>4} | "
        f"trades/day={data['trades_per_day']:>5.2f} | "
        f"avg_r={data['avg_r']:>7.4f} | "
        f"win={data['win_rate']*100:>5.1f}% | "
        f"avg_win={data['avg_win_r']:>6.3f} | "
        f"avg_loss={data['avg_loss_r']:>7.3f} | "
        f"mae(p50/p90/p95)=({data['mae_p50']:.2f}/{data['mae_p90']:.2f}/{data['mae_p95']:.2f}) | "
        f"early_stop={data['early_stop_rate']*100:>5.1f}% | "
        f"dd_p95=${data['dd_p95']:>7.0f} | "
        f"daily_breach={data['daily_loss_breach_rate']*100:>5.1f}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare reclaim ON/OFF validation runs")
    parser.add_argument("--run-on", required=True, help="Validation run dir for reclaim ON")
    parser.add_argument("--run-off", required=True, help="Validation run dir for reclaim OFF")
    args = parser.parse_args()

    on = _extract(Path(args.run_on))
    off = _extract(Path(args.run_off))

    print("\nReclaim Ablation Comparison")
    print("=" * 120)
    _print_row("R1-ON", on)
    _print_row("R0-OFF", off)
    print("=" * 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
