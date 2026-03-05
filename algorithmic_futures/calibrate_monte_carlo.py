"""
calibrate_monte_carlo.py — Monte Carlo horizon calibration helper.

Sweeps candidate max-trade horizons and reports the first horizon that
satisfies readiness gates for required scenarios.

Usage:
  python calibrate_monte_carlo.py
  python calibrate_monte_carlo.py --json

Outputs:
  - Console report
  - JSON artifact in logs/monte_carlo_calibration_YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from config import (
    LOG_DIR,
    MC_CALIBRATION_CANDIDATE_TRADES,
    MC_DRAWDOWN_P95_MAX,
    MC_READINESS_REQUIRED_SCENARIOS,
    MC_READINESS_STREAK_P95_MAX,
    MC_RUIN_THRESHOLD,
    MC_SIMULATIONS,
    MC_TARGET_THRESHOLD,
)
from risk.monte_carlo import MonteCarloValidator


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    max_trades: int
    ruin_probability: float
    target_probability: float
    drawdown_p95: float
    streak_p95: float
    readiness_pass: bool


def _scenario_defs() -> list[tuple[str, float, float, float]]:
    return [
        ("vwap", 0.55, 30.0, -22.0),
        ("orb", 0.45, 33.0, -22.0),
        ("blended", 0.50, 30.0, -22.0),
    ]


def run_calibration() -> dict:
    required = set(MC_READINESS_REQUIRED_SCENARIOS)
    rows: list[ScenarioResult] = []

    for max_trades in MC_CALIBRATION_CANDIDATE_TRADES:
        validator = MonteCarloValidator(
            n_simulations=MC_SIMULATIONS,
            max_trades=max_trades,
        )
        for seed, (name, wr, avg_win, avg_loss) in enumerate(_scenario_defs(), start=1):
            result = validator.run(
                win_rate=wr,
                avg_win=avg_win,
                avg_loss=avg_loss,
                seed=seed,
            )
            readiness_pass = (
                result.ruin_probability <= MC_RUIN_THRESHOLD
                and result.target_probability >= MC_TARGET_THRESHOLD
                and result.max_drawdown_p95 <= MC_DRAWDOWN_P95_MAX
                and result.max_losing_streak_p95 <= MC_READINESS_STREAK_P95_MAX
            )
            rows.append(
                ScenarioResult(
                    scenario=name,
                    max_trades=max_trades,
                    ruin_probability=float(result.ruin_probability),
                    target_probability=float(result.target_probability),
                    drawdown_p95=float(result.max_drawdown_p95),
                    streak_p95=float(result.max_losing_streak_p95),
                    readiness_pass=bool(readiness_pass),
                )
            )

    recommended_horizon: int | None = None
    for max_trades in MC_CALIBRATION_CANDIDATE_TRADES:
        subset = [r for r in rows if r.max_trades == max_trades and r.scenario in required]
        if subset and all(r.readiness_pass for r in subset):
            recommended_horizon = max_trades
            break

    return {
        "generated_at": datetime.now().isoformat(),
        "required_scenarios": sorted(required),
        "thresholds": {
            "ruin_max": MC_RUIN_THRESHOLD,
            "target_min": MC_TARGET_THRESHOLD,
            "drawdown_p95_max": MC_DRAWDOWN_P95_MAX,
            "streak_p95_max": MC_READINESS_STREAK_P95_MAX,
        },
        "candidates": list(MC_CALIBRATION_CANDIDATE_TRADES),
        "recommended_horizon": recommended_horizon,
        "results": [asdict(r) for r in rows],
    }


def render_text(payload: dict) -> str:
    lines: list[str] = []
    lines.append("Monte Carlo Calibration Report")
    lines.append("=" * 30)
    lines.append(
        "Required scenarios: " + ", ".join(payload["required_scenarios"])
    )
    lines.append(
        "Thresholds: ruin<=%.0f%%, target>=%.0f%%, dd_p95<=%.0f, streak_p95<=%d"
        % (
            payload["thresholds"]["ruin_max"] * 100,
            payload["thresholds"]["target_min"] * 100,
            payload["thresholds"]["drawdown_p95_max"],
            payload["thresholds"]["streak_p95_max"],
        )
    )

    rec = payload["recommended_horizon"]
    if rec is None:
        lines.append("Recommended horizon: none found in candidate set")
    else:
        lines.append(f"Recommended horizon: {rec} trades")

    lines.append("-" * 30)
    for max_trades in payload["candidates"]:
        subset = [
            r for r in payload["results"]
            if r["max_trades"] == max_trades and r["scenario"] in payload["required_scenarios"]
        ]
        if not subset:
            continue
        r = subset[0]
        status = "PASS" if r["readiness_pass"] else "FAIL"
        lines.append(
            f"{max_trades:>4} trades | {r['scenario']:<8} {status} "
            f"target={r['target_probability']:.2%} ruin={r['ruin_probability']:.2%} "
            f"dd_p95=${r['drawdown_p95']:.0f} streak_p95={r['streak_p95']:.0f}"
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Monte Carlo trade horizon")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload")
    args = parser.parse_args()

    payload = run_calibration()

    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = log_dir / f"monte_carlo_calibration_{datetime.now().strftime('%Y-%m-%d')}.json"
    out_file.write_text(json.dumps(payload, indent=2))

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(payload))
        print(f"\nSaved calibration artifact: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
