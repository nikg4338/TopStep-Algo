"""
readiness_check.py — Operational go/no-go checklist for Algorithmic Futures.

Runs fast, local checks to answer: "Can we proceed to paper trading?"

Checks include:
  1) Environment variables for ProjectX connectivity
  2) State/log directory writability
  3) Circuit-breaker baseline sanity
  4) Monte Carlo gate status across core scenarios

Usage:
  python readiness_check.py
  python readiness_check.py --json

Exit codes:
  0 = GO (no FAIL items)
  1 = NO-GO (one or more FAIL items)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pytz
from dotenv import load_dotenv

from config import (
    DATA_PROVIDER,
    LOG_DIR,
    MC_DRAWDOWN_P95_MAX,
    MC_READINESS_MAX_TRADES,
    MC_READINESS_REQUIRED_SCENARIOS,
    MC_READINESS_STREAK_P95_MAX,
    REQUIRE_BROKER_ENV_FOR_READINESS,
    MC_RUIN_THRESHOLD,
    MC_SIMULATIONS,
    MC_TARGET_THRESHOLD,
    STATE_FILE,
)
from execution.circuit_breakers import CircuitBreakers
from regime.regime_state import RegimeState
from risk.monte_carlo import MonteCarloValidator


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


def check_env() -> list[CheckResult]:
    results: list[CheckResult] = []

    # Data-provider key checks
    if DATA_PROVIDER.lower() == "databento":
        db_key = os.getenv("DATABENTO_API_KEY", "")
        if db_key.strip():
            results.append(CheckResult("DATABENTO_API_KEY", "PASS", "set"))
        else:
            results.append(
                CheckResult(
                    "DATABENTO_API_KEY",
                    "FAIL",
                    "missing; required for Databento historical/replay data",
                )
            )

    # Broker checks (optional during data-only bring-up)
    broker_keys = ["PROJECTX_API_KEY", "PROJECTX_BASE_URL", "ACCOUNT_ID"]
    for key in broker_keys:
        value = os.getenv(key, "")
        if value.strip():
            results.append(CheckResult(key, "PASS", "set"))
        else:
            status = "FAIL" if REQUIRE_BROKER_ENV_FOR_READINESS else "WARN"
            results.append(
                CheckResult(
                    key,
                    status,
                    "missing; required for broker execution/paper connectivity",
                )
            )
    return results


def check_paths() -> list[CheckResult]:
    results: list[CheckResult] = []

    log_dir = Path(LOG_DIR)
    state_file = Path(STATE_FILE)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        probe = log_dir / ".readiness_probe"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        results.append(CheckResult("log_dir", "PASS", f"writable: {log_dir}"))
    except Exception as exc:
        results.append(CheckResult("log_dir", "FAIL", f"not writable: {exc}"))

    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        probe = state_file.parent / ".state_probe"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        results.append(
            CheckResult("state_dir", "PASS", f"writable: {state_file.parent}")
        )
    except Exception as exc:
        results.append(CheckResult("state_dir", "FAIL", f"not writable: {exc}"))

    return results


def check_breaker_baseline() -> list[CheckResult]:
    cb = CircuitBreakers(account_mode="combine")
    et = pytz.timezone("US/Eastern")
    in_session_time = et.localize(datetime(2026, 2, 20, 10, 30, 0))
    res = cb.check_all(
        daily_pnl=0.0,
        cumulative_pnl=500.0,
        account_balance=50_000.0,
        account_high_water_mark=50_000.0,
        daily_trade_count=0,
        active_strategy="VWAP",
        current_regime=RegimeState.BALANCED,
        now=in_session_time,
    )
    if res.allowed:
        return [CheckResult("breaker_baseline", "PASS", "normal session permits trading")]
    return [
        CheckResult(
            "breaker_baseline",
            "FAIL",
            f"unexpected breaker blocks under normal inputs: {res.reasons}",
        )
    ]


def check_monte_carlo() -> list[CheckResult]:
    scenarios = [
        ("mc_vwap", 0.55, 30.0, -22.0),
        ("mc_orb", 0.45, 33.0, -22.0),
        ("mc_blended", 0.50, 30.0, -22.0),
    ]

    required_names = {f"mc_{name}" for name in MC_READINESS_REQUIRED_SCENARIOS}
    validator = MonteCarloValidator(
        n_simulations=MC_SIMULATIONS,
        max_trades=MC_READINESS_MAX_TRADES,
    )
    out: list[CheckResult] = []
    for idx, (name, wr, win, loss) in enumerate(scenarios, start=1):
        result = validator.run(win_rate=wr, avg_win=win, avg_loss=loss, seed=idx)
        readiness_pass = (
            result.ruin_probability <= MC_RUIN_THRESHOLD
            and result.target_probability >= MC_TARGET_THRESHOLD
            and result.max_drawdown_p95 <= MC_DRAWDOWN_P95_MAX
            and result.max_losing_streak_p95 <= MC_READINESS_STREAK_P95_MAX
        )

        if readiness_pass:
            out.append(
                CheckResult(
                    name,
                    "PASS",
                    (
                        f"horizon={MC_READINESS_MAX_TRADES} | "
                        f"accepted | ruin={result.ruin_probability:.2%}, "
                        f"target={result.target_probability:.2%}, "
                        f"dd_p95=${result.max_drawdown_p95:,.0f}"
                    ),
                )
            )
        else:
            reasons: list[str] = []
            if result.ruin_probability > MC_RUIN_THRESHOLD:
                reasons.append(
                    f"ruin {result.ruin_probability:.2%} > {MC_RUIN_THRESHOLD:.0%}"
                )
            if result.target_probability < MC_TARGET_THRESHOLD:
                reasons.append(
                    f"target {result.target_probability:.2%} < {MC_TARGET_THRESHOLD:.0%}"
                )
            if result.max_drawdown_p95 > MC_DRAWDOWN_P95_MAX:
                reasons.append(
                    f"dd_p95 ${result.max_drawdown_p95:,.0f} > ${MC_DRAWDOWN_P95_MAX:,.0f}"
                )
            if result.max_losing_streak_p95 > MC_READINESS_STREAK_P95_MAX:
                reasons.append(
                    f"streak_p95 {result.max_losing_streak_p95:.0f} > {MC_READINESS_STREAK_P95_MAX}"
                )

            status = "FAIL" if name in required_names else "WARN"
            out.append(
                CheckResult(
                    name,
                    status,
                    (
                        f"horizon={MC_READINESS_MAX_TRADES} | "
                        + "; ".join(reasons)
                    ),
                )
            )
    return out


def render_text(results: list[CheckResult]) -> str:
    lines: list[str] = []
    lines.append("Algorithmic Futures Readiness Check")
    lines.append("=" * 36)

    for r in results:
        icon = "✅" if r.status == "PASS" else ("⚠️" if r.status == "WARN" else "❌")
        lines.append(f"{icon} {r.name:<20} {r.status:<4} {r.detail}")

    has_fail = any(r.status == "FAIL" for r in results)
    outcome = "NO-GO" if has_fail else "GO"
    lines.append("-" * 36)
    lines.append(f"Outcome: {outcome}")
    return "\n".join(lines)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Operational readiness checklist")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    results: list[CheckResult] = []
    results.extend(check_env())
    results.extend(check_paths())
    results.extend(check_breaker_baseline())
    results.extend(check_monte_carlo())

    has_fail = any(r.status == "FAIL" for r in results)

    if args.json:
        payload = {
            "outcome": "NO-GO" if has_fail else "GO",
            "results": [asdict(r) for r in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(results))

    return 1 if has_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
