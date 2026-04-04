"""Promotion-aware MR approval calibration runner.

Runs a focused grid of MR parameter overrides through the existing
validation pack pipeline, generates scorecards, and ranks candidates by
promotion-style constraints plus approval-rate and expectancy tradeoffs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import config
from validation.scorecard import ScorecardAggregator
from validation.validation_pack import ValidationPackRunner, load_pack

DEFAULT_ARTIFACT_ROOT = Path(config.VALIDATION_ARTIFACTS_ROOT)
DEFAULT_REPORT_ROOT = Path("artifacts/candidate_reports")


def _mode_enabled(mode: str) -> bool:
    return mode == "on"


def _format_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sum_drop_ledger(run_dir: Path) -> dict[str, int]:
    totals: dict[str, int] = {
        "bars_evaluated": 0,
        "eligible_session_bars": 0,
        "z_cross_events": 0,
        "dedupe_rejects": 0,
        "attempt_limit_rejects": 0,
        "cooldown_rejects": 0,
        "in_position_rejects": 0,
        "regime_rejects": 0,
        "spread_liquidity_rejects": 0,
        "candidates_formed": 0,
        "orders_submitted": 0,
        "fills": 0,
        "trades": 0,
    }
    for summary_path in sorted((run_dir / "sessions").glob("*/session_summary.json")):
        summary = _load_json(summary_path)
        gate = summary.get("gate_funnel") or {}
        drop = gate.get("drop_ledger") or {}
        for key in totals:
            totals[key] += int(drop.get(key, gate.get(key, 0)) or 0)
    return totals


@dataclass(frozen=True)
class MRCalibrationCandidate:
    label: str
    mr_sigma_entry: float
    mr_reclaim_mode: str
    mr_cooldown_bars: int
    mr_first_outside_enabled: bool
    mr_dedupe_enabled: bool
    mr_attempt_cap_enabled: bool
    mr_regime_enabled: bool
    mr_soft_range_impulse_k: float


def build_candidate_grid(
    *,
    sigma_entries: list[float],
    reclaim_modes: list[str],
    cooldown_bars: list[int],
    first_outside_modes: list[str],
    dedupe_modes: list[str],
    attempt_cap_modes: list[str],
    regime_modes: list[str],
    soft_range_impulse_k: float,
) -> list[MRCalibrationCandidate]:
    candidates: list[MRCalibrationCandidate] = []
    for sigma_entry, reclaim_mode, cooldown, first_outside, dedupe_mode, attempt_cap, regime_mode in product(
        sigma_entries,
        reclaim_modes,
        cooldown_bars,
        first_outside_modes,
        dedupe_modes,
        attempt_cap_modes,
        regime_modes,
    ):
        label = (
            f"mr_sigma{_format_token(sigma_entry)}"
            f"_{reclaim_mode}"
            f"_cd{max(0, int(cooldown))}"
            f"_fo{1 if _mode_enabled(first_outside) else 0}"
            f"_ded{1 if _mode_enabled(dedupe_mode) else 0}"
            f"_cap{1 if _mode_enabled(attempt_cap) else 0}"
            f"_reg{1 if _mode_enabled(regime_mode) else 0}"
        )
        candidates.append(
            MRCalibrationCandidate(
                label=label,
                mr_sigma_entry=max(0.1, float(sigma_entry)),
                mr_reclaim_mode=reclaim_mode,
                mr_cooldown_bars=max(0, int(cooldown)),
                mr_first_outside_enabled=_mode_enabled(first_outside),
                mr_dedupe_enabled=_mode_enabled(dedupe_mode),
                mr_attempt_cap_enabled=_mode_enabled(attempt_cap),
                mr_regime_enabled=_mode_enabled(regime_mode),
                mr_soft_range_impulse_k=max(0.0, float(soft_range_impulse_k)),
            )
        )
    return candidates


def summarize_candidate(
    run_dir: Path,
    candidate: MRCalibrationCandidate,
    scorecard: dict[str, Any],
) -> dict[str, Any]:
    manifest = _load_json(run_dir / "manifest.json")
    aggregate_metrics = _load_json(run_dir / "aggregate_metrics.json")
    mc_results = _load_json(run_dir / "mc_results.json")
    gate_result = _load_json(run_dir / "gate_result.json")

    sessions = manifest.get("sessions", [])
    succeeded = sum(1 for session in sessions if session.get("success", False))
    total_sessions = len(sessions)
    session_success_rate = succeeded / total_sessions if total_sessions else 0.0

    trade_metrics = scorecard.get("trade_metrics") or {}
    approval_rate = _safe_float(scorecard.get("aggregate_approval_rate"), 0.0)
    expectancy_r = _safe_float(trade_metrics.get("expectancy_r"), _safe_float(aggregate_metrics.get("avg_r"), 0.0))
    p_target = _safe_float(mc_results.get("p_target_before_ruin"), 0.0)
    p_ruin = _safe_float(mc_results.get("p_ruin"), 1.0)
    dd_p95 = _safe_float(mc_results.get("dd_p95"), 1_000_000.0)
    losing_streak_p95 = _safe_float(mc_results.get("losing_streak_p95"), 1_000_000.0)
    trade_count_total = int(aggregate_metrics.get("trade_count_total", 0) or 0)

    checks = {
        "session_success_rate": session_success_rate >= float(config.PROMOTION_MIN_SESSION_SUCCESS_RATE),
        "approval_rate_min": approval_rate >= float(config.PROMOTION_APPROVAL_RATE_MIN),
        "approval_rate_max": approval_rate <= float(config.PROMOTION_APPROVAL_RATE_MAX),
        "expectancy_r": expectancy_r >= float(config.PROMOTION_MIN_EXPECTANCY),
        "mc_target_prob": p_target >= float(config.MC_TARGET_THRESHOLD),
        "mc_ruin_prob": p_ruin <= float(config.MC_RUIN_THRESHOLD),
        "mc_dd_p95": dd_p95 <= float(config.PROMOTION_MC_MAX_DD_P95),
        "mc_losing_streak_p95": losing_streak_p95 < float(config.MC_LOSING_STREAK_P95_MAX),
        "min_trade_count": trade_count_total >= int(config.MC_PROFILE_MIN_TRADE_COUNT),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    drop_ledger = _sum_drop_ledger(run_dir)

    return {
        "label": candidate.label,
        "run_id": manifest.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "candidate": asdict(candidate),
        "gate_pass": bool(gate_result.get("overall_pass", False)),
        "promotion_like_pass": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "approval_rate": round(approval_rate, 6),
        "expectancy_r": round(expectancy_r, 6),
        "win_rate": round(_safe_float(trade_metrics.get("win_rate"), _safe_float(aggregate_metrics.get("win_rate"), 0.0)), 6),
        "avg_r": round(_safe_float(aggregate_metrics.get("avg_r"), 0.0), 6),
        "trade_count_total": trade_count_total,
        "trades_per_session_mean": round(_safe_float(aggregate_metrics.get("trades_per_session_mean"), 0.0), 6),
        "session_success_rate": round(session_success_rate, 6),
        "p_target_before_ruin": round(p_target, 6),
        "p_ruin": round(p_ruin, 6),
        "dd_p95": round(dd_p95, 6),
        "losing_streak_p95": round(losing_streak_p95, 6),
        "readiness": bool(aggregate_metrics.get("readiness", False)),
        "drop_ledger_total": drop_ledger,
    }


def rank_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [dict(row) for row in rows]
    ranked.sort(
        key=lambda row: (
            not row["promotion_like_pass"],
            len(row["failed_checks"]),
            -row["p_target_before_ruin"],
            -row["expectancy_r"],
            row["p_ruin"],
            row["dd_p95"],
            row["losing_streak_p95"],
            -row["approval_rate"],
            -row["trade_count_total"],
        )
    )
    for idx, row in enumerate(ranked, 1):
        row["rank"] = idx
    return ranked


def run_mr_approval_calibration(
    *,
    pack_name: str,
    artifacts_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_root: Path = DEFAULT_REPORT_ROOT,
    continue_on_error: bool = True,
    sigma_entries: list[float],
    reclaim_modes: list[str],
    cooldown_bars: list[int],
    first_outside_modes: list[str],
    dedupe_modes: list[str],
    attempt_cap_modes: list[str],
    regime_modes: list[str],
    soft_range_impulse_k: float,
    engine_mode: str = "mr",
    allocator_policy: str = "none",
) -> dict[str, Any]:
    pack = load_pack(pack_name)
    candidates = build_candidate_grid(
        sigma_entries=sigma_entries,
        reclaim_modes=reclaim_modes,
        cooldown_bars=cooldown_bars,
        first_outside_modes=first_outside_modes,
        dedupe_modes=dedupe_modes,
        attempt_cap_modes=attempt_cap_modes,
        regime_modes=regime_modes,
        soft_range_impulse_k=soft_range_impulse_k,
    )
    rows: list[dict[str, Any]] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_root / f"mr_approval_calibration_{pack_name}_{ts}"
    report_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        runner = ValidationPackRunner(
            pack,
            artifacts_root=str(artifacts_root),
            continue_on_error=continue_on_error,
            batch_fast_mode=True,
            mr_reclaim_mode=candidate.mr_reclaim_mode,
            mr_sigma_entry=candidate.mr_sigma_entry,
            mr_soft_impulse_k=candidate.mr_soft_range_impulse_k,
            mr_dedupe_enabled=candidate.mr_dedupe_enabled,
            mr_attempt_cap_enabled=candidate.mr_attempt_cap_enabled,
            mr_cooldown_bars=candidate.mr_cooldown_bars,
            mr_first_outside_enabled=candidate.mr_first_outside_enabled,
            mr_regime_enabled=candidate.mr_regime_enabled,
            engine_mode=engine_mode,
            allocator_policy=allocator_policy,
            orb_enabled=False,
        )
        manifest = runner.run()
        run_dir = artifacts_root / manifest.run_id
        scorecard = ScorecardAggregator(str(run_dir)).generate()
        row = summarize_candidate(run_dir, candidate, scorecard)
        rows.append(row)
        print(
            f"[{candidate.label}] approval={row['approval_rate']:.2%} "
            f"avg_r={row['avg_r']:.3f} p_target={row['p_target_before_ruin']:.2%} "
            f"pass={row['promotion_like_pass']}"
        )

    ranked = rank_candidate_rows(rows)
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "pack_name": pack_name,
        "artifacts_root": str(artifacts_root),
        "engine_mode": engine_mode,
        "allocator_policy": allocator_policy,
        "candidate_count": len(ranked),
        "search_space": {
            "sigma_entries": sigma_entries,
            "reclaim_modes": reclaim_modes,
            "cooldown_bars": cooldown_bars,
            "first_outside_modes": first_outside_modes,
            "dedupe_modes": dedupe_modes,
            "attempt_cap_modes": attempt_cap_modes,
            "regime_modes": regime_modes,
            "soft_range_impulse_k": soft_range_impulse_k,
        },
        "best_candidate": ranked[0] if ranked else None,
        "rows": ranked,
    }

    json_path = report_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = report_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "rank",
            "label",
            "run_id",
            "promotion_like_pass",
            "gate_pass",
            "failed_checks",
            "approval_rate",
            "expectancy_r",
            "win_rate",
            "avg_r",
            "trade_count_total",
            "session_success_rate",
            "p_target_before_ruin",
            "p_ruin",
            "dd_p95",
            "losing_streak_p95",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "label": row["label"],
                    "run_id": row["run_id"],
                    "promotion_like_pass": row["promotion_like_pass"],
                    "gate_pass": row["gate_pass"],
                    "failed_checks": ",".join(row["failed_checks"]),
                    "approval_rate": row["approval_rate"],
                    "expectancy_r": row["expectancy_r"],
                    "win_rate": row["win_rate"],
                    "avg_r": row["avg_r"],
                    "trade_count_total": row["trade_count_total"],
                    "session_success_rate": row["session_success_rate"],
                    "p_target_before_ruin": row["p_target_before_ruin"],
                    "p_ruin": row["p_ruin"],
                    "dd_p95": row["dd_p95"],
                    "losing_streak_p95": row["losing_streak_p95"],
                }
            )

    markdown_lines = [
        "# MR Approval Calibration",
        "",
        f"- Pack: {pack_name}",
        f"- Engine mode: {engine_mode}",
        f"- Allocator policy: {allocator_policy}",
        f"- Candidates: {len(ranked)}",
        "",
        "## Ranking",
        "",
        "| Rank | Label | Promotion-like pass | Approval | Avg R | P(target) | P(ruin) | DD p95 | Failed checks |",
        "|------|-------|----------------------|----------|-------|-----------|---------|--------|---------------|",
    ]
    for row in ranked:
        markdown_lines.append(
            "| "
            f"{row['rank']} | {row['label']} | {'PASS' if row['promotion_like_pass'] else 'FAIL'} | "
            f"{row['approval_rate']:.2%} | {row['avg_r']:.3f} | {row['p_target_before_ruin']:.2%} | "
            f"{row['p_ruin']:.2%} | {row['dd_p95']:.0f} | {', '.join(row['failed_checks']) or '-'} |"
        )
    (report_dir / "summary.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    summary["output_dir"] = str(report_dir)
    return summary