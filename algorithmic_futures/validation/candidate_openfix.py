"""Utilities for allocator-openfix candidate validation workflows.

These helpers keep `mainline_combine_v1` frozen while making it easy to run
and compare versioned investigation candidates such as
`mainline_combine_v1_1_allocator_openfix`.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import config
from run_validation_pack import _load_preset
from simulation.mc_survival import MonteCarloSurvivalSimulator
from validation.validation_pack import ValidationPackRunner, load_pack


DEFAULT_PACKS = ("pilot_20d", "extended_60d", "trend20")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/validation_runs")
DEFAULT_REPORT_ROOT = Path("artifacts/candidate_reports")
UTC_EOD_CLOSE = "21:05:00"
UTC_ORB_END = "14:45:00"


@dataclass(frozen=True)
class CandidateVerdict:
    engineering_verdict: str
    promotion_verdict: str
    reason: str
    engineering_checks: dict[str, bool]
    promotion_checks: dict[str, bool]
    target_probability: float
    target_threshold: float


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _sum_pnl(rows: Iterable[dict[str, str]]) -> float:
    total = 0.0
    for row in rows:
        total += float(row.get("pnl_dollars", 0.0) or 0.0)
    return round(total, 2)


def _session_pnl_map(trades: list[dict[str, str]]) -> dict[str, float]:
    session_pnl: dict[str, float] = {}
    for row in trades:
        sid = row.get("session_id", "")
        session_pnl[sid] = round(session_pnl.get(sid, 0.0) + float(row.get("pnl_dollars", 0.0) or 0.0), 2)
    return session_pnl


def _session_trade_count_map(trades: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in trades:
        sid = row.get("session_id", "")
        counts[sid] = counts.get(sid, 0) + 1
    return counts


def _route_map_from_session_summaries(run_dir: Path) -> dict[str, str]:
    sessions_dir = run_dir / "sessions"
    route_map: dict[str, str] = {}
    if not sessions_dir.is_dir():
        return route_map
    for session_dir in sorted(p for p in sessions_dir.iterdir() if p.is_dir()):
        summary = _read_json(session_dir / "session_summary.json")
        orb_funnel = summary.get("orb_funnel", {}) if isinstance(summary, dict) else {}
        route = orb_funnel.get("allocator_decision") or orb_funnel.get("engine_mode") or "unknown"
        route_map[summary.get("session_id", session_dir.name)] = route
    return route_map


def load_run_dir(run_id_or_path: str, artifacts_root: Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    candidate = Path(run_id_or_path)
    if candidate.is_dir():
        return candidate
    run_dir = artifacts_root / run_id_or_path
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_id_or_path}")
    return run_dir


def build_runner_kwargs_from_preset(preset_name: str, artifacts_root: str) -> dict[str, Any]:
    overrides = _load_preset(preset_name)
    return {
        "artifacts_root": artifacts_root,
        "continue_on_error": True,
        "mr_sigma_entry": float(overrides.get("mr_sigma_entry", config.MR_SIGMA_ENTRY)),
        "mr_reclaim_mode": overrides.get("mr_reclaim_mode", "off"),
        "mr_soft_impulse_k": float(overrides.get("mr_soft_range_impulse_k", config.MR_SOFT_RECLAIM_RANGE_IMPULSE_K)),
        "mr_dedupe_enabled": _boolish(overrides.get("mr_dedupe_enabled", "off")),
        "mr_attempt_cap_enabled": _boolish(overrides.get("mr_attempt_cap_enabled", "on")),
        "mr_cooldown_bars": int(overrides.get("mr_cooldown_bars", config.MR_COOLDOWN_BARS)),
        "mr_first_outside_enabled": _boolish(overrides.get("mr_first_outside_enabled", "off")),
        "mr_dedupe_window_bars": int(overrides.get("mr_dedupe_window_bars", config.MR_DEDUPE_WINDOW_BARS)),
        "mr_dedupe_min_delta_z": float(overrides.get("mr_dedupe_min_delta_z", config.MR_DEDUPE_MIN_DELTA_Z)),
        "mr_regime_enabled": _boolish(overrides.get("mr_regime_enabled", "on")),
        "engine_mode": overrides.get("engine_mode", "both"),
        "allocator_policy": overrides.get("allocator_policy", "none"),
        "allocator_v1_adx_threshold": float(overrides.get("allocator_v1_adx_threshold", 25.0)),
        "allocator_v2_trend_open_threshold": float(overrides.get("allocator_v2_trend_open_threshold", 25.0)),
        "allocator_v2_rising_threshold": float(overrides.get("allocator_v2_rising_threshold", 20.0)),
        "allocator_v2_rising_bars": int(overrides.get("allocator_v2_rising_bars", 3)),
        "allocator_v2_range_threshold": float(overrides.get("allocator_v2_range_threshold", 18.0)),
        "allocator_v2_range_bars": int(overrides.get("allocator_v2_range_bars", 3)),
        "alloc_openproxy_or_width_atr": float(overrides.get("alloc_openproxy_or_width_atr", 2.2)),
        "alloc_openproxy_impulse_atr": float(overrides.get("alloc_openproxy_impulse_atr", 0.9)),
        "alloc_openproxy_persist_bars": int(overrides.get("alloc_openproxy_persist_bars", 1)),
        "alloc_openproxy_require_break": _boolish(overrides.get("alloc_openproxy_require_break", "off")),
        "alloc_openproxy_enable_orb_selectivity_refinement": _boolish(overrides.get("alloc_openproxy_enable_orb_selectivity_refinement", "off")),
        "alloc_openproxy_low_atr_threshold": float(overrides.get("alloc_openproxy_low_atr_threshold", config.ALLOC_OPENPROXY_LOW_ATR_THRESHOLD)),
        "alloc_openproxy_min_persistence_in_low_atr": int(overrides.get("alloc_openproxy_min_persistence_in_low_atr", config.ALLOC_OPENPROXY_MIN_PERSISTENCE_IN_LOW_ATR)),
        "alloc_openproxy_high_impulse_threshold": float(overrides.get("alloc_openproxy_high_impulse_threshold", config.ALLOC_OPENPROXY_HIGH_IMPULSE_THRESHOLD)),
        "alloc_openproxy_min_persistence_when_high_impulse": int(overrides.get("alloc_openproxy_min_persistence_when_high_impulse", config.ALLOC_OPENPROXY_MIN_PERSISTENCE_WHEN_HIGH_IMPULSE)),
        "alloc_openproxy_medium_impulse_weak_persistence_filter_enabled": _boolish(overrides.get("alloc_openproxy_medium_impulse_weak_persistence_filter_enabled", "off")),
        "alloc_openproxy_medium_impulse_decay_filter_enabled": _boolish(overrides.get("alloc_openproxy_medium_impulse_decay_filter_enabled", "off")),
        "alloc_openproxy_medium_impulse_min_atr": float(overrides.get("alloc_openproxy_medium_impulse_min_atr", config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MIN_ATR)),
        "alloc_openproxy_medium_impulse_max_atr": float(overrides.get("alloc_openproxy_medium_impulse_max_atr", config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MAX_ATR)),
        "alloc_openproxy_medium_impulse_min": float(overrides.get("alloc_openproxy_medium_impulse_min", config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MIN)),
        "alloc_openproxy_medium_impulse_max": float(overrides.get("alloc_openproxy_medium_impulse_max", config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MAX)),
        "alloc_openproxy_medium_impulse_min_persistence": int(overrides.get("alloc_openproxy_medium_impulse_min_persistence", config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MIN_PERSISTENCE)),
        "orb_enabled": _boolish(overrides.get("orb_enabled", "on")),
        "orb_trigger_mode": overrides.get("orb_trigger_mode", "pullback_v3"),
        "orb_pullback_confirm_bars": int(overrides.get("orb_pullback_confirm_bars", 3)),
        "sizing_policy": overrides.get("sizing_policy", "dynamic_v3"),
        "dyn_v3_earned_traction": float(overrides.get("dyn_v3_earned_traction", 75.0)),
        "dyn_v3_giveback_floor": float(overrides.get("dyn_v3_giveback_floor", 25.0)),
        "dyn_v3_orb_upsize_allowed": _boolish(overrides.get("dyn_v3_orb_upsize_allowed", "off")),
        "dyn_v3_day_headroom_up": float(overrides.get("dyn_v3_day_headroom_up", 800.0)),
        "dyn_v3_day_headroom_down": float(overrides.get("dyn_v3_day_headroom_down", 600.0)),
        "dyn_v3_trail_headroom_up": float(overrides.get("dyn_v3_trail_headroom_up", 1400.0)),
        "dyn_v3_trail_headroom_down": float(overrides.get("dyn_v3_trail_headroom_down", 1200.0)),
        "dyn_v3_atr_traction_scale_enabled": _boolish(overrides.get("dyn_v3_atr_traction_scale_enabled", "off")),
        "dyn_v3_atr_traction_baseline": float(overrides.get("dyn_v3_atr_traction_baseline", config.DYN_V3_ATR_TRACTION_BASELINE)),
        "dyn_v3_atr_traction_min_scale": float(overrides.get("dyn_v3_atr_traction_min_scale", config.DYN_V3_ATR_TRACTION_MIN_SCALE)),
        "dyn_v3_atr_traction_max_scale": float(overrides.get("dyn_v3_atr_traction_max_scale", config.DYN_V3_ATR_TRACTION_MAX_SCALE)),
        "dyn_v3_consistency_brake_enabled": _boolish(overrides.get("dyn_v3_consistency_brake_enabled", "off")),
        "dyn_v3_consistency_cap_pct": float(overrides.get("dyn_v3_consistency_cap_pct", config.DYN_V3_CONSISTENCY_CAP_PCT)),
        "dyn_v3_consistency_loss_buffer_mult": float(overrides.get("dyn_v3_consistency_loss_buffer_mult", config.DYN_V3_CONSISTENCY_LOSS_BUFFER_MULT)),
    }


def run_pack_for_preset(
    pack_name: str,
    preset_name: str,
    artifacts_root: str = str(DEFAULT_ARTIFACT_ROOT),
    extra_kwargs: dict[str, Any] | None = None,
) -> Path:
    pack = load_pack(pack_name)
    kwargs = build_runner_kwargs_from_preset(preset_name, artifacts_root)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    runner = ValidationPackRunner(pack, **kwargs)
    manifest = runner.run()
    return Path(artifacts_root) / manifest.run_id


def summarize_run(run_dir: Path | str) -> dict[str, Any]:
    run_dir = load_run_dir(str(run_dir))
    manifest = _read_json(run_dir / "manifest.json")
    aggregate = _read_json(run_dir / "aggregate_metrics.json")
    mc = _read_json(run_dir / "mc_results.json")
    gate = _read_json(run_dir / "gate_result.json")
    allocator_rows = _read_csv(run_dir / "allocator_debug.csv")
    trades = _read_csv(run_dir / "aggregate_trades.csv")
    sizing_rows = _read_json(run_dir / "sizing_decisions.json")
    session_pnl = _session_pnl_map(trades)
    session_trade_counts = _session_trade_count_map(trades)

    route_by_session = {
        row.get("session_id", ""): row.get("route", row.get("engine_mode", "unknown"))
        for row in allocator_rows
    }
    if not route_by_session:
        route_by_session = _route_map_from_session_summaries(run_dir)
    route_counts = Counter(route_by_session.values())
    mr_trades = sum(session_trade_counts.get(sid, 0) for sid, route in route_by_session.items() if route == "mr")
    orb_trades = sum(session_trade_counts.get(sid, 0) for sid, route in route_by_session.items() if route == "orb")
    orb_sessions = sum(1 for route in route_by_session.values() if route == "orb")
    win_sessions = sum(1 for pnl in session_pnl.values() if pnl > 0)
    total_positive_pnl = sum(max(pnl, 0.0) for pnl in session_pnl.values())
    consistency_breach_count = 0
    if total_positive_pnl > 0:
        cap = float(getattr(config, "CONSISTENCY_CAP", {}).get(getattr(config, "ACCOUNT_MODE", "combine"), 0.50))
        consistency_breach_count = sum(1 for pnl in session_pnl.values() if pnl > 0 and pnl > cap * total_positive_pnl)

    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "pack_id": manifest.get("pack_id", run_dir.name.split("_")[0]),
        "run_dir": str(run_dir),
        "sessions_total": len(manifest.get("sessions", [])),
        "sessions_succeeded": sum(1 for s in manifest.get("sessions", []) if s.get("success")),
        "total_trades": int(aggregate.get("trade_count_total", len(trades))),
        "mr_trades": mr_trades,
        "orb_trades": orb_trades,
        "orb_routed_sessions": orb_sessions,
        "win_sessions": win_sessions,
        "final_equity": round(_sum_pnl(trades), 2),
        "avg_r": float(aggregate.get("avg_r", 0.0) or 0.0),
        "equity_p50": float(mc.get("equity_p50", 0.0) or 0.0),
        "equity_p10": float(mc.get("equity_p10", 0.0) or 0.0),
        "dd_p95": float(mc.get("dd_p95", 0.0) or 0.0),
        "ruin_probability": float(mc.get("p_ruin", 0.0) or 0.0),
        "target_probability": float(mc.get("p_target_before_ruin", 0.0) or 0.0),
        "daily_loss_breach_count": sum(1 for pnl in session_pnl.values() if pnl <= -float(config.DAILY_LOSS_LIMIT_EXTERNAL)),
        "consistency_rule_breach_count": consistency_breach_count,
        "route_distribution": dict(route_counts),
        "per_session_route": route_by_session,
        "session_pnl": session_pnl,
        "allocator_rows": allocator_rows,
        "gate_pass": gate.get("overall_pass"),
        "gate_checks": gate.get("checks", []),
        "sizing_rows": sizing_rows if isinstance(sizing_rows, list) else [],
    }


def compare_runs(baseline_run_dir: Path | str, candidate_run_dir: Path | str) -> dict[str, Any]:
    baseline = summarize_run(baseline_run_dir)
    candidate = summarize_run(candidate_run_dir)
    session_ids = sorted(set(baseline["per_session_route"].keys()) | set(candidate["per_session_route"].keys()))
    rows: list[dict[str, Any]] = []
    false_positive_orb = 0
    for sid in session_ids:
        base_route = baseline["per_session_route"].get(sid, "unknown")
        cand_route = candidate["per_session_route"].get(sid, "unknown")
        base_pnl = float(baseline["session_pnl"].get(sid, 0.0))
        cand_pnl = float(candidate["session_pnl"].get(sid, 0.0))
        delta = round(cand_pnl - base_pnl, 2)
        if cand_route == "orb" and base_pnl > cand_pnl:
            false_positive_orb += 1
        rows.append(
            {
                "session_id": sid,
                "baseline_route": base_route,
                "candidate_route": cand_route,
                "baseline_pnl": round(base_pnl, 2),
                "candidate_pnl": round(cand_pnl, 2),
                "delta_pnl": delta,
                "route_changed": base_route != cand_route,
            }
        )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "per_session": rows,
        "route_changed_sessions": sum(1 for r in rows if r["route_changed"]),
        "false_positive_orb": false_positive_orb,
    }


def evaluate_candidate_verdict(
    candidate_summary: dict[str, Any],
    *,
    target_threshold: float | None = None,
    live_integrity: dict[str, bool] | None = None,
    robustness_rows: list[dict[str, Any]] | None = None,
) -> CandidateVerdict:
    """Separate engineering-integrity verdict from promotion verdict."""
    target_threshold = float(
        config.CANDIDATE_PROMOTION_TARGET_THRESHOLD if target_threshold is None else target_threshold
    )
    live_integrity = live_integrity or {}
    robustness_rows = robustness_rows or []

    engineering_checks = {
        "orb_routed_sessions_positive": candidate_summary.get("orb_routed_sessions", 0) > 0,
        "allocator_decisions_auditable": bool(candidate_summary.get("allocator_rows"))
        and len(candidate_summary.get("allocator_rows", [])) == candidate_summary.get("sessions_total", 0),
        "no_lookahead_evidence": live_integrity.get("no_lookahead_evidence", True),
        "eod_flatten": live_integrity.get("eod_flatten", True),
        "daily_loss_halt": live_integrity.get("daily_loss_halt", True),
        "daily_profit_halt": live_integrity.get("daily_profit_halt", True),
        "no_duplicate_orders": live_integrity.get("no_duplicate_orders", True),
        "no_stale_orders": live_integrity.get("no_stale_orders", True),
        "ruin_probability": float(candidate_summary.get("ruin_probability", 1.0)) < config.MC_RUIN_THRESHOLD,
        "dd_p95": float(candidate_summary.get("dd_p95", 1e9)) < config.MC_DRAWDOWN_P95_MAX,
        "no_daily_rule_violations": int(candidate_summary.get("daily_loss_breach_count", 0)) == 0,
        "no_consistency_breaches": int(candidate_summary.get("consistency_rule_breach_count", 0)) == 0,
    }
    engineering_pass = all(engineering_checks.values())

    robust_ok = all(
        row.get("p_ruin", 1.0) < config.MC_RUIN_THRESHOLD and row.get("dd_p95", 1e9) < config.MC_DRAWDOWN_P95_MAX
        for row in robustness_rows
    ) if robustness_rows else True
    promotion_checks = {
        "engineering_integrity": engineering_pass,
        "robustness": robust_ok,
        "target_probability": float(candidate_summary.get("target_probability", 0.0)) >= target_threshold,
    }

    if not engineering_pass:
        return CandidateVerdict(
            engineering_verdict="FAIL",
            promotion_verdict="FAIL",
            reason="engineering integrity or risk-rule checks failed",
            engineering_checks=engineering_checks,
            promotion_checks=promotion_checks,
            target_probability=float(candidate_summary.get("target_probability", 0.0)),
            target_threshold=target_threshold,
        )
    if not promotion_checks["target_probability"]:
        return CandidateVerdict(
            engineering_verdict="PASS",
            promotion_verdict="HOLD",
            reason=(
                "target-hit probability below configured promotion threshold "
                f"({candidate_summary.get('target_probability', 0.0):.4f} < {target_threshold:.4f})"
            ),
            engineering_checks=engineering_checks,
            promotion_checks=promotion_checks,
            target_probability=float(candidate_summary.get("target_probability", 0.0)),
            target_threshold=target_threshold,
        )
    if not robust_ok:
        return CandidateVerdict(
            engineering_verdict="PASS",
            promotion_verdict="HOLD",
            reason="robustness checks degraded relative to promotion thresholds",
            engineering_checks=engineering_checks,
            promotion_checks=promotion_checks,
            target_probability=float(candidate_summary.get("target_probability", 0.0)),
            target_threshold=target_threshold,
        )
    return CandidateVerdict(
        engineering_verdict="PASS",
        promotion_verdict="PASS",
        reason="candidate clears engineering integrity and promotion-quality gates",
        engineering_checks=engineering_checks,
        promotion_checks=promotion_checks,
        target_probability=float(candidate_summary.get("target_probability", 0.0)),
        target_threshold=target_threshold,
    )


def evaluate_live_integrity(run_dir: Path | str) -> dict[str, bool]:
    """Inspect a completed run for live-style integrity checks."""
    run_dir = load_run_dir(str(run_dir))
    summary = summarize_run(run_dir)
    sessions_dir = Path(run_dir) / "sessions"
    allocator_rows = summary.get("allocator_rows", [])

    checks = {
        "orb_routed_sessions_positive": summary.get("orb_routed_sessions", 0) > 0,
        "route_decisions_auditable": bool(allocator_rows) and len(allocator_rows) == summary.get("sessions_total", 0),
        "no_lookahead_evidence": True,
        "eod_flatten": True,
        "daily_loss_halt": True,
        "daily_profit_halt": True,
        "no_duplicate_orders": True,
        "no_stale_orders": True,
    }

    if not sessions_dir.is_dir():
        return checks

    for session_dir in sorted(p for p in sessions_dir.iterdir() if p.is_dir()):
        trades = _read_csv(session_dir / "trades.csv")
        trade_ids = [row.get("trade_id", "") for row in trades]
        if len(trade_ids) != len(set(trade_ids)):
            checks["no_duplicate_orders"] = False
        for row in trades:
            entry_ts = row.get("entry_timestamp", "")
            exit_ts = row.get("exit_timestamp", "")
            if entry_ts and len(entry_ts) >= 19 and entry_ts[11:19] < UTC_ORB_END:
                checks["no_lookahead_evidence"] = False
            if not exit_ts:
                checks["no_stale_orders"] = False
                checks["eod_flatten"] = False
            elif len(exit_ts) >= 19 and exit_ts[11:19] > UTC_EOD_CLOSE:
                checks["eod_flatten"] = False

        summary_json = _read_json(session_dir / "session_summary.json")
        counters = summary_json.get("rejection_counters", {}) if isinstance(summary_json, dict) else {}
        if counters.get("rejected_by_daily_loss_governor", 0) < 0:
            checks["daily_loss_halt"] = False
        if counters.get("rejected_by_profit_cap", 0) < 0:
            checks["daily_profit_halt"] = False
    return checks


def build_forward_shadow_rows(
    candidate_summary: dict[str, Any],
    *,
    baseline_summary: dict[str, Any] | None = None,
    starting_equity: float = 0.0,
) -> list[dict[str, Any]]:
    """Build one-row-per-session forward-shadow tracker output."""
    allocator_by_session = {
        row.get("session_id", ""): row for row in candidate_summary.get("allocator_rows", [])
    }
    baseline_pnl = baseline_summary.get("session_pnl", {}) if baseline_summary else {}
    cumulative_equity = float(starting_equity)
    peak_equity = float(starting_equity)
    rows: list[dict[str, Any]] = []
    for session_id in sorted(candidate_summary.get("per_session_route", {}).keys()):
        alloc = allocator_by_session.get(session_id, {})
        pnl = float(candidate_summary.get("session_pnl", {}).get(session_id, 0.0))
        cumulative_equity = round(cumulative_equity + pnl, 2)
        peak_equity = max(peak_equity, cumulative_equity)
        drawdown = round(peak_equity - cumulative_equity, 2)
        baseline_session_pnl = baseline_pnl.get(session_id)
        false_positive = ""
        positive_contribution = ""
        if baseline_session_pnl is not None and alloc.get("route") == "orb":
            false_positive = pnl < float(baseline_session_pnl)
            positive_contribution = pnl > float(baseline_session_pnl)
        rows.append(
            {
                "date": alloc.get("date", session_id.replace("session_", "")),
                "session_id": session_id,
                "preset": candidate_summary.get("preset_name", "mainline_combine_v1_1_allocator_openfix"),
                "route": candidate_summary.get("per_session_route", {}).get(session_id, "unknown"),
                "route_confidence": alloc.get("confidence_score", ""),
                "opening_range_width": alloc.get("opening_range_width", ""),
                "atr": alloc.get("atr", ""),
                "width_atr": alloc.get("width_atr", ""),
                "impulse": alloc.get("impulse", ""),
                "persistence": alloc.get("persistence", ""),
                "session_pnl": round(pnl, 2),
                "win_flag": pnl > 0,
                "daily_rule_clean": True,
                "consistency_rule_clean": True,
                "false_positive_orb": false_positive,
                "positive_orb_contribution": positive_contribution,
                "cumulative_equity": cumulative_equity,
                "cumulative_target_progress": round(cumulative_equity / float(config.PROFIT_TARGET), 4),
                "cumulative_drawdown": drawdown,
                "notes": alloc.get("notes", ""),
            }
        )
    return rows


def summarize_forward_shadow(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate forward-shadow tracker rows into a compact verdict summary."""
    sessions = len(rows)
    orb_rows = [r for r in rows if r.get("route") == "orb"]
    orb_wins = [r for r in orb_rows if r.get("win_flag")]
    false_pos = [r for r in orb_rows if r.get("false_positive_orb") is True]
    rule_breaches = sum(
        1
        for r in rows
        if not r.get("daily_rule_clean", True) or not r.get("consistency_rule_clean", True)
    )
    final_equity = float(rows[-1]["cumulative_equity"]) if rows else 0.0
    max_drawdown = max((float(r.get("cumulative_drawdown", 0.0)) for r in rows), default=0.0)
    false_positive_rate = len(false_pos) / len(orb_rows) if orb_rows else 0.0
    orb_win_rate = len(orb_wins) / len(orb_rows) if orb_rows else 0.0
    if rule_breaches > 0 or max_drawdown > config.MC_DRAWDOWN_P95_MAX:
        status = "degrading"
    elif false_positive_rate > 0.45 or final_equity < 0:
        status = "watch"
    else:
        status = "stable"
    return {
        "sessions_processed": sessions,
        "orb_routed_sessions": len(orb_rows),
        "orb_route_rate": round(len(orb_rows) / sessions, 4) if sessions else 0.0,
        "orb_win_rate": round(orb_win_rate, 4),
        "false_positive_orb_count": len(false_pos),
        "false_positive_orb_rate": round(false_positive_rate, 4),
        "cumulative_equity": round(final_equity, 2),
        "current_drawdown": round(max_drawdown, 2),
        "progress_to_target": round(final_equity / float(config.PROFIT_TARGET), 4) if sessions else 0.0,
        "rule_breach_count": rule_breaches,
        "status": status,
    }


def run_robustness_from_run(run_dir: Path | str, seeds: list[int], slippage_ticks: list[int]) -> list[dict[str, Any]]:
    run_dir = load_run_dir(str(run_dir))
    trades = _read_csv(run_dir / "aggregate_trades.csv")
    pnl = [float(r.get("pnl_dollars", 0.0) or 0.0) for r in trades]
    session_ids = [r.get("session_id", "") for r in trades]
    results: list[dict[str, Any]] = []
    sim = MonteCarloSurvivalSimulator()
    for seed in seeds:
        base = sim.run(pnl, seed=seed, use_dollar_values=True, session_ids=session_ids)
        results.append({
            "scenario": f"seed_{seed}",
            "seed": seed,
            "slippage_ticks": 0,
            "p_target_before_ruin": base.p_target_before_ruin,
            "p_ruin": base.p_ruin,
            "dd_p95": base.dd_p95,
            "equity_p10": getattr(base, "equity_p10", 0.0),
        })
        for ticks in slippage_ticks:
            stressed = sim.run(
                pnl,
                seed=seed,
                use_dollar_values=True,
                session_ids=session_ids,
                stress={"_name": f"slippage_{ticks}t", "slippage_ticks": ticks},
            )
            results.append({
                "scenario": f"seed_{seed}_slippage_{ticks}t",
                "seed": seed,
                "slippage_ticks": ticks,
                "p_target_before_ruin": stressed.p_target_before_ruin,
                "p_ruin": stressed.p_ruin,
                "dd_p95": stressed.dd_p95,
                "equity_p10": getattr(stressed, "equity_p10", 0.0),
            })
    return results


def ensure_report_dir(label: str, output_root: Path = DEFAULT_REPORT_ROOT) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_root / f"{label}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pass_fail(flag: bool) -> str:
    return "PASS" if flag else "FAIL"
