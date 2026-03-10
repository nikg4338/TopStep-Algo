"""Utilities for ORB false-positive autopsy research.

This module builds research datasets from existing validation artifacts without
changing frozen strategy behavior.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from validation.candidate_openfix import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    ensure_report_dir,
    load_run_dir,
    summarize_run,
    write_csv,
    write_markdown,
)


NUMERIC_FEATURES = (
    "route_confidence",
    "opening_range_width",
    "atr",
    "width_atr",
    "impulse",
    "persistence",
    "session_pnl",
    "baseline_session_pnl",
    "delta_vs_baseline",
    "cumulative_equity",
    "cumulative_drawdown",
)


@dataclass(frozen=True)
class WindowSpec:
    label: str
    candidate_run: str
    baseline_run: str | None = None
    preset_name: str = "mainline_combine_v1_1_allocator_openfix"
    baseline_preset_name: str = "mainline_combine_v1"


def _parse_mapping(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Mappings must use LABEL=VALUE, got: {item}")
        label, value = item.split("=", 1)
        mapping[label.strip()] = value.strip()
    return mapping


def parse_window_specs(
    candidate_windows: list[str],
    baseline_windows: list[str] | None = None,
    *,
    preset_name: str = "mainline_combine_v1_1_allocator_openfix",
    baseline_preset_name: str = "mainline_combine_v1",
) -> list[WindowSpec]:
    baseline_map = _parse_mapping(baseline_windows or [])
    specs = []
    for item in candidate_windows:
        if "=" not in item:
            raise ValueError(f"Candidate window must use LABEL=RUN_ID, got: {item}")
        label, run_id = item.split("=", 1)
        label = label.strip()
        specs.append(
            WindowSpec(
                label=label,
                candidate_run=run_id.strip(),
                baseline_run=baseline_map.get(label),
                preset_name=preset_name,
                baseline_preset_name=baseline_preset_name,
            )
        )
    return specs


def _to_float(value: Any, default: float | str = "") -> float | str:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int | str = "") -> int | str:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def assign_orb_label(
    *,
    route: str,
    session_pnl: float,
    baseline_session_pnl: float | None,
    false_positive: bool | str,
) -> str:
    if route != "orb":
        return "unknown"
    if false_positive is True:
        return "bad_orb"
    if baseline_session_pnl is not None:
        delta = session_pnl - float(baseline_session_pnl)
        if delta > 0:
            return "good_orb"
        if delta < 0:
            return "bad_orb"
        return "neutral_orb"
    if session_pnl > 0:
        return "good_orb"
    if session_pnl < 0:
        return "bad_orb"
    return "neutral_orb"


def _build_session_feature_map(candidate_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    feature_map = {
        row.get("session_id", ""): row
        for row in candidate_summary.get("allocator_rows", [])
        if row.get("session_id")
    }
    return feature_map


def build_orb_autopsy_rows(
    candidate_summary: dict[str, Any],
    *,
    window_label: str,
    candidate_run_id: str,
    baseline_summary: dict[str, Any] | None = None,
    baseline_run_id: str | None = None,
    preset_name: str = "mainline_combine_v1_1_allocator_openfix",
) -> list[dict[str, Any]]:
    feature_map = _build_session_feature_map(candidate_summary)
    baseline_pnl = baseline_summary.get("session_pnl", {}) if baseline_summary else {}
    baseline_routes = baseline_summary.get("per_session_route", {}) if baseline_summary else {}
    cumulative_equity = 0.0
    peak_equity = 0.0
    cumulative_state: dict[str, tuple[float, float]] = {}
    for session_id in sorted(candidate_summary.get("session_pnl", {}).keys()):
        pnl = float(candidate_summary.get("session_pnl", {}).get(session_id, 0.0))
        cumulative_equity = round(cumulative_equity + pnl, 2)
        peak_equity = max(peak_equity, cumulative_equity)
        cumulative_state[session_id] = (cumulative_equity, round(peak_equity - cumulative_equity, 2))

    rows: list[dict[str, Any]] = []
    for session_id, route in sorted(candidate_summary.get("per_session_route", {}).items()):
        if route != "orb":
            continue
        alloc = feature_map.get(session_id, {})
        session_pnl = round(float(candidate_summary.get("session_pnl", {}).get(session_id, 0.0)), 2)
        baseline_session_pnl_raw = baseline_pnl.get(session_id)
        baseline_session_pnl = None if baseline_session_pnl_raw is None else round(float(baseline_session_pnl_raw), 2)
        delta = ""
        false_positive = ""
        if baseline_session_pnl is not None:
            delta = round(session_pnl - baseline_session_pnl, 2)
            false_positive = delta < 0
        label = assign_orb_label(
            route=route,
            session_pnl=session_pnl,
            baseline_session_pnl=baseline_session_pnl,
            false_positive=false_positive,
        )
        cumulative_equity_at_session, cumulative_drawdown = cumulative_state.get(session_id, ("", ""))
        rows.append(
            {
                "date": alloc.get("date", session_id.replace("session_", "")),
                "session_id": session_id,
                "source_run_id": candidate_run_id,
                "window_label": window_label,
                "pack_id": candidate_summary.get("pack_id", ""),
                "preset_name": preset_name,
                "route": route,
                "route_confidence": _to_float(alloc.get("confidence_score", "")),
                "opening_range_width": _to_float(alloc.get("opening_range_width", "")),
                "atr": _to_float(alloc.get("atr", "")),
                "width_atr": _to_float(alloc.get("width_atr", "")),
                "impulse": _to_float(alloc.get("impulse", "")),
                "persistence": _to_int(alloc.get("persistence", "")),
                "close_location": alloc.get("close_location", ""),
                "one_sidedness": _to_float(alloc.get("one_sidedness", "")),
                "breakout_direction": alloc.get("breakout_direction", ""),
                "breakout_persistence": alloc.get("breakout_persistence", ""),
                "trigger_width": alloc.get("trigger_width", ""),
                "trigger_impulse": alloc.get("trigger_impulse", ""),
                "trigger_persist": alloc.get("trigger_persist", ""),
                "notes": alloc.get("notes", ""),
                "session_pnl": session_pnl,
                "win_flag": session_pnl > 0,
                "cumulative_equity": cumulative_equity_at_session,
                "cumulative_drawdown": cumulative_drawdown,
                "baseline_available": baseline_summary is not None,
                "baseline_run_id": baseline_run_id or "",
                "baseline_route": baseline_routes.get(session_id, "") if baseline_summary else "",
                "baseline_session_pnl": baseline_session_pnl if baseline_session_pnl is not None else "",
                "delta_vs_baseline": delta,
                "false_positive": false_positive,
                "label": label,
            }
        )
    return rows


def build_dataset_for_windows(
    specs: list[WindowSpec],
    *,
    artifacts_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        candidate_dir = load_run_dir(spec.candidate_run, artifacts_root)
        candidate_summary = summarize_run(candidate_dir)
        candidate_summary["preset_name"] = spec.preset_name
        baseline_summary = None
        baseline_run_id = None
        if spec.baseline_run:
            baseline_dir = load_run_dir(spec.baseline_run, artifacts_root)
            baseline_summary = summarize_run(baseline_dir)
            baseline_summary["preset_name"] = spec.baseline_preset_name
            baseline_run_id = baseline_summary["run_id"]
        rows.extend(
            build_orb_autopsy_rows(
                candidate_summary,
                window_label=spec.label,
                candidate_run_id=candidate_summary["run_id"],
                baseline_summary=baseline_summary,
                baseline_run_id=baseline_run_id,
                preset_name=spec.preset_name,
            )
        )
    return rows


def _numeric_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(r[field]) for r in rows if r.get(field) not in ("", None)]
    if not values:
        return {"count": 0, "mean": "", "median": "", "min": "", "max": ""}
    return {
        "count": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def summarize_orb_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(row.get("label", "unknown")) for row in rows)
    window_counts = Counter(str(row.get("window_label", "unknown")) for row in rows)
    false_positive_count = sum(1 for row in rows if _boolish(row.get("false_positive")))
    baseline_available_count = sum(1 for row in rows if _boolish(row.get("baseline_available")))
    return {
        "rows": len(rows),
        "label_counts": dict(label_counts),
        "window_counts": dict(window_counts),
        "false_positive_count": false_positive_count,
        "false_positive_rate": round(false_positive_count / len(rows), 4) if rows else 0.0,
        "baseline_available_count": baseline_available_count,
    }


def grouped_numeric_summary(
    rows: list[dict[str, Any]],
    *,
    group_field: str = "label",
    numeric_fields: tuple[str, ...] = NUMERIC_FEATURES,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field, "unknown"))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        for field in numeric_fields:
            stats = _numeric_stats(group_rows, field)
            summary_rows.append(
                {
                    "group": group,
                    "feature": field,
                    **stats,
                }
            )
    return summary_rows


def grouped_categorical_summary(
    rows: list[dict[str, Any]],
    *,
    group_field: str = "label",
    category_fields: tuple[str, ...] = ("close_location", "breakout_direction", "window_label"),
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field, "unknown"))].append(row)
    for group, group_rows in sorted(grouped.items()):
        for field in category_fields:
            counts = Counter(str(row.get(field, "") or "") for row in group_rows)
            for value, count in counts.most_common():
                summary_rows.append(
                    {
                        "group": group,
                        "feature": field,
                        "value": value,
                        "count": count,
                        "share": round(count / len(group_rows), 4) if group_rows else 0.0,
                    }
                )
    return summary_rows


def strongest_numeric_discriminators(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    good_rows = [row for row in rows if row.get("label") == "good_orb"]
    bad_rows = [row for row in rows if row.get("label") == "bad_orb"]
    discriminators: list[dict[str, Any]] = []
    if not good_rows or not bad_rows:
        return discriminators
    for field in ("route_confidence", "opening_range_width", "atr", "width_atr", "impulse", "persistence", "one_sidedness"):
        good_vals = [float(row[field]) for row in good_rows if row.get(field) not in ("", None)]
        bad_vals = [float(row[field]) for row in bad_rows if row.get(field) not in ("", None)]
        if not good_vals or not bad_vals:
            continue
        good_median = median(good_vals)
        bad_median = median(bad_vals)
        discriminators.append(
            {
                "feature": field,
                "good_median": round(good_median, 4),
                "bad_median": round(bad_median, 4),
                "median_gap": round(good_median - bad_median, 4),
                "abs_median_gap": round(abs(good_median - bad_median), 4),
            }
        )
    return sorted(discriminators, key=lambda row: row["abs_median_gap"], reverse=True)


def candidate_failure_hypotheses(rows: list[dict[str, Any]]) -> list[str]:
    hypotheses: list[str] = []
    discriminators = strongest_numeric_discriminators(rows)
    top = {row["feature"]: row for row in discriminators[:4]}
    if "persistence" in top:
        if top["persistence"]["bad_median"] < top["persistence"]["good_median"]:
            hypotheses.append("Bad ORB sessions show weaker persistence than good ORB sessions, suggesting early move exhaustion.")
        elif top["persistence"]["bad_median"] > top["persistence"]["good_median"]:
            hypotheses.append("Bad ORB sessions still show persistence, which suggests the current persistence signal alone is not selective enough.")
    if "width_atr" in top:
        if top["width_atr"]["bad_median"] > top["width_atr"]["good_median"]:
            hypotheses.append("Bad ORB sessions tend to have wider opening ranges relative to ATR, consistent with noisy expansion or gap-and-fade behavior.")
        elif top["width_atr"]["bad_median"] < top["width_atr"]["good_median"]:
            hypotheses.append("Bad ORB sessions occur even with narrower width/ATR than good ORB sessions, so width alone is not a sufficient guardrail.")
    if "impulse" in top:
        if top["impulse"]["bad_median"] < top["impulse"]["good_median"]:
            hypotheses.append("Bad ORB sessions often have lower directional impulse than good ORB sessions, suggesting continuation quality is weaker even when routed to ORB.")
        elif top["impulse"]["bad_median"] > top["impulse"]["good_median"]:
            hypotheses.append("Bad ORB sessions often show larger early impulse than good ORB sessions, consistent with opening move exhaustion being mistaken for continuation.")
    if "atr" in top:
        if top["atr"]["bad_median"] < top["atr"]["good_median"]:
            hypotheses.append("Bad ORB sessions cluster in lower-ATR contexts than good ORB sessions, suggesting the allocator may be over-trusting directional opens in thinner volatility regimes.")
        elif top["atr"]["bad_median"] > top["atr"]["good_median"]:
            hypotheses.append("Bad ORB sessions cluster in higher-ATR contexts than good ORB sessions, suggesting raw volatility may be overpowering the current selectivity checks.")
    false_positive_rows = [row for row in rows if _boolish(row.get("false_positive"))]
    if false_positive_rows:
        window_counts = Counter(str(row.get("window_label", "unknown")) for row in false_positive_rows)
        window, count = window_counts.most_common(1)[0]
        hypotheses.append(f"False-positive ORB behavior clusters in {window} ({count} session(s)), reinforcing a regime-sensitive failure mode rather than a universal defect.")
    if not hypotheses:
        hypotheses.append("Dataset is too small or too mixed to surface a strong separator; gather more labeled ORB sessions before changing selectivity rules.")
    return hypotheses


def write_orb_autopsy_artifacts(
    rows: list[dict[str, Any]],
    *,
    label: str,
    output_root: Path = DEFAULT_REPORT_ROOT,
) -> Path:
    report_dir = ensure_report_dir(label, output_root)
    dataset_summary = summarize_orb_dataset(rows)
    numeric_summary = grouped_numeric_summary(rows)
    categorical_summary = grouped_categorical_summary(rows)
    write_csv(report_dir / "orb_autopsy_dataset.csv", rows)
    write_csv(report_dir / "orb_autopsy_numeric_summary.csv", numeric_summary)
    write_csv(report_dir / "orb_autopsy_categorical_summary.csv", categorical_summary)
    lines = [
        "# ORB Autopsy Dataset",
        "",
        f"- ORB rows: {dataset_summary['rows']}",
        f"- False-positive rows: {dataset_summary['false_positive_count']} ({dataset_summary['false_positive_rate']:.2%})",
        f"- Windows: {dataset_summary['window_counts']}",
        f"- Labels: {dataset_summary['label_counts']}",
        "",
        "## Hypotheses",
    ]
    lines.extend([f"- {item}" for item in candidate_failure_hypotheses(rows)])
    write_markdown(report_dir / "summary.md", lines)
    return report_dir
