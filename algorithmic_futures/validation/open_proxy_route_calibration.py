"""Route-quality calibration for open_proxy_v1 persistence and medium-impulse boundaries."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import config
from validation.candidate_openfix import build_runner_kwargs_from_preset, summarize_run
from validation.validation_pack import ValidationPackRunner, load_pack

DEFAULT_ARTIFACT_ROOT = Path(config.VALIDATION_ARTIFACTS_ROOT)
DEFAULT_REPORT_ROOT = Path("artifacts/candidate_reports")


def _format_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class RouteCalibrationCandidate:
    label: str
    open_proxy_persist_bars: int
    orb_selectivity_min_persistence_in_low_atr: int
    orb_selectivity_min_persistence_when_high_impulse: int
    medium_impulse_min_atr: float
    medium_impulse_max_atr: float
    medium_impulse_min: float
    medium_impulse_max: float
    medium_impulse_min_persistence: int


def build_candidate_grid(
    *,
    persist_bars: list[int],
    low_atr_persistences: list[int],
    high_impulse_persistences: list[int],
    medium_impulse_min_atrs: list[float],
    medium_impulse_max_atrs: list[float],
    medium_impulse_mins: list[float],
    medium_impulse_maxs: list[float],
    medium_impulse_min_persistences: list[int],
) -> list[RouteCalibrationCandidate]:
    candidates: list[RouteCalibrationCandidate] = []
    for persist, low_atr_p, high_impulse_p, min_atr, max_atr, min_impulse, max_impulse, mid_persist in product(
        persist_bars,
        low_atr_persistences,
        high_impulse_persistences,
        medium_impulse_min_atrs,
        medium_impulse_max_atrs,
        medium_impulse_mins,
        medium_impulse_maxs,
        medium_impulse_min_persistences,
    ):
        if max_impulse <= min_impulse or max_atr <= min_atr:
            continue
        label = (
            f"op_p{int(persist)}"
            f"_low{int(low_atr_p)}"
            f"_hi{int(high_impulse_p)}"
            f"_m{_format_token(min_impulse)}-{_format_token(max_impulse)}"
            f"_atr{_format_token(min_atr)}-{_format_token(max_atr)}"
            f"_mp{int(mid_persist)}"
        )
        candidates.append(
            RouteCalibrationCandidate(
                label=label,
                open_proxy_persist_bars=max(0, int(persist)),
                orb_selectivity_min_persistence_in_low_atr=max(0, int(low_atr_p)),
                orb_selectivity_min_persistence_when_high_impulse=max(0, int(high_impulse_p)),
                medium_impulse_min_atr=float(min_atr),
                medium_impulse_max_atr=float(max_atr),
                medium_impulse_min=float(min_impulse),
                medium_impulse_max=float(max_impulse),
                medium_impulse_min_persistence=max(0, int(mid_persist)),
            )
        )
    return candidates


def derive_route_quality_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    allocator_rows = summary.get("allocator_rows") or []
    orb_rows = [row for row in allocator_rows if str(row.get("route", "")).lower() == "orb"]
    mr_rows = [row for row in allocator_rows if str(row.get("route", "")).lower() == "mr"]

    def _session_pnl(row: dict[str, Any]) -> float:
        return _safe_float(row.get("session_pnl_dollars"))

    def _confidence(row: dict[str, Any]) -> float:
        return _safe_float(row.get("confidence_score"))

    false_positive_orb_count = sum(1 for row in orb_rows if _session_pnl(row) <= 0.0)
    orb_win_count = sum(1 for row in orb_rows if _session_pnl(row) > 0.0)
    orb_confidences = [_confidence(row) for row in orb_rows]
    orb_route_rate = len(orb_rows) / len(allocator_rows) if allocator_rows else 0.0
    orb_win_rate = orb_win_count / len(orb_rows) if orb_rows else 0.0
    false_positive_orb_rate = false_positive_orb_count / len(orb_rows) if orb_rows else 0.0
    avg_orb_session_pnl = sum(_session_pnl(row) for row in orb_rows) / len(orb_rows) if orb_rows else 0.0
    avg_mr_session_pnl = sum(_session_pnl(row) for row in mr_rows) / len(mr_rows) if mr_rows else 0.0
    confidence_mean = sum(orb_confidences) / len(orb_confidences) if orb_confidences else 0.0

    if _safe_float(summary.get("dd_p95")) > float(config.PROMOTION_MC_MAX_DD_P95) or _safe_float(summary.get("ruin_probability")) > float(config.MC_RUIN_THRESHOLD):
        route_quality_status = "degrading"
    elif false_positive_orb_rate > 0.45 or _safe_float(summary.get("final_equity")) < 0.0:
        route_quality_status = "watch"
    else:
        route_quality_status = "stable"

    return {
        "orb_route_rate": round(orb_route_rate, 6),
        "orb_win_rate": round(orb_win_rate, 6),
        "false_positive_orb_count": false_positive_orb_count,
        "false_positive_orb_rate": round(false_positive_orb_rate, 6),
        "avg_orb_session_pnl": round(avg_orb_session_pnl, 6),
        "avg_mr_session_pnl": round(avg_mr_session_pnl, 6),
        "orb_confidence_mean": round(confidence_mean, 6),
        "route_quality_status": route_quality_status,
    }


def summarize_route_candidate(run_dir: Path, candidate: RouteCalibrationCandidate | None, reference_preset: str) -> dict[str, Any]:
    summary = summarize_run(run_dir)
    route_metrics = derive_route_quality_metrics(summary)
    return {
        "label": candidate.label if candidate is not None else reference_preset,
        "run_id": summary.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "candidate": asdict(candidate) if candidate is not None else None,
        "reference_preset": reference_preset,
        "final_equity": round(_safe_float(summary.get("final_equity")), 6),
        "total_trades": int(summary.get("total_trades", 0) or 0),
        "orb_routed_sessions": int(summary.get("orb_routed_sessions", 0) or 0),
        "target_probability": round(_safe_float(summary.get("target_probability")), 6),
        "ruin_probability": round(_safe_float(summary.get("ruin_probability")), 6),
        "dd_p95": round(_safe_float(summary.get("dd_p95")), 6),
        "avg_r": round(_safe_float(summary.get("avg_r")), 6),
        **route_metrics,
    }


def _route_classification(reference: dict[str, Any], candidate: dict[str, Any]) -> str:
    false_positive_delta = candidate["false_positive_orb_rate"] - reference["false_positive_orb_rate"]
    dd_delta = candidate["dd_p95"] - reference["dd_p95"]
    target_delta = candidate["target_probability"] - reference["target_probability"]
    ruin_delta = candidate["ruin_probability"] - reference["ruin_probability"]
    orb_win_delta = candidate["orb_win_rate"] - reference["orb_win_rate"]
    if false_positive_delta < 0.0 and dd_delta <= 0.0 and target_delta >= 0.0 and ruin_delta <= 0.0:
        return "route_quality_progress"
    if false_positive_delta <= 0.0 and dd_delta <= 0.0 and ruin_delta <= 0.0:
        return "engineering_progress"
    if target_delta > 0.0 or orb_win_delta > 0.0:
        return "mixed"
    return "route_regression"


def build_route_tightening_ranking(reference: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for row in rows:
        ranked = dict(row)
        ranked["dd_p95_delta"] = round(row["dd_p95"] - reference["dd_p95"], 6)
        ranked["target_probability_delta"] = round(row["target_probability"] - reference["target_probability"], 6)
        ranked["ruin_probability_delta"] = round(row["ruin_probability"] - reference["ruin_probability"], 6)
        ranked["false_positive_orb_rate_delta"] = round(row["false_positive_orb_rate"] - reference["false_positive_orb_rate"], 6)
        ranked["orb_win_rate_delta"] = round(row["orb_win_rate"] - reference["orb_win_rate"], 6)
        ranked["classification"] = _route_classification(reference, ranked)
        ranking.append(ranked)

    classification_order = {
        "route_quality_progress": 0,
        "engineering_progress": 1,
        "mixed": 2,
        "route_regression": 3,
    }
    ranking.sort(
        key=lambda row: (
            classification_order.get(row["classification"], 99),
            row["false_positive_orb_rate"],
            row["dd_p95"],
            -row["target_probability"],
            row["ruin_probability"],
            -row["orb_win_rate"],
            -row["final_equity"],
        )
    )
    for idx, row in enumerate(ranking, 1):
        row["route_rank"] = idx
    return ranking


def run_open_proxy_route_calibration(
    *,
    pack_name: str,
    reference_preset: str,
    artifacts_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_root: Path = DEFAULT_REPORT_ROOT,
    continue_on_error: bool = True,
    persist_bars: list[int],
    low_atr_persistences: list[int],
    high_impulse_persistences: list[int],
    medium_impulse_min_atrs: list[float],
    medium_impulse_max_atrs: list[float],
    medium_impulse_mins: list[float],
    medium_impulse_maxs: list[float],
    medium_impulse_min_persistences: list[int],
) -> dict[str, Any]:
    pack = load_pack(pack_name)
    candidates = build_candidate_grid(
        persist_bars=persist_bars,
        low_atr_persistences=low_atr_persistences,
        high_impulse_persistences=high_impulse_persistences,
        medium_impulse_min_atrs=medium_impulse_min_atrs,
        medium_impulse_max_atrs=medium_impulse_max_atrs,
        medium_impulse_mins=medium_impulse_mins,
        medium_impulse_maxs=medium_impulse_maxs,
        medium_impulse_min_persistences=medium_impulse_min_persistences,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_root / f"open_proxy_route_calibration_{pack_name}_{ts}"
    report_dir.mkdir(parents=True, exist_ok=True)

    reference_kwargs = build_runner_kwargs_from_preset(reference_preset, str(artifacts_root))
    if reference_kwargs.get("allocator_policy") != "open_proxy_v1":
        raise ValueError(f"Reference preset '{reference_preset}' must use open_proxy_v1 allocator")
    reference_kwargs["continue_on_error"] = continue_on_error
    reference_kwargs["batch_fast_mode"] = True

    reference_runner = ValidationPackRunner(pack, **reference_kwargs)
    reference_manifest = reference_runner.run()
    reference_run_dir = artifacts_root / reference_manifest.run_id
    reference_summary = summarize_route_candidate(reference_run_dir, None, reference_preset)

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        kwargs = dict(reference_kwargs)
        kwargs.update(
            {
                "alloc_openproxy_persist_bars": candidate.open_proxy_persist_bars,
                "alloc_openproxy_enable_orb_selectivity_refinement": True,
                "alloc_openproxy_min_persistence_in_low_atr": candidate.orb_selectivity_min_persistence_in_low_atr,
                "alloc_openproxy_min_persistence_when_high_impulse": candidate.orb_selectivity_min_persistence_when_high_impulse,
                "alloc_openproxy_medium_impulse_decay_filter_enabled": True,
                "alloc_openproxy_medium_impulse_min_atr": candidate.medium_impulse_min_atr,
                "alloc_openproxy_medium_impulse_max_atr": candidate.medium_impulse_max_atr,
                "alloc_openproxy_medium_impulse_min": candidate.medium_impulse_min,
                "alloc_openproxy_medium_impulse_max": candidate.medium_impulse_max,
                "alloc_openproxy_medium_impulse_min_persistence": candidate.medium_impulse_min_persistence,
            }
        )
        manifest = ValidationPackRunner(pack, **kwargs).run()
        run_dir = artifacts_root / manifest.run_id
        row = summarize_route_candidate(run_dir, candidate, reference_preset)
        rows.append(row)
        print(
            f"[{candidate.label}] orb_rate={row['orb_route_rate']:.2%} "
            f"orb_win={row['orb_win_rate']:.2%} fp_orb={row['false_positive_orb_rate']:.2%} "
            f"p_target={row['target_probability']:.2%} dd_p95={row['dd_p95']:.0f}"
        )

    ranking = build_route_tightening_ranking(reference_summary, rows)
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "pack_name": pack_name,
        "reference_preset": reference_preset,
        "reference_run_id": reference_summary["run_id"],
        "reference_summary": reference_summary,
        "candidate_count": len(ranking),
        "search_space": {
            "persist_bars": persist_bars,
            "low_atr_persistences": low_atr_persistences,
            "high_impulse_persistences": high_impulse_persistences,
            "medium_impulse_min_atrs": medium_impulse_min_atrs,
            "medium_impulse_max_atrs": medium_impulse_max_atrs,
            "medium_impulse_mins": medium_impulse_mins,
            "medium_impulse_maxs": medium_impulse_maxs,
            "medium_impulse_min_persistences": medium_impulse_min_persistences,
        },
        "best_candidate": ranking[0] if ranking else None,
        "rows": ranking,
        "output_dir": str(report_dir),
    }

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = report_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "route_rank",
            "label",
            "classification",
            "orb_route_rate",
            "orb_win_rate",
            "false_positive_orb_rate",
            "target_probability",
            "ruin_probability",
            "dd_p95",
            "final_equity",
            "dd_p95_delta",
            "target_probability_delta",
            "ruin_probability_delta",
            "false_positive_orb_rate_delta",
            "orb_win_rate_delta",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranking:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    markdown_lines = [
        "# Open Proxy Route Calibration",
        "",
        f"- Pack: {pack_name}",
        f"- Reference preset: {reference_preset}",
        f"- Reference run: {reference_summary['run_id']}",
        "",
        "## Ranking",
        "",
        "| Rank | Label | Class | ORB rate | ORB win | FP ORB | P(target) | P(ruin) | DD p95 | ΔP(target) | ΔFP ORB |",
        "|------|-------|-------|----------|---------|--------|-----------|---------|--------|------------|---------|",
    ]
    for row in ranking:
        markdown_lines.append(
            "| "
            f"{row['route_rank']} | {row['label']} | {row['classification']} | "
            f"{100.0 * row['orb_route_rate']:.2f}% | {100.0 * row['orb_win_rate']:.2f}% | {100.0 * row['false_positive_orb_rate']:.2f}% | "
            f"{100.0 * row['target_probability']:.2f}% | {100.0 * row['ruin_probability']:.2f}% | {row['dd_p95']:.0f} | "
            f"{100.0 * row['target_probability_delta']:.2f}% | {100.0 * row['false_positive_orb_rate_delta']:.2f}% |"
        )
    (report_dir / "summary.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return summary