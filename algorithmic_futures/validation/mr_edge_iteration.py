"""Narrow MR edge iteration around first-outside and entry-quality levers.

This focuses on the only MR formation lever that moved the recent pilot data:
first-outside. It keeps reclaim mode and other dead dimensions fixed while
varying sigma entry and soft range impulse to test whether the incremental edge
survives tighter entry-quality tuning.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

from validation.mr_approval_calibration import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    MRCalibrationCandidate,
    summarize_candidate,
)
from validation.scorecard import ScorecardAggregator
from validation.validation_pack import ValidationPackRunner, load_pack


def _mode_enabled(mode: str) -> bool:
    return mode == "on"


def _format_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_candidate_grid(
    *,
    sigma_entries: list[float],
    soft_range_impulse_ks: list[float],
    cooldown_bars: list[int],
    first_outside_modes: list[str],
    dedupe_modes: list[str],
    reclaim_mode: str,
    attempt_cap_mode: str,
    regime_mode: str,
) -> list[MRCalibrationCandidate]:
    candidates: list[MRCalibrationCandidate] = []
    for sigma_entry, soft_k, cooldown, first_outside, dedupe_mode in product(
        sigma_entries,
        soft_range_impulse_ks,
        cooldown_bars,
        first_outside_modes,
        dedupe_modes,
    ):
        label = (
            f"mr_edge_sigma{_format_token(sigma_entry)}"
            f"_k{_format_token(soft_k)}"
            f"_fo{1 if _mode_enabled(first_outside) else 0}"
            f"_ded{1 if _mode_enabled(dedupe_mode) else 0}"
            f"_cd{max(0, int(cooldown))}"
        )
        candidates.append(
            MRCalibrationCandidate(
                label=label,
                mr_sigma_entry=max(0.1, float(sigma_entry)),
                mr_reclaim_mode=reclaim_mode,
                mr_cooldown_bars=max(0, int(cooldown)),
                mr_first_outside_enabled=_mode_enabled(first_outside),
                mr_dedupe_enabled=_mode_enabled(dedupe_mode),
                mr_attempt_cap_enabled=_mode_enabled(attempt_cap_mode),
                mr_regime_enabled=_mode_enabled(regime_mode),
                mr_soft_range_impulse_k=max(0.0, float(soft_k)),
            )
        )
    return candidates


def _default_reference_label(rows: list[dict[str, Any]]) -> str:
    preferred = "mr_edge_sigma1p3_k1p2_fo1_ded1_cd1"
    labels = {str(row.get("label", "")) for row in rows}
    if preferred in labels:
        return preferred
    return str(rows[0].get("label", ""))


def _edge_classification(reference: dict[str, Any], candidate: dict[str, Any]) -> str:
    expectancy_delta = _safe_float(candidate.get("expectancy_delta"))
    target_delta = _safe_float(candidate.get("target_probability_delta"))
    ruin_delta = _safe_float(candidate.get("ruin_probability_delta"))
    dd_delta = _safe_float(candidate.get("dd_p95_delta"))
    if expectancy_delta > 0.0 and target_delta > 0.0 and ruin_delta <= 0.0 and dd_delta <= 0.0:
        return "edge_progress"
    if expectancy_delta > 0.0 or target_delta > 0.0:
        return "mixed"
    if ruin_delta < 0.0 or dd_delta < 0.0:
        return "risk_only"
    return "edge_regression"


def build_edge_ranking(
    rows: list[dict[str, Any]],
    *,
    reference_label: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not rows:
        return None, []

    by_label = {str(row.get("label", "")): row for row in rows}
    resolved_reference = reference_label if reference_label in by_label else _default_reference_label(rows)
    reference = dict(by_label[resolved_reference])

    ranking: list[dict[str, Any]] = []
    for row in rows:
        if row.get("label") == resolved_reference:
            continue
        ranked = dict(row)
        ranked["reference_label"] = resolved_reference
        ranked["expectancy_delta"] = round(_safe_float(row.get("expectancy_r")) - _safe_float(reference.get("expectancy_r")), 6)
        ranked["target_probability_delta"] = round(_safe_float(row.get("p_target_before_ruin")) - _safe_float(reference.get("p_target_before_ruin")), 6)
        ranked["ruin_probability_delta"] = round(_safe_float(row.get("p_ruin")) - _safe_float(reference.get("p_ruin")), 6)
        ranked["dd_p95_delta"] = round(_safe_float(row.get("dd_p95")) - _safe_float(reference.get("dd_p95")), 6)
        ranked["trade_count_delta"] = int(row.get("trade_count_total", 0) or 0) - int(reference.get("trade_count_total", 0) or 0)
        ranked["edge_classification"] = _edge_classification(reference, ranked)
        ranking.append(ranked)

    classification_order = {
        "edge_progress": 0,
        "mixed": 1,
        "risk_only": 2,
        "edge_regression": 3,
    }
    ranking.sort(
        key=lambda row: (
            classification_order.get(str(row.get("edge_classification", "")), 99),
            -_safe_float(row.get("target_probability_delta")),
            -_safe_float(row.get("expectancy_delta")),
            _safe_float(row.get("ruin_probability_delta")),
            _safe_float(row.get("dd_p95_delta")),
            -int(row.get("trade_count_delta", 0) or 0),
        )
    )
    for idx, row in enumerate(ranking, 1):
        row["edge_rank"] = idx
    return reference, ranking


def run_mr_edge_iteration(
    *,
    pack_name: str,
    artifacts_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_root: Path = DEFAULT_REPORT_ROOT,
    continue_on_error: bool = True,
    sigma_entries: list[float],
    soft_range_impulse_ks: list[float],
    cooldown_bars: list[int],
    first_outside_modes: list[str],
    dedupe_modes: list[str],
    reclaim_mode: str = "off",
    attempt_cap_mode: str = "on",
    regime_mode: str = "on",
    engine_mode: str = "both",
    allocator_policy: str = "open_proxy_v1",
    reference_label: str | None = None,
) -> dict[str, Any]:
    pack = load_pack(pack_name)
    candidates = build_candidate_grid(
        sigma_entries=sigma_entries,
        soft_range_impulse_ks=soft_range_impulse_ks,
        cooldown_bars=cooldown_bars,
        first_outside_modes=first_outside_modes,
        dedupe_modes=dedupe_modes,
        reclaim_mode=reclaim_mode,
        attempt_cap_mode=attempt_cap_mode,
        regime_mode=regime_mode,
    )

    rows: list[dict[str, Any]] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_root / f"mr_edge_iteration_{pack_name}_{ts}"
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
            f"[{candidate.label}] avg_r={row['avg_r']:.3f} "
            f"p_target={row['p_target_before_ruin']:.2%} "
            f"p_ruin={row['p_ruin']:.2%} "
            f"dd_p95={row['dd_p95']:.0f}"
        )

    reference_row, ranking = build_edge_ranking(rows, reference_label=reference_label)
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "pack_name": pack_name,
        "artifacts_root": str(artifacts_root),
        "engine_mode": engine_mode,
        "allocator_policy": allocator_policy,
        "candidate_count": len(rows),
        "search_space": {
            "sigma_entries": sigma_entries,
            "soft_range_impulse_ks": soft_range_impulse_ks,
            "cooldown_bars": cooldown_bars,
            "first_outside_modes": first_outside_modes,
            "dedupe_modes": dedupe_modes,
            "reclaim_mode": reclaim_mode,
            "attempt_cap_mode": attempt_cap_mode,
            "regime_mode": regime_mode,
        },
        "reference_label": reference_row.get("label") if reference_row else None,
        "reference_row": reference_row,
        "best_candidate": ranking[0] if ranking else None,
        "rows": rows,
        "edge_ranking": ranking,
        "output_dir": str(report_dir),
    }

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = report_dir / "edge_ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "edge_rank",
            "label",
            "edge_classification",
            "avg_r",
            "expectancy_r",
            "p_target_before_ruin",
            "p_ruin",
            "dd_p95",
            "trade_count_total",
            "expectancy_delta",
            "target_probability_delta",
            "ruin_probability_delta",
            "dd_p95_delta",
            "trade_count_delta",
            "failed_checks",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranking:
            writer.writerow(
                {
                    "edge_rank": row.get("edge_rank", 0),
                    "label": row.get("label", ""),
                    "edge_classification": row.get("edge_classification", ""),
                    "avg_r": row.get("avg_r", 0.0),
                    "expectancy_r": row.get("expectancy_r", 0.0),
                    "p_target_before_ruin": row.get("p_target_before_ruin", 0.0),
                    "p_ruin": row.get("p_ruin", 0.0),
                    "dd_p95": row.get("dd_p95", 0.0),
                    "trade_count_total": row.get("trade_count_total", 0),
                    "expectancy_delta": row.get("expectancy_delta", 0.0),
                    "target_probability_delta": row.get("target_probability_delta", 0.0),
                    "ruin_probability_delta": row.get("ruin_probability_delta", 0.0),
                    "dd_p95_delta": row.get("dd_p95_delta", 0.0),
                    "trade_count_delta": row.get("trade_count_delta", 0),
                    "failed_checks": ",".join(row.get("failed_checks", [])),
                }
            )

    markdown_lines = [
        "# MR Edge Iteration",
        "",
        f"- Pack: {pack_name}",
        f"- Engine mode: {engine_mode}",
        f"- Allocator policy: {allocator_policy}",
        f"- Reference: {summary['reference_label'] or '-'}",
        f"- Candidates: {len(rows)}",
        "",
        "## Edge Ranking",
        "",
        "| Rank | Label | Class | Avg R | P(target) | P(ruin) | DD p95 | dAvgR | dP(target) | dP(ruin) | dDD | Trades | dTrades | Failed checks |",
        "|------|-------|-------|-------|-----------|---------|--------|-------|------------|----------|-----|--------|---------|---------------|",
    ]
    for row in ranking:
        markdown_lines.append(
            "| "
            f"{row.get('edge_rank', 0)} | {row.get('label', '')} | {row.get('edge_classification', '')} | "
            f"{_safe_float(row.get('avg_r')):.3f} | {100.0 * _safe_float(row.get('p_target_before_ruin')):.2f}% | "
            f"{100.0 * _safe_float(row.get('p_ruin')):.2f}% | {_safe_float(row.get('dd_p95')):.0f} | "
            f"{_safe_float(row.get('expectancy_delta')):.3f} | {100.0 * _safe_float(row.get('target_probability_delta')):.2f}% | "
            f"{100.0 * _safe_float(row.get('ruin_probability_delta')):.2f}% | {_safe_float(row.get('dd_p95_delta')):.0f} | "
            f"{int(row.get('trade_count_total', 0) or 0)} | {int(row.get('trade_count_delta', 0) or 0)} | "
            f"{', '.join(row.get('failed_checks', [])) or '-'} |"
        )
    (report_dir / "edge_ranking.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return summary