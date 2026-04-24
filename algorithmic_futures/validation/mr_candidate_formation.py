"""Focused MR candidate-formation sweep around first-outside, dedupe, and cooldown.

This builds on the existing MR approval calibration flow but re-ranks
candidates by formation-quality diagnostics so narrow candidate-generation
changes can be inspected without changing sizing or allocator plumbing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from validation.mr_approval_calibration import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    run_mr_approval_calibration,
)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _formation_classification(row: dict[str, Any]) -> str:
    expectancy = _safe_float(row.get("expectancy_r"))
    p_target = _safe_float(row.get("p_target_before_ruin"))
    suppression = _safe_float(row.get("suppression_rate"))
    if expectancy > 0.0 and p_target >= 0.20 and suppression <= 0.25:
        return "formation_progress"
    if suppression > 0.50:
        return "over_suppressed"
    if expectancy > 0.0:
        return "edge_positive"
    return "negative_edge"


def build_candidate_formation_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for row in rows:
        drop = row.get("drop_ledger_total") or {}
        z_cross_events = _safe_int(drop.get("z_cross_events"))
        candidates_formed = _safe_int(drop.get("candidates_formed"))
        cooldown_rejects = _safe_int(drop.get("cooldown_rejects"))
        dedupe_rejects = _safe_int(drop.get("dedupe_rejects"))
        eligible_bars = _safe_int(drop.get("eligible_session_bars"))
        trades = _safe_int(drop.get("trades", row.get("trade_count_total")))

        candidate_yield = trades / candidates_formed if candidates_formed else 0.0
        z_cross_yield = trades / z_cross_events if z_cross_events else 0.0
        eligible_bar_yield = candidates_formed / eligible_bars if eligible_bars else 0.0
        suppression_rate = (cooldown_rejects + dedupe_rejects) / z_cross_events if z_cross_events else 0.0

        ranked_row = dict(row)
        ranked_row.update(
            {
                "candidate_yield": round(candidate_yield, 6),
                "z_cross_yield": round(z_cross_yield, 6),
                "eligible_bar_yield": round(eligible_bar_yield, 6),
                "suppression_rate": round(suppression_rate, 6),
                "formation_classification": "",
            }
        )
        ranked_row["formation_classification"] = _formation_classification(ranked_row)
        ranking.append(ranked_row)

    classification_order = {
        "formation_progress": 0,
        "edge_positive": 1,
        "negative_edge": 2,
        "over_suppressed": 3,
    }
    ranking.sort(
        key=lambda row: (
            classification_order.get(row["formation_classification"], 99),
            -_safe_float(row.get("p_target_before_ruin")),
            -_safe_float(row.get("expectancy_r")),
            -_safe_float(row.get("candidate_yield")),
            _safe_float(row.get("suppression_rate")),
            _safe_float(row.get("p_ruin")),
            _safe_float(row.get("dd_p95")),
            -_safe_int(row.get("trade_count_total")),
        )
    )
    for idx, row in enumerate(ranking, 1):
        row["formation_rank"] = idx
    return ranking


def run_mr_candidate_formation_sweep(
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
    engine_mode: str = "both",
    allocator_policy: str = "open_proxy_v1",
) -> dict[str, Any]:
    summary = run_mr_approval_calibration(
        pack_name=pack_name,
        artifacts_root=artifacts_root,
        output_root=output_root,
        continue_on_error=continue_on_error,
        sigma_entries=sigma_entries,
        reclaim_modes=reclaim_modes,
        cooldown_bars=cooldown_bars,
        first_outside_modes=first_outside_modes,
        dedupe_modes=dedupe_modes,
        attempt_cap_modes=attempt_cap_modes,
        regime_modes=regime_modes,
        soft_range_impulse_k=soft_range_impulse_k,
        engine_mode=engine_mode,
        allocator_policy=allocator_policy,
        report_label="mr_candidate_formation_sweep",
    )

    ranking = build_candidate_formation_ranking(summary.get("rows", []))
    summary["formation_ranking"] = ranking

    report_dir = Path(summary["output_dir"])
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = report_dir / "formation_ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "formation_rank",
            "label",
            "formation_classification",
            "approval_rate",
            "avg_r",
            "p_target_before_ruin",
            "p_ruin",
            "dd_p95",
            "candidate_yield",
            "z_cross_yield",
            "eligible_bar_yield",
            "suppression_rate",
            "trade_count_total",
            "failed_checks",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranking:
            writer.writerow(
                {
                    "formation_rank": row.get("formation_rank", 0),
                    "label": row.get("label", ""),
                    "formation_classification": row.get("formation_classification", ""),
                    "approval_rate": row.get("approval_rate", 0.0),
                    "avg_r": row.get("avg_r", 0.0),
                    "p_target_before_ruin": row.get("p_target_before_ruin", 0.0),
                    "p_ruin": row.get("p_ruin", 0.0),
                    "dd_p95": row.get("dd_p95", 0.0),
                    "candidate_yield": row.get("candidate_yield", 0.0),
                    "z_cross_yield": row.get("z_cross_yield", 0.0),
                    "eligible_bar_yield": row.get("eligible_bar_yield", 0.0),
                    "suppression_rate": row.get("suppression_rate", 0.0),
                    "trade_count_total": row.get("trade_count_total", 0),
                    "failed_checks": ",".join(row.get("failed_checks", [])),
                }
            )

    markdown_lines = [
        "# MR Candidate Formation Sweep",
        "",
        f"- Pack: {summary.get('pack_name')}",
        f"- Engine mode: {summary.get('engine_mode')}",
        f"- Allocator policy: {summary.get('allocator_policy')}",
        f"- Candidates: {len(ranking)}",
        "",
        "## Formation Ranking",
        "",
        "| Rank | Label | Class | Avg R | P(target) | P(ruin) | DD p95 | Cand yield | Suppression | Failed checks |",
        "|------|-------|-------|-------|-----------|---------|--------|------------|-------------|---------------|",
    ]
    for row in ranking:
        markdown_lines.append(
            "| "
            f"{row.get('formation_rank', 0)} | {row.get('label', '')} | {row.get('formation_classification', '')} | "
            f"{_safe_float(row.get('avg_r')):.3f} | {100.0 * _safe_float(row.get('p_target_before_ruin')):.2f}% | "
            f"{100.0 * _safe_float(row.get('p_ruin')):.2f}% | {_safe_float(row.get('dd_p95')):.0f} | "
            f"{_safe_float(row.get('candidate_yield')):.3f} | {_safe_float(row.get('suppression_rate')):.3f} | "
            f"{', '.join(row.get('failed_checks', [])) or '-'} |"
        )
    (report_dir / "formation_ranking.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return summary