"""Compare frozen baseline vs allocator-openfix candidate across validation packs."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.candidate_openfix import (
    DEFAULT_ARTIFACT_ROOT,
    compare_runs,
    ensure_report_dir,
    run_pack_for_preset,
    write_csv,
    write_markdown,
)


def _parse_run_map(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Run mapping must use PACK=RUN_ID, got: {item}")
        pack, run_id = item.split("=", 1)
        mapping[pack.strip()] = run_id.strip()
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline vs allocator-openfix candidate.")
    parser.add_argument("--baseline-preset", default="mainline_combine_v1")
    parser.add_argument("--candidate-preset", default="mainline_combine_v1_1_allocator_openfix")
    parser.add_argument("--packs", nargs="+", default=["pilot_20d", "extended_60d", "trend20"])
    parser.add_argument("--baseline-run", action="append", default=[], help="PACK=RUN_ID override")
    parser.add_argument("--candidate-run", action="append", default=[], help="PACK=RUN_ID override")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--output-root", default="artifacts/candidate_reports")
    args = parser.parse_args()

    baseline_map = _parse_run_map(args.baseline_run)
    candidate_map = _parse_run_map(args.candidate_run)
    report_dir = ensure_report_dir("allocator_openfix_compare", Path(args.output_root))

    summary_rows = []
    markdown = [
        "# Allocator Openfix Comparison",
        "",
        f"- Baseline preset: {args.baseline_preset}",
        f"- Candidate preset: {args.candidate_preset}",
        "",
    ]

    for pack in args.packs:
        baseline_run = baseline_map.get(pack)
        candidate_run = candidate_map.get(pack)
        if not baseline_run:
            baseline_run = str(run_pack_for_preset(pack, args.baseline_preset, args.artifacts_root))
        if not candidate_run:
            candidate_run = str(run_pack_for_preset(pack, args.candidate_preset, args.artifacts_root))

        comparison = compare_runs(baseline_run, candidate_run)
        baseline = comparison["baseline"]
        candidate = comparison["candidate"]
        summary_rows.append(
            {
                "pack": pack,
                "baseline_run_id": baseline["run_id"],
                "candidate_run_id": candidate["run_id"],
                "baseline_final_equity": baseline["final_equity"],
                "candidate_final_equity": candidate["final_equity"],
                "delta_final_equity": round(candidate["final_equity"] - baseline["final_equity"], 2),
                "baseline_total_trades": baseline["total_trades"],
                "candidate_total_trades": candidate["total_trades"],
                "baseline_orb_sessions": baseline["orb_routed_sessions"],
                "candidate_orb_sessions": candidate["orb_routed_sessions"],
                "candidate_dd_p95": candidate["dd_p95"],
                "candidate_ruin_probability": candidate["ruin_probability"],
                "candidate_target_probability": candidate["target_probability"],
                "route_changed_sessions": comparison["route_changed_sessions"],
                "false_positive_orb": comparison["false_positive_orb"],
            }
        )
        write_csv(report_dir / f"{pack}_per_session.csv", comparison["per_session"])
        markdown.extend(
            [
                f"## {pack}",
                "",
                f"- Baseline run: {baseline['run_id']}",
                f"- Candidate run: {candidate['run_id']}",
                f"- Final equity: ${baseline['final_equity']:,.2f} → ${candidate['final_equity']:,.2f}",
                f"- Total trades: {baseline['total_trades']} → {candidate['total_trades']}",
                f"- ORB-routed sessions: {baseline['orb_routed_sessions']} → {candidate['orb_routed_sessions']}",
                f"- DD p95: ${baseline['dd_p95']:,.2f} → ${candidate['dd_p95']:,.2f}",
                f"- P(ruin): {baseline['ruin_probability']:.4f} → {candidate['ruin_probability']:.4f}",
                f"- P(target): {baseline['target_probability']:.4f} → {candidate['target_probability']:.4f}",
                f"- Route-changed sessions: {comparison['route_changed_sessions']}",
                f"- False-positive ORB sessions: {comparison['false_positive_orb']}",
                "",
            ]
        )
        print(
            f"[{pack}] equity ${baseline['final_equity']:,.0f} -> ${candidate['final_equity']:,.0f} | "
            f"ORB {baseline['orb_routed_sessions']} -> {candidate['orb_routed_sessions']} | "
            f"dd_p95 ${candidate['dd_p95']:,.0f}"
        )

    write_csv(report_dir / "summary.csv", summary_rows)
    write_markdown(report_dir / "summary.md", markdown)
    print(f"\nComparison outputs written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
