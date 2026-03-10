"""Attribution report for allocator-openfix candidate vs frozen baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.candidate_openfix import compare_runs, ensure_report_dir, load_run_dir, write_csv, write_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate allocator openfix attribution report.")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--output-root", default="artifacts/candidate_reports")
    args = parser.parse_args()

    baseline_run = load_run_dir(args.baseline_run)
    candidate_run = load_run_dir(args.candidate_run)
    comparison = compare_runs(baseline_run, candidate_run)
    report_dir = ensure_report_dir("allocator_openfix_attribution", Path(args.output_root))

    rows = comparison["per_session"]
    route_flip_rows = [r for r in rows if r["route_changed"]]
    orb_flip_rows = [r for r in route_flip_rows if r["candidate_route"] == "orb"]
    orb_flip_gain = round(sum(r["delta_pnl"] for r in orb_flip_rows if r["delta_pnl"] > 0), 2)
    orb_flip_loss = round(sum(r["delta_pnl"] for r in orb_flip_rows if r["delta_pnl"] < 0), 2)
    broad_improvement = sum(1 for r in rows if r["delta_pnl"] > 0)
    false_positive_rate = (comparison["false_positive_orb"] / max(len(orb_flip_rows), 1)) if orb_flip_rows else 0.0

    summary_rows = [
        {
            "baseline_run_id": comparison["baseline"]["run_id"],
            "candidate_run_id": comparison["candidate"]["run_id"],
            "route_changed_sessions": comparison["route_changed_sessions"],
            "orb_route_flips": len(orb_flip_rows),
            "orb_flip_gain": orb_flip_gain,
            "orb_flip_loss": orb_flip_loss,
            "orb_flip_net": round(orb_flip_gain + orb_flip_loss, 2),
            "broad_improvement_sessions": broad_improvement,
            "false_positive_orb": comparison["false_positive_orb"],
            "false_positive_rate": round(false_positive_rate, 4),
            "baseline_mr_trades": comparison["baseline"]["mr_trades"],
            "candidate_mr_trades": comparison["candidate"]["mr_trades"],
            "baseline_orb_trades": comparison["baseline"]["orb_trades"],
            "candidate_orb_trades": comparison["candidate"]["orb_trades"],
        }
    ]

    write_csv(report_dir / "per_session_attribution.csv", rows)
    write_csv(report_dir / "summary.csv", summary_rows)
    write_markdown(
        report_dir / "summary.md",
        [
            "# Allocator Openfix Attribution",
            "",
            f"- Baseline run: {comparison['baseline']['run_id']}",
            f"- Candidate run: {comparison['candidate']['run_id']}",
            "",
            "## Findings",
            "",
            f"- Route-changed sessions: {comparison['route_changed_sessions']}",
            f"- ORB route flips: {len(orb_flip_rows)}",
            f"- ORB flip gain: ${orb_flip_gain:,.2f}",
            f"- ORB flip loss: ${orb_flip_loss:,.2f}",
            f"- ORB flip net: ${orb_flip_gain + orb_flip_loss:,.2f}",
            f"- Sessions improved: {broad_improvement}/{len(rows)}",
            f"- False-positive ORB count: {comparison['false_positive_orb']}",
            f"- False-positive ORB rate: {false_positive_rate:.2%}",
            "",
            "## Interpretation",
            "",
            "- MR-only contribution is approximated by unchanged MR-routed sessions.",
            "- ORB-only contribution is approximated by sessions flipped to ORB under the candidate.",
            "- Route effect is captured by per-session PnL deltas on changed sessions.",
            "- Sizing effect is read indirectly from scaled trade PnL because both runs already include sizing policy application.",
        ],
    )
    print(f"Attribution outputs written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
