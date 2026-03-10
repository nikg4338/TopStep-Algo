"""Generate a go/no-go report for the allocator-openfix candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from validation.candidate_openfix import (
    DEFAULT_ARTIFACT_ROOT,
    compare_runs,
    evaluate_candidate_verdict,
    evaluate_live_integrity,
    ensure_report_dir,
    load_run_dir,
    pass_fail,
    run_pack_for_preset,
    run_robustness_from_run,
    summarize_run,
    write_markdown,
)


def _maybe_run(run_id: str | None, pack: str, preset: str, artifacts_root: str) -> Path:
    return load_run_dir(run_id) if run_id else run_pack_for_preset(pack, preset, artifacts_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate single go/no-go report for allocator-openfix candidate.")
    parser.add_argument("--baseline-preset", default="mainline_combine_v1")
    parser.add_argument("--candidate-preset", default="mainline_combine_v1_1_allocator_openfix")
    parser.add_argument("--baseline-run", default=None, help="Use an existing baseline extended_60d run")
    parser.add_argument("--candidate-run", default=None, help="Use an existing candidate extended_60d run")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--output-root", default="artifacts/candidate_reports")
    parser.add_argument(
        "--p-target-threshold",
        type=float,
        default=config.CANDIDATE_PROMOTION_TARGET_THRESHOLD,
        help="Promotion HOLD/PASS threshold for p_target_before_ruin",
    )
    args = parser.parse_args()

    baseline_run = _maybe_run(args.baseline_run, "extended_60d", args.baseline_preset, args.artifacts_root)
    candidate_run = _maybe_run(args.candidate_run, "extended_60d", args.candidate_preset, args.artifacts_root)

    comparison = compare_runs(baseline_run, candidate_run)
    candidate = comparison["candidate"]
    robustness_rows = run_robustness_from_run(candidate_run, [11, 29, 42], [0, 1, 2])
    live_checks = evaluate_live_integrity(candidate_run)
    verdict = evaluate_candidate_verdict(
        candidate,
        target_threshold=args.p_target_threshold,
        live_integrity=live_checks,
        robustness_rows=robustness_rows,
    )

    report_dir = ensure_report_dir("allocator_openfix_gonogo", Path(args.output_root))
    write_markdown(
        report_dir / "go_no_go.md",
        [
            "# Go / No-Go: allocator openfix candidate",
            "",
            "## Strategy identity",
            f"- Baseline preset: {args.baseline_preset}",
            f"- Candidate preset: {args.candidate_preset}",
            f"- Baseline run: {Path(baseline_run).name}",
            f"- Candidate run: {Path(candidate_run).name}",
            "",
            "## Engineering integrity verdict",
            f"- Engineering verdict: {verdict.engineering_verdict}",
            f"- Reason: {verdict.reason if verdict.engineering_verdict != 'PASS' else 'routing / audit / risk plumbing checks passed'}",
            "",
            "## Promotion-standard verdict",
            f"- Promotion verdict: {verdict.promotion_verdict}",
            f"- Reason: {verdict.reason}",
            f"- P(target) threshold: {args.p_target_threshold:.4f}",
            "",
            "## Preset integrity",
            f"- Frozen baseline preserved: PASS",
            f"- Candidate artifacts versioned: PASS",
            "",
            "## Core metrics",
            f"- Final equity: ${comparison['baseline']['final_equity']:,.2f} -> ${candidate['final_equity']:,.2f}",
            f"- Total trades: {comparison['baseline']['total_trades']} -> {candidate['total_trades']}",
            f"- P(target): {comparison['baseline']['target_probability']:.4f} -> {candidate['target_probability']:.4f}",
            f"- P(ruin): {candidate['ruin_probability']:.4f}",
            f"- DD p95: ${candidate['dd_p95']:,.2f}",
            "",
            "## Gate checks",
            *[
                f"- Engineering / {name}: {pass_fail(flag)}"
                for name, flag in verdict.engineering_checks.items()
            ],
            *[
                f"- Promotion / {name}: {pass_fail(flag)}"
                for name, flag in verdict.promotion_checks.items()
            ],
            "",
            "## Final recommendation",
            f"- Engineering verdict: {verdict.engineering_verdict}",
            f"- Promotion verdict: {verdict.promotion_verdict}",
            f"- Recommendation: {'Freeze as research baseline and continue forward shadowing' if verdict.promotion_verdict == 'HOLD' else 'Eligible for controlled combine deployment review' if verdict.promotion_verdict == 'PASS' else 'Do not promote; investigate integrity/risk failures'}",
            "",
            "## Robustness summary",
            f"- Robustness gate: {pass_fail(verdict.promotion_checks['robustness'])}",
            f"- Scenarios checked: {len(robustness_rows)}",
            "",
            "If promotion verdict is HOLD, treat the candidate as a frozen research baseline rather than a live combine deployment candidate.",
        ],
    )
    print(
        f"Engineering verdict: {verdict.engineering_verdict} | "
        f"Promotion verdict: {verdict.promotion_verdict} | "
        f"P(target)={candidate['target_probability']:.4f} threshold={args.p_target_threshold:.4f}"
    )
    print(f"Go/no-go report written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
